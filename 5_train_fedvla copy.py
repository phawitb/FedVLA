# 5_train_fedvla.py
#
# FedVLA (Dual-Gating MoE + EDA aggregation) training script
# - Uses prepared/prepared_dataset.npz (v_feats, states, actions, task_ids, train_idx, val_idx)
# - Same logging style as centralized/fedavg: CSV per round + JSONL detailed steps
# - Saves best_model.pt + last_model.pt
#
# Note:
# This version matches your V3 pipeline (no language yet; v_feats are CLIP vision features).
# If you later add language (t_feats), we can extend FedVLA_Model similarly.

import os
import json
import time
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_csv_header(path: str, header: list):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")


def append_csv_row(path: str, header: list, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join([str(row.get(k, "")) for k in header]) + "\n")


def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def tensor_stats(x: torch.Tensor):
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.data.norm(2).item()
        total += g * g
    return float(total ** 0.5)


def cosine_similarity_matrix(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    V: (N, K) nonnegative counts
    returns S: (N, N) cosine sim in [0,1]
    """
    N = V.shape[0]
    norms = np.linalg.norm(V, axis=1, keepdims=True) + eps
    Vn = V / norms
    S = Vn @ Vn.T
    S = np.clip(S, 0.0, 1.0)
    if N == 1:
        S[0, 0] = 1.0
    return S


# =========================
# DGMoE (Dual Gating MoE)
# =========================
class DGMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        lambda_scale: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.lambda_scale = float(lambda_scale)

        # token-side gate
        self.Wt = nn.Linear(d_model, n_experts, bias=False)
        self.Wgt = nn.Linear(n_experts, n_experts, bias=False)

        # expert-side gate thresholds
        self.We_logits = nn.Parameter(torch.zeros(n_experts))

        # experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: torch.Tensor, prev_logits: torch.Tensor = None):
        """
        x: (B, T, D)
        prev_logits: (B, T, K) or None

        returns:
          y: (B, T, D)
          logits: (B, T, K)  (detached for residual)
          sel_counts: (K,) number of active expert decisions in batch
          avg_density: scalar (#active experts per token averaged)
        """
        B, T, D = x.shape
        logits = self.Wt(x)  # (B,T,K)
        if prev_logits is not None:
            logits = logits + self.Wgt(prev_logits)

        st = torch.softmax(logits, dim=-1)

        We = torch.sigmoid(self.We_logits).view(1, 1, self.n_experts)  # (1,1,K)
        mask = (st > (self.lambda_scale * We)).float()  # (B,T,K)
        g = st * mask  # (B,T,K)

        sel_counts = mask.sum(dim=(0, 1))  # (K,)
        avg_density = float(mask.sum(dim=-1).mean().item())

        y = torch.zeros_like(x)
        for k, Ek in enumerate(self.experts):
            outk = Ek(x)
            wk = g[..., k].unsqueeze(-1)
            y = y + wk * outk

        denom = g.sum(dim=-1, keepdim=True)
        y = y / (denom + 1e-6)

        return y, logits.detach(), sel_counts.detach(), avg_density


class TrunkBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_experts: int, d_ff: int, dropout: float, lambda_scale: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.dgm = DGMoE(d_model=d_model, n_experts=n_experts, d_ff=d_ff, dropout=dropout, lambda_scale=lambda_scale)

    def forward(self, x: torch.Tensor, prev_logits: torch.Tensor = None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h2 = self.ln2(x)
        y, logits, sel_counts, avg_density = self.dgm(h2, prev_logits=prev_logits)
        x = x + y
        return x, logits, sel_counts, avg_density


# =========================
# FedVLA Model
# - trunk is global (aggregated)
# - stem + head are per-client (personalized state kept)
# =========================
class FedVLA_Model(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        state_dim: int,
        action_dim: int,
        d_model: int = 512,
        n_tokens_vision: int = 4,
        n_tokens_state: int = 2,
        n_layers: int = 6,
        n_heads: int = 8,
        n_experts: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        lambda_scale: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tokens_vision = n_tokens_vision
        self.n_tokens_state = n_tokens_state
        self.seq_len = n_tokens_vision + n_tokens_state

        # Stem (personal)
        self.vision_to_tokens = nn.Sequential(
            nn.Linear(vision_dim, d_model * n_tokens_vision),
            nn.LayerNorm(d_model * n_tokens_vision),
        )
        self.state_to_tokens = nn.Sequential(
            nn.Linear(state_dim, d_model * n_tokens_state),
            nn.LayerNorm(d_model * n_tokens_state),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Trunk (global)
        self.trunk = nn.ModuleList(
            [
                TrunkBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_experts=n_experts,
                    d_ff=d_ff,
                    dropout=dropout,
                    lambda_scale=lambda_scale,
                )
                for _ in range(n_layers)
            ]
        )
        self.trunk_norm = nn.LayerNorm(d_model)

        # Head (personal)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, v: torch.Tensor, s: torch.Tensor):
        """
        returns:
          pred: (B, action_dim)
          sel_mat: (L, K) selection counts
          densities: list length L (avg_density per layer)
        """
        B = v.shape[0]
        vt = self.vision_to_tokens(v).view(B, self.n_tokens_vision, self.d_model)
        st = self.state_to_tokens(s).view(B, self.n_tokens_state, self.d_model)
        x = torch.cat([vt, st], dim=1)
        x = x + self.pos_emb

        L = len(self.trunk)
        K = self.trunk[0].dgm.n_experts
        sel_mat = torch.zeros(L, K, device=x.device)
        densities = []

        prev_logits = None
        for li, blk in enumerate(self.trunk):
            x, logits, sel_counts, avg_density = blk(x, prev_logits=prev_logits)
            prev_logits = logits
            sel_mat[li] += sel_counts.to(x.device)
            densities.append(avg_density)

        x = self.trunk_norm(x)
        pooled = x.mean(dim=1)
        pred = self.head(pooled)
        return pred, sel_mat, densities

    # --- split helpers ---
    def get_trunk_state(self) -> Dict[str, torch.Tensor]:
        trunk_state = {}
        for k, v in self.state_dict().items():
            if k.startswith("trunk.") or k.startswith("trunk_norm."):
                trunk_state[k] = v
        return trunk_state

    def set_trunk_state(self, trunk_state: Dict[str, torch.Tensor]):
        sd = self.state_dict()
        for k, v in trunk_state.items():
            if k in sd:
                sd[k].copy_(v)
        self.load_state_dict(sd, strict=True)

    def get_personal_state(self) -> Dict[str, torch.Tensor]:
        pers = {}
        for k, v in self.state_dict().items():
            if not (k.startswith("trunk.") or k.startswith("trunk_norm.")):
                pers[k] = v
        return pers

    def set_personal_state(self, personal_state: Dict[str, torch.Tensor]):
        sd = self.state_dict()
        for k, v in personal_state.items():
            if k in sd:
                sd[k].copy_(v)
        self.load_state_dict(sd, strict=True)


# =========================
# Config
# =========================
@dataclass
class CFG:
    prepared_path: str = "prepared/prepared_dataset.npz"

    out_root: str = "runs_fedvla"
    seed: int = 42

    rounds: int = 50
    clients_per_round: int = 3
    local_epochs: int = 3
    batch_size: int = 256

    lr: float = 5e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # DGMoE
    n_experts: int = 4
    lambda_scale: float = 0.5
    d_ff: int = 2048

    # Model
    d_model: int = 512
    n_tokens_vision: int = 4
    n_tokens_state: int = 2
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1

    # Logging
    log_every_steps: int = 50


# =========================
# Loss and Eval
# =========================
def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, beta=delta, reduction="mean")


@torch.no_grad()
def evaluate_client(model: FedVLA_Model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    mse_list, mae_list, hub_list = [], [], []
    sat_list = []
    dens_list = []

    for v, s, a in loader:
        v, s, a = v.to(device), s.to(device), a.to(device)
        pred, _, densities = model(v, s)

        mse = F.mse_loss(pred, a, reduction="mean")
        mae = F.l1_loss(pred, a, reduction="mean")
        hub = huber_loss(pred, a)

        sat = ((pred.abs() > 0.98).float().mean()).item()
        avg_dens = float(np.mean(densities)) if len(densities) else 0.0

        mse_list.append(mse.item())
        mae_list.append(mae.item())
        hub_list.append(hub.item())
        sat_list.append(sat)
        dens_list.append(avg_dens)

    return {
        "mse": float(np.mean(mse_list)) if mse_list else float("nan"),
        "mae": float(np.mean(mae_list)) if mae_list else float("nan"),
        "huber": float(np.mean(hub_list)) if hub_list else float("nan"),
        "sat": float(np.mean(sat_list)) if sat_list else float("nan"),
        "avg_density": float(np.mean(dens_list)) if dens_list else float("nan"),
    }


# =========================
# Data splits per client
# =========================
def build_client_indices(task_ids: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray):
    clients = sorted(list(set(task_ids[train_idx].tolist())))
    train_map = {}
    val_map = {}
    for cid in clients:
        train_map[int(cid)] = train_idx[task_ids[train_idx] == cid]
        val_map[int(cid)] = val_idx[task_ids[val_idx] == cid]
    return clients, train_map, val_map


# =========================
# EDA Aggregation (Algorithm 2)
# =========================
def eda_aggregate_trunk(
    client_trunks: Dict[int, Dict[str, torch.Tensor]],
    client_sel_mats: Dict[int, np.ndarray],  # cid -> (L,K) counts
    layer_prefixes: List[str],
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    cids = sorted(client_trunks.keys())
    N = len(cids)
    if N == 0:
        raise RuntimeError("No client trunks to aggregate")

    L, K = client_sel_mats[cids[0]].shape

    W_layer_client = np.zeros((L, N), dtype=np.float64)

    for l in range(L):
        V = np.stack([client_sel_mats[cid][l] for cid in cids], axis=0).astype(np.float64)  # (N,K)
        S = cosine_similarity_matrix(V, eps=eps)  # (N,N)
        denom = np.sum(S)
        if denom <= 0:
            W_layer_client[l] = 1.0 / N
        else:
            W_layer_client[l] = np.sum(S, axis=1) / denom

    any_trunk = client_trunks[cids[0]]
    agg: Dict[str, torch.Tensor] = {}

    # trunk_norm uniform
    trunk_norm_keys = [k for k in any_trunk.keys() if k.startswith("trunk_norm.")]
    for k in trunk_norm_keys:
        acc = None
        for cid in cids:
            t = client_trunks[cid][k].detach().float()
            acc = t / N if acc is None else acc + t / N
        agg[k] = acc

    # layer-wise
    for l in range(L):
        pref = layer_prefixes[l]
        layer_keys = [k for k in any_trunk.keys() if k.startswith(pref)]
        for k in layer_keys:
            acc = None
            for i, cid in enumerate(cids):
                w = float(W_layer_client[l, i])
                t = client_trunks[cid][k].detach().float()
                acc = t * w if acc is None else acc + t * w
            agg[k] = acc

    return agg


# =========================
# Client local train
# =========================
def client_local_train(
    cid: int,
    model: FedVLA_Model,
    V: np.ndarray,
    S: np.ndarray,
    A: np.ndarray,
    idx_train: np.ndarray,
    cfg: CFG,
    device: torch.device,
    jsonl_step: str,
    round_idx: int,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ds = TensorDataset(
        torch.from_numpy(V[idx_train].astype(np.float32)),
        torch.from_numpy(S[idx_train].astype(np.float32)),
        torch.from_numpy(A[idx_train].astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model.train()
    mse_list, hub_list = [], []
    sel_accum = None
    dens_accum = []

    local_step = 0
    last_gn = 0.0

    for ep in range(1, cfg.local_epochs + 1):
        for step_in_epoch, (bv, bs, ba) in enumerate(loader, start=1):
            bv, bs, ba = bv.to(device), bs.to(device), ba.to(device)

            pred, sel_mat, densities = model(bv, bs)
            loss = huber_loss(pred, ba)

            optimizer.zero_grad()
            loss.backward()

            last_gn = grad_global_norm(model)
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            mse = F.mse_loss(pred, ba, reduction="mean").item()
            mse_list.append(mse)
            hub_list.append(loss.item())

            if sel_accum is None:
                sel_accum = sel_mat.detach().cpu().numpy().astype(np.float64)
            else:
                sel_accum += sel_mat.detach().cpu().numpy().astype(np.float64)
            dens_accum.append(float(np.mean(densities)) if len(densities) else 0.0)

            local_step += 1
            if cfg.log_every_steps and (local_step % cfg.log_every_steps == 0):
                append_jsonl(
                    jsonl_step,
                    {
                        "type": "client_step",
                        "round": round_idx,
                        "client_id": cid,
                        "local_epoch": ep,
                        "step_in_epoch": step_in_epoch,
                        "local_step": local_step,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "loss_huber": float(loss.item()),
                        "train_mse": float(mse),
                        "grad_norm": float(last_gn),
                        "pred": tensor_stats(pred.detach()),
                        "act": tensor_stats(ba.detach()),
                        "avg_density_batch": float(np.mean(densities)) if len(densities) else 0.0,
                        "num_samples_client": int(len(ds)),
                    },
                )

    if sel_accum is None:
        sel_accum = np.zeros((cfg.n_layers, cfg.n_experts), dtype=np.float64)

    return {
        "client_id": cid,
        "n_samples": int(len(ds)),
        "train_mse": float(np.mean(mse_list)) if mse_list else float("nan"),
        "train_huber": float(np.mean(hub_list)) if hub_list else float("nan"),
        "last_grad_norm": float(last_gn),
        "sel_mat": sel_accum,
        "avg_density": float(np.mean(dens_accum)) if dens_accum else 0.0,
        "personal_state": {k: v.detach().cpu() for k, v in model.get_personal_state().items()},
        "trunk_state": {k: v.detach().cpu() for k, v in model.get_trunk_state().items()},
    }


# =========================
# Checkpoint
# =========================
def save_ckpt(
    path: str,
    global_trunk: Dict[str, torch.Tensor],
    client_personal: Dict[int, Dict[str, torch.Tensor]],
    round_idx: int,
    best_val: float,
    cfg: CFG,
):
    torch.save(
        {
            "round": round_idx,
            "best_val": best_val,
            "global_trunk_state": global_trunk,
            "client_personal_state": client_personal,
            "cfg": cfg.__dict__,
        },
        path,
    )


# =========================
# Main
# =========================
def main():
    cfg = CFG()
    set_seed(cfg.seed)
    device = get_device()

    if not os.path.exists(cfg.prepared_path):
        raise FileNotFoundError("prepared dataset not found. Run 2_load_prepare_dataset.py first.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg.out_root, f"fedvla_{run_id}")
    ensure_dir(out_dir)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    data = np.load(cfg.prepared_path, allow_pickle=True)
    V = data["v_feats"].astype(np.float32)
    S = data["states"].astype(np.float32)
    A = data["actions"].astype(np.float32)
    task_ids = data["task_ids"].astype(np.int64)
    tr_idx = data["train_idx"].astype(np.int64)
    va_idx = data["val_idx"].astype(np.int64)

    task_map = {}
    if "task_map_json" in data.files:
        task_map = json.loads(str(data["task_map_json"][0]))

    clients, train_map, val_map = build_client_indices(task_ids, tr_idx, va_idx)
    if len(clients) == 0:
        raise RuntimeError("No clients found from task_ids/train_idx.")

    stats = {
        "V": {"shape": list(V.shape), "mean": float(V.mean()), "std": float(V.std()), "min": float(V.min()), "max": float(V.max())},
        "S": {"shape": list(S.shape), "mean": float(S.mean()), "std": float(S.std()), "min": float(S.min()), "max": float(S.max())},
        "A": {"shape": list(A.shape), "mean": float(A.mean()), "std": float(A.std()), "min": float(A.min()), "max": float(A.max())},
        "num_train": int(len(tr_idx)),
        "num_val": int(len(va_idx)),
        "num_clients_total": int(len(clients)),
        "task_map": task_map,
        "note": "IOSP skipped (prepared v_feats used). DGMoE + EDA implemented.",
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    template = FedVLA_Model(
        vision_dim=V.shape[1],
        state_dim=S.shape[1],
        action_dim=A.shape[1],
        d_model=cfg.d_model,
        n_tokens_vision=cfg.n_tokens_vision,
        n_tokens_state=cfg.n_tokens_state,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_experts=cfg.n_experts,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        lambda_scale=cfg.lambda_scale,
    ).to(device)

    global_trunk = {k: v.detach().cpu() for k, v in template.get_trunk_state().items()}

    client_personal: Dict[int, Dict[str, torch.Tensor]] = {}
    for cid in clients:
        client_personal[int(cid)] = {k: v.detach().cpu() for k, v in template.get_personal_state().items()}

    csv_round = os.path.join(out_dir, "metrics_round.csv")
    jsonl_step = os.path.join(out_dir, "metrics_step.jsonl")
    round_header = [
        "round",
        "selected_clients",
        "avg_client_train_mse",
        "avg_client_train_huber",
        "avg_client_density",
        "val_mse_avg",
        "val_huber_avg",
        "val_sat_avg",
        "time_sec_round",
    ]
    write_csv_header(csv_round, round_header)

    layer_prefixes = [f"trunk.{l}." for l in range(cfg.n_layers)]

    best_val = float("inf")

    for r in range(1, cfg.rounds + 1):
        t0 = time.time()

        k = min(cfg.clients_per_round, len(clients))
        selected = random.sample(clients, k=k)

        client_trunks: Dict[int, Dict[str, torch.Tensor]] = {}
        client_sel_mats: Dict[int, np.ndarray] = {}
        train_mse_list, train_hub_list, dens_list = [], [], []

        for cid in selected:
            m = FedVLA_Model(
                vision_dim=V.shape[1],
                state_dim=S.shape[1],
                action_dim=A.shape[1],
                d_model=cfg.d_model,
                n_tokens_vision=cfg.n_tokens_vision,
                n_tokens_state=cfg.n_tokens_state,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                n_experts=cfg.n_experts,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                lambda_scale=cfg.lambda_scale,
            ).to(device)

            m.set_trunk_state(global_trunk)
            m.set_personal_state(client_personal[int(cid)])

            res = client_local_train(
                cid=int(cid),
                model=m,
                V=V, S=S, A=A,
                idx_train=train_map[int(cid)],
                cfg=cfg,
                device=device,
                jsonl_step=jsonl_step,
                round_idx=r,
            )

            client_personal[int(cid)] = res["personal_state"]
            client_trunks[int(cid)] = res["trunk_state"]
            client_sel_mats[int(cid)] = res["sel_mat"]

            train_mse_list.append(res["train_mse"])
            train_hub_list.append(res["train_huber"])
            dens_list.append(res["avg_density"])

        new_global_trunk = eda_aggregate_trunk(
            client_trunks=client_trunks,
            client_sel_mats=client_sel_mats,
            layer_prefixes=layer_prefixes,
        )
        global_trunk = {k: v.detach().cpu() for k, v in new_global_trunk.items()}

        val_mse_all, val_hub_all, val_sat_all = [], [], []
        for cid in clients:
            idxv = val_map[int(cid)]
            if idxv is None or len(idxv) == 0:
                continue

            m = FedVLA_Model(
                vision_dim=V.shape[1],
                state_dim=S.shape[1],
                action_dim=A.shape[1],
                d_model=cfg.d_model,
                n_tokens_vision=cfg.n_tokens_vision,
                n_tokens_state=cfg.n_tokens_state,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                n_experts=cfg.n_experts,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                lambda_scale=cfg.lambda_scale,
            ).to(device)
            m.set_trunk_state(global_trunk)
            m.set_personal_state(client_personal[int(cid)])

            ds_val = TensorDataset(
                torch.from_numpy(V[idxv].astype(np.float32)),
                torch.from_numpy(S[idxv].astype(np.float32)),
                torch.from_numpy(A[idxv].astype(np.float32)),
            )
            ld_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
            met = evaluate_client(m, ld_val, device)

            val_mse_all.append(met["mse"])
            val_hub_all.append(met["huber"])
            val_sat_all.append(met["sat"])

        val_mse_avg = float(np.mean(val_mse_all)) if val_mse_all else float("nan")
        val_hub_avg = float(np.mean(val_hub_all)) if val_hub_all else float("nan")
        val_sat_avg = float(np.mean(val_sat_all)) if val_sat_all else float("nan")

        row = {
            "round": r,
            "selected_clients": "|".join(map(str, selected)),
            "avg_client_train_mse": float(np.mean(train_mse_list)) if train_mse_list else float("nan"),
            "avg_client_train_huber": float(np.mean(train_hub_list)) if train_hub_list else float("nan"),
            "avg_client_density": float(np.mean(dens_list)) if dens_list else float("nan"),
            "val_mse_avg": val_mse_avg,
            "val_huber_avg": val_hub_avg,
            "val_sat_avg": val_sat_avg,
            "time_sec_round": round(time.time() - t0, 3),
        }
        append_csv_row(csv_round, round_header, row)

        print(f"Round {r:03d} | val_mse {val_mse_avg:.6f} | clients {selected}")

        if val_mse_avg < best_val:
            best_val = val_mse_avg
            save_ckpt(
                os.path.join(out_dir, "best_model.pt"),
                global_trunk=global_trunk,
                client_personal=client_personal,
                round_idx=r,
                best_val=best_val,
                cfg=cfg,
            )

        save_ckpt(
            os.path.join(out_dir, "last_model.pt"),
            global_trunk=global_trunk,
            client_personal=client_personal,
            round_idx=r,
            best_val=best_val,
            cfg=cfg,
        )

    with open(os.path.join(out_dir, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_mse_avg": best_val,
                "rounds": cfg.rounds,
                "num_clients_total": int(len(clients)),
                "clients_per_round": int(min(cfg.clients_per_round, len(clients))),
                "notes": {
                    "IOSP": "skipped (using prepared v_feats)",
                    "DGMoE": "implemented",
                    "EDA": "implemented (layer-wise)",
                    "aggregate": "trunk only; stem+head personalized",
                },
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()