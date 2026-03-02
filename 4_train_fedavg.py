# 4_train_fedavg.py

import os
import json
import time
import math
import random
from dataclasses import dataclass
from datetime import datetime

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


def make_cosine_warmup_lr(base_lr, min_lr, warmup_steps, total_steps, step):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr + (base_lr - min_lr) * cos


# =========================
# Model (same backbone as centralized, now with text)
# =========================
class HPTLikePolicy(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        state_dim: int,
        action_dim: int,
        d_model: int = 512,
        n_tokens_vision: int = 4,
        n_tokens_text: int = 2,
        n_tokens_state: int = 2,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tokens_vision = n_tokens_vision
        self.n_tokens_text = n_tokens_text
        self.n_tokens_state = n_tokens_state
        self.seq_len = n_tokens_vision + n_tokens_text + n_tokens_state

        self.vision_to_tokens = nn.Sequential(
            nn.Linear(vision_dim, d_model * n_tokens_vision),
            nn.LayerNorm(d_model * n_tokens_vision),
        )
        self.text_to_tokens = nn.Sequential(
            nn.Linear(text_dim, d_model * n_tokens_text),
            nn.LayerNorm(d_model * n_tokens_text),
        )
        self.state_to_tokens = nn.Sequential(
            nn.Linear(state_dim, d_model * n_tokens_state),
            nn.LayerNorm(d_model * n_tokens_state),
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.trunk = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.trunk_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, v: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        B = v.shape[0]
        vt = self.vision_to_tokens(v).view(B, self.n_tokens_vision, self.d_model)
        tt = self.text_to_tokens(t).view(B, self.n_tokens_text, self.d_model)
        st = self.state_to_tokens(s).view(B, self.n_tokens_state, self.d_model)

        x = torch.cat([vt, tt, st], dim=1)
        x = x + self.pos_emb
        x = self.trunk(x)
        x = self.trunk_norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# =========================
# Config
# =========================
@dataclass
class CFG:
    prepared_path: str = "prepared/prepared_dataset.npz"

    out_root: str = "runs_fedavg"
    seed: int = 42

    # FedAvg
    rounds: int = 50
    clients_per_round: int = 3
    local_epochs: int = 1
    batch_size: int = 256

    # Optimizer
    lr: float = 5e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Scheduler (per client local steps)
    use_scheduler: bool = True
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1

    # Logging
    log_every_steps: int = 50

    # Model
    d_model: int = 512
    n_tokens_vision: int = 4
    n_tokens_text: int = 2
    n_tokens_state: int = 2
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    norm_first: bool = True


def save_ckpt(path: str, model: nn.Module, optim_state: dict, round_idx: int, best_val: float, cfg: CFG):
    torch.save(
        {
            "round": round_idx,
            "best_val": best_val,
            "model_state": model.state_dict(),
            "server_optim_state": optim_state,
            "cfg": cfg.__dict__,
        },
        path,
    )


@torch.no_grad()
def evaluate_global(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    mse_list, mae_list = [], []
    sat_list = []

    for v, t, s, a in loader:
        v, t, s, a = v.to(device), t.to(device), s.to(device), a.to(device)
        pred = model(v, t, s)

        mse = F.mse_loss(pred, a, reduction="mean")
        mae = F.l1_loss(pred, a, reduction="mean")
        sat = ((pred.abs() > 0.98).float().mean()).item()

        mse_list.append(mse.item())
        mae_list.append(mae.item())
        sat_list.append(sat)

    return {
        "mse": float(np.mean(mse_list)) if mse_list else float("nan"),
        "mae": float(np.mean(mae_list)) if mae_list else float("nan"),
        "sat": float(np.mean(sat_list)) if sat_list else float("nan"),
    }


def fedavg_state_dict(state_dicts, weights):
    wsum = float(np.sum(weights))
    if wsum <= 0:
        raise ValueError("Sum of FedAvg weights must be > 0")

    avg = {}
    for k in state_dicts[0].keys():
        acc = None
        for sd, w in zip(state_dicts, weights):
            t = sd[k].detach().float()
            acc = t * (w / wsum) if acc is None else acc + t * (w / wsum)
        avg[k] = acc
    return avg


def build_client_indices(task_ids: np.ndarray, train_idx: np.ndarray):
    client_map = {}
    train_task_ids = task_ids[train_idx]
    unique_clients = np.unique(train_task_ids)
    for cid in unique_clients:
        mask = (train_task_ids == cid)
        client_map[int(cid)] = train_idx[mask]
    return client_map


def client_local_train(
    client_id: int,
    base_state: dict,
    V: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    A: np.ndarray,
    client_indices: np.ndarray,
    cfg: CFG,
    device: torch.device,
    jsonl_step_path: str,
    round_idx: int,
):
    model = HPTLikePolicy(
        vision_dim=V.shape[1],
        text_dim=T.shape[1],
        state_dim=S.shape[1],
        action_dim=A.shape[1],
        d_model=cfg.d_model,
        n_tokens_vision=cfg.n_tokens_vision,
        n_tokens_text=cfg.n_tokens_text,
        n_tokens_state=cfg.n_tokens_state,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        norm_first=cfg.norm_first,
    ).to(device)
    model.load_state_dict(base_state, strict=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Vc = V[client_indices].astype(np.float32)
    Tc = T[client_indices].astype(np.float32)
    Sc = S[client_indices].astype(np.float32)
    Ac = A[client_indices].astype(np.float32)

    ds = TensorDataset(torch.from_numpy(Vc), torch.from_numpy(Tc), torch.from_numpy(Sc), torch.from_numpy(Ac))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    steps_per_epoch = len(loader)
    total_steps = max(1, cfg.local_epochs * steps_per_epoch)
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    min_lr = cfg.lr * cfg.min_lr_ratio

    local_step = 0
    model.train()

    mse_list, mae_list = [], []
    last_gn = 0.0

    for local_ep in range(1, cfg.local_epochs + 1):
        for step_in_epoch, (bv, bt, bs, ba) in enumerate(loader, start=1):
            if cfg.use_scheduler:
                lr_now = make_cosine_warmup_lr(cfg.lr, min_lr, warmup_steps, total_steps, local_step)
                optimizer.param_groups[0]["lr"] = lr_now

            bv, bt, bs, ba = bv.to(device), bt.to(device), bs.to(device), ba.to(device)

            pred = model(bv, bt, bs)
            mse = F.mse_loss(pred, ba, reduction="mean")
            mae = F.l1_loss(pred, ba, reduction="mean")

            optimizer.zero_grad()
            mse.backward()

            last_gn = grad_global_norm(model)
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            mse_list.append(mse.item())
            mae_list.append(mae.item())

            local_step += 1

            if cfg.log_every_steps and (local_step % cfg.log_every_steps == 0):
                append_jsonl(
                    jsonl_step_path,
                    {
                        "type": "client_step",
                        "round": round_idx,
                        "client_id": client_id,
                        "local_epoch": local_ep,
                        "step_in_epoch": step_in_epoch,
                        "local_step": local_step,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "train_mse": float(mse.item()),
                        "train_mae": float(mae.item()),
                        "grad_norm": float(last_gn),
                        "pred": tensor_stats(pred.detach()),
                        "act": tensor_stats(ba.detach()),
                        "vision": tensor_stats(bv.detach()),
                        "text": tensor_stats(bt.detach()),
                        "state": tensor_stats(bs.detach()),
                        "num_samples_client": int(len(ds)),
                    },
                )

    return {
        "client_id": client_id,
        "n_samples": int(len(ds)),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "train_mse": float(np.mean(mse_list)) if mse_list else float("nan"),
        "train_mae": float(np.mean(mae_list)) if mae_list else float("nan"),
        "last_grad_norm": float(last_gn),
    }


def main():
    cfg = CFG()
    set_seed(cfg.seed)
    device = get_device()

    if not os.path.exists(cfg.prepared_path):
        raise FileNotFoundError("prepared dataset not found. Run 2_load_prepare_dataset.py first.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg.out_root, f"fedavg_{run_id}")
    ensure_dir(out_dir)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    data = np.load(cfg.prepared_path, allow_pickle=True)

    if "t_feats" not in data.files:
        raise KeyError("prepared_dataset.npz missing 't_feats'. Run 2_load_prepare_dataset.py that saves t_feats first.")

    V = data["v_feats"].astype(np.float32)
    T = data["t_feats"].astype(np.float32)
    S = data["states"].astype(np.float32)
    A = data["actions"].astype(np.float32)

    task_ids = data["task_ids"].astype(np.int64)
    tr_idx = data["train_idx"].astype(np.int64)
    va_idx = data["val_idx"].astype(np.int64)

    task_map = {}
    if "task_map_json" in data.files:
        task_map = json.loads(str(data["task_map_json"][0]))

    stats = {
        "V": {"shape": list(V.shape), "mean": float(V.mean()), "std": float(V.std()), "min": float(V.min()), "max": float(V.max())},
        "T": {"shape": list(T.shape), "mean": float(T.mean()), "std": float(T.std()), "min": float(T.min()), "max": float(T.max())},
        "S": {"shape": list(S.shape), "mean": float(S.mean()), "std": float(S.std()), "min": float(S.min()), "max": float(S.max())},
        "A": {"shape": list(A.shape), "mean": float(A.mean()), "std": float(A.std()), "min": float(A.min()), "max": float(A.max())},
        "num_train": int(len(tr_idx)),
        "num_val": int(len(va_idx)),
        "num_clients_total": int(len(np.unique(task_ids[tr_idx]))),
        "task_map": task_map,
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    V_va, T_va, S_va, A_va = V[va_idx], T[va_idx], S[va_idx], A[va_idx]
    val_ds = TensorDataset(torch.from_numpy(V_va), torch.from_numpy(T_va), torch.from_numpy(S_va), torch.from_numpy(A_va))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    client_map = build_client_indices(task_ids, tr_idx)
    all_clients = sorted(client_map.keys())
    if len(all_clients) == 0:
        raise RuntimeError("No clients found from task_ids/train_idx.")

    server_model = HPTLikePolicy(
        vision_dim=V.shape[1],
        text_dim=T.shape[1],
        state_dim=S.shape[1],
        action_dim=A.shape[1],
        d_model=cfg.d_model,
        n_tokens_vision=cfg.n_tokens_vision,
        n_tokens_text=cfg.n_tokens_text,
        n_tokens_state=cfg.n_tokens_state,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        norm_first=cfg.norm_first,
    ).to(device)

    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    csv_round = os.path.join(out_dir, "metrics_round.csv")
    jsonl_step = os.path.join(out_dir, "metrics_step.jsonl")

    round_header = [
        "round",
        "selected_clients",
        "num_samples_selected",
        "avg_client_train_mse",
        "avg_client_train_mae",
        "val_mse",
        "val_mae",
        "val_sat",
        "time_sec_round",
    ]
    write_csv_header(csv_round, round_header)

    best_val = float("inf")

    for r in range(1, cfg.rounds + 1):
        t0 = time.time()

        k = min(cfg.clients_per_round, len(all_clients))
        selected = random.sample(all_clients, k=k)

        base_state = {k: v.detach().cpu() for k, v in server_model.state_dict().items()}

        client_results = []
        for cid in selected:
            res = client_local_train(
                client_id=cid,
                base_state=base_state,
                V=V, T=T, S=S, A=A,
                client_indices=client_map[cid],
                cfg=cfg,
                device=device,
                jsonl_step_path=jsonl_step,
                round_idx=r,
            )
            client_results.append(res)

        state_dicts = [cr["state_dict"] for cr in client_results]
        weights = [cr["n_samples"] for cr in client_results]
        new_state = fedavg_state_dict(state_dicts, weights)

        server_model.load_state_dict(new_state, strict=True)

        val_metrics = evaluate_global(server_model, val_loader, device)
        val_mse, val_mae, val_sat = val_metrics["mse"], val_metrics["mae"], val_metrics["sat"]

        avg_client_train_mse = float(np.mean([cr["train_mse"] for cr in client_results]))
        avg_client_train_mae = float(np.mean([cr["train_mae"] for cr in client_results]))
        num_samples_selected = int(np.sum([cr["n_samples"] for cr in client_results]))

        row = {
            "round": r,
            "selected_clients": "|".join(map(str, selected)),
            "num_samples_selected": num_samples_selected,
            "avg_client_train_mse": avg_client_train_mse,
            "avg_client_train_mae": avg_client_train_mae,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_sat": val_sat,
            "time_sec_round": round(time.time() - t0, 3),
        }
        append_csv_row(csv_round, round_header, row)

        print(f"Round {r:03d} | val_mse {val_mse:.6f} | clients {selected}")

        if val_mse < best_val:
            best_val = val_mse
            save_ckpt(os.path.join(out_dir, "best_model.pt"), server_model, server_optimizer.state_dict(), r, best_val, cfg)

        save_ckpt(os.path.join(out_dir, "last_model.pt"), server_model, server_optimizer.state_dict(), r, best_val, cfg)

    with open(os.path.join(out_dir, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_mse": best_val,
                "rounds": cfg.rounds,
                "num_clients_total": int(len(all_clients)),
                "clients_per_round": int(min(cfg.clients_per_round, len(all_clients))),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()