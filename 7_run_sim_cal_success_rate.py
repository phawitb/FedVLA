# 7_run_sim_cal_success_rate.py
#
# Evaluate success rate in MetaWorld using the BEST checkpoint of:
# - Centralized (best_model.pt)
# - FedAvg      (best_model.pt)
# - FedVLA      (best_model.pt)  [global trunk + per-client personal states]
#
# This version uses "REAL TEXT" from prepared_dataset.npz:
#   - Reads text_inputs_json from prepared_dataset.npz (preferred)
#   - If missing, falls back to external --text_inputs JSON file
#   - Encodes text via CLIP text encoder -> t_feat
#
# Also:
# - Auto-detect whether checkpoint expects text by checking state_dict keys.
# - Auto-infer n_tokens_text from pos_emb length when not provided in cfg.
#
# Output:
#   reports/sim_eval_<timestamp>/
#     videos/<model>/<task>/ep_XXX.mp4
#     summary.json

import os
import json
import argparse
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import cv2
from PIL import Image
import mujoco
import metaworld
from transformers import CLIPProcessor, CLIPModel


# -------------------------
# Basic utils
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def latest_run_dir(root: str, prefix: str):
    if not root or not os.path.isdir(root):
        return None
    cands = []
    for d in os.listdir(root):
        if d.startswith(prefix + "_"):
            full = os.path.join(root, d)
            if os.path.isdir(full):
                cands.append(full)
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def find_ckpt(run_dir: str, ckpt_name: str = "best_model.pt"):
    if not run_dir:
        return None
    p = os.path.join(run_dir, ckpt_name)
    return p if os.path.exists(p) else None


def _normalize_text_bank(obj) -> Dict[str, List[str]]:
    """
    Normalize to:
      { task_name: [prompt1, prompt2, ...] }
    """
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in obj.items():
        if isinstance(v, list):
            vv = [str(x).strip() for x in v if str(x).strip()]
            if vv:
                out[str(k)] = vv
        elif isinstance(v, str) and v.strip():
            out[str(k)] = [v.strip()]
    return out


def load_text_inputs_json_file(path: str) -> Dict[str, List[str]]:
    """
    External JSON file fallback.
    Expected JSON:
    {
      "door-lock-v3": ["...", "...", ...],
      "drawer-close-v3": [...],
      "window-open-v3": [...]
    }
    """
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[warn] text_inputs file not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return _normalize_text_bank(obj)
    except Exception as e:
        print(f"[warn] failed to read text_inputs file: {path} err={e}")
        return {}


def load_text_inputs_from_prepared_npz(npz_path: str) -> Dict[str, List[str]]:
    """
    Preferred: read text_inputs_json from prepared_dataset.npz.

    We saved it as JSON string in npz, often using:
      np.array([json.dumps(text_inputs_dict)], dtype=object)
    So here we robustly parse it.
    """
    if not npz_path or not os.path.exists(npz_path):
        return {}

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"[warn] cannot open prepared npz: {npz_path} err={e}")
        return {}

    if "text_inputs_json" not in data.files:
        return {}

    try:
        raw = data["text_inputs_json"]
        # raw may be 0-d array, 1-d array, bytes, or str
        if isinstance(raw, np.ndarray):
            # common: shape (1,)
            if raw.shape == ():
                raw0 = raw.item()
            else:
                raw0 = raw[0]
        else:
            raw0 = raw

        if isinstance(raw0, bytes):
            raw0 = raw0.decode("utf-8", errors="ignore")

        if isinstance(raw0, str):
            obj = json.loads(raw0)
            return _normalize_text_bank(obj)

        # if already dict-like
        return _normalize_text_bank(raw0)

    except Exception as e:
        print(f"[warn] failed to parse text_inputs_json from npz: err={e}")
        return {}


# -------------------------
# CLIP feature extraction
# -------------------------
@torch.no_grad()
def frame_to_vfeat(frame_rgb: np.ndarray, clip_model: CLIPModel, clip_processor: CLIPProcessor, device: torch.device):
    """
    frame_rgb: HxWx3 uint8 RGB
    returns: (1, 512) float tensor
    """
    img = Image.fromarray(frame_rgb)
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    out = clip_model.vision_model(**inputs).pooler_output
    v = clip_model.visual_projection(out)
    return v


@torch.no_grad()
def text_to_tfeat(text: str, clip_model: CLIPModel, clip_processor: CLIPProcessor, device: torch.device):
    """
    text: string
    returns: (1, 512) float tensor
    """
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    out = clip_model.text_model(**inputs).pooler_output
    t = clip_model.text_projection(out)
    return t


# -------------------------
# Models (support optional text tokens)
# -------------------------
class HPTLikePolicy(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        state_dim: int,
        action_dim: int,
        text_dim: int = 512,
        d_model: int = 512,
        n_tokens_vision: int = 4,
        n_tokens_state: int = 2,
        n_tokens_text: int = 0,  # 0 means "no text branch"
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tokens_vision = int(n_tokens_vision)
        self.n_tokens_state = int(n_tokens_state)
        self.n_tokens_text = int(n_tokens_text)
        self.seq_len = self.n_tokens_vision + self.n_tokens_state + self.n_tokens_text

        self.vision_to_tokens = nn.Sequential(
            nn.Linear(vision_dim, d_model * self.n_tokens_vision),
            nn.LayerNorm(d_model * self.n_tokens_vision),
        )
        self.state_to_tokens = nn.Sequential(
            nn.Linear(state_dim, d_model * self.n_tokens_state),
            nn.LayerNorm(d_model * self.n_tokens_state),
        )

        if self.n_tokens_text > 0:
            self.text_to_tokens = nn.Sequential(
                nn.Linear(text_dim, d_model * self.n_tokens_text),
                nn.LayerNorm(d_model * self.n_tokens_text),
            )
        else:
            self.text_to_tokens = None

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

    def forward(self, v: torch.Tensor, s: torch.Tensor, t: torch.Tensor = None):
        """
        v: (B, vision_dim)
        s: (B, state_dim)
        t: (B, text_dim) or None
        """
        B = v.shape[0]
        vt = self.vision_to_tokens(v).view(B, self.n_tokens_vision, self.d_model)
        st = self.state_to_tokens(s).view(B, self.n_tokens_state, self.d_model)

        toks = [vt, st]
        if self.n_tokens_text > 0:
            if t is None:
                raise ValueError("Model expects text features (t), but got None.")
            tt = self.text_to_tokens(t).view(B, self.n_tokens_text, self.d_model)
            toks.append(tt)

        x = torch.cat(toks, dim=1)
        x = x + self.pos_emb
        x = self.trunk(x)
        x = self.trunk_norm(x)
        x = x.mean(dim=1)
        return self.head(x)


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
        self.n_experts = n_experts
        self.lambda_scale = float(lambda_scale)

        self.Wt = nn.Linear(d_model, n_experts, bias=False)
        self.Wgt = nn.Linear(n_experts, n_experts, bias=False)
        self.We_logits = nn.Parameter(torch.zeros(n_experts))

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
        logits = self.Wt(x)
        if prev_logits is not None:
            logits = logits + self.Wgt(prev_logits)

        st = torch.softmax(logits, dim=-1)
        We = torch.sigmoid(self.We_logits).view(1, 1, self.n_experts)
        mask = (st > (self.lambda_scale * We)).float()
        g = st * mask

        y = torch.zeros_like(x)
        for k, Ek in enumerate(self.experts):
            outk = Ek(x)
            wk = g[..., k].unsqueeze(-1)
            y = y + wk * outk

        denom = g.sum(dim=-1, keepdim=True)
        y = y / (denom + 1e-6)
        return y, logits.detach()


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
        y, logits = self.dgm(h2, prev_logits=prev_logits)
        x = x + y
        return x, logits


class FedVLA_Model(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        state_dim: int,
        action_dim: int,
        text_dim: int = 512,
        d_model: int = 512,
        n_tokens_vision: int = 4,
        n_tokens_state: int = 2,
        n_tokens_text: int = 0,
        n_layers: int = 6,
        n_heads: int = 8,
        n_experts: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        lambda_scale: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tokens_vision = int(n_tokens_vision)
        self.n_tokens_state = int(n_tokens_state)
        self.n_tokens_text = int(n_tokens_text)
        self.seq_len = self.n_tokens_vision + self.n_tokens_state + self.n_tokens_text

        # Stem (personalized)
        self.vision_to_tokens = nn.Sequential(
            nn.Linear(vision_dim, d_model * self.n_tokens_vision),
            nn.LayerNorm(d_model * self.n_tokens_vision),
        )
        self.state_to_tokens = nn.Sequential(
            nn.Linear(state_dim, d_model * self.n_tokens_state),
            nn.LayerNorm(d_model * self.n_tokens_state),
        )

        if self.n_tokens_text > 0:
            self.text_to_tokens = nn.Sequential(
                nn.Linear(text_dim, d_model * self.n_tokens_text),
                nn.LayerNorm(d_model * self.n_tokens_text),
            )
        else:
            self.text_to_tokens = None

        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Trunk (aggregated)
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

        # Head (personalized)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, v: torch.Tensor, s: torch.Tensor, t: torch.Tensor = None):
        B = v.shape[0]
        vt = self.vision_to_tokens(v).view(B, self.n_tokens_vision, self.d_model)
        st = self.state_to_tokens(s).view(B, self.n_tokens_state, self.d_model)

        toks = [vt, st]
        if self.n_tokens_text > 0:
            if t is None:
                raise ValueError("FedVLA model expects text features (t), but got None.")
            tt = self.text_to_tokens(t).view(B, self.n_tokens_text, self.d_model)
            toks.append(tt)

        x = torch.cat(toks, dim=1)
        x = x + self.pos_emb

        prev_logits = None
        for blk in self.trunk:
            x, logits = blk(x, prev_logits=prev_logits)
            prev_logits = logits

        x = self.trunk_norm(x)
        pooled = x.mean(dim=1)
        pred = self.head(pooled)
        return pred

    def get_trunk_state(self):
        trunk_state = {}
        for k, v in self.state_dict().items():
            if k.startswith("trunk.") or k.startswith("trunk_norm."):
                trunk_state[k] = v
        return trunk_state

    def set_trunk_state(self, trunk_state):
        sd = self.state_dict()
        for k, v in trunk_state.items():
            if k in sd:
                sd[k].copy_(v)
        self.load_state_dict(sd, strict=True)

    def get_personal_state(self):
        pers = {}
        for k, v in self.state_dict().items():
            if not (k.startswith("trunk.") or k.startswith("trunk_norm.")):
                pers[k] = v
        return pers

    def set_personal_state(self, personal_state):
        sd = self.state_dict()
        for k, v in personal_state.items():
            if k in sd:
                sd[k].copy_(v)
        self.load_state_dict(sd, strict=True)


# -------------------------
# MetaWorld helpers
# -------------------------
def make_env(task_name: str):
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    return ml1, env


# -------------------------
# Checkpoint -> build model (AUTO detect text + seq_len)
# -------------------------
def ckpt_has_text(model_state: dict) -> bool:
    return any(k.startswith("text_to_tokens.") for k in model_state.keys())


def infer_n_tokens_text_from_pos_emb(pos_emb: torch.Tensor, n_tokens_vision: int, n_tokens_state: int) -> int:
    seq_len = int(pos_emb.shape[1])
    n_tokens_text = seq_len - int(n_tokens_vision) - int(n_tokens_state)
    return max(0, int(n_tokens_text))


def build_hptlike_from_ckpt(
    ck: dict,
    vision_dim: int,
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> Tuple[HPTLikePolicy, bool]:
    cfg = ck.get("cfg", {}) or {}
    ms = ck["model_state"]

    d_model = int(cfg.get("d_model", 512))
    n_tokens_vision = int(cfg.get("n_tokens_vision", 4))
    n_tokens_state = int(cfg.get("n_tokens_state", 2))
    n_layers = int(cfg.get("n_layers", 6))
    n_heads = int(cfg.get("n_heads", 8))
    dropout = float(cfg.get("dropout", 0.1))
    norm_first = bool(cfg.get("norm_first", True))

    has_text = ckpt_has_text(ms)
    n_tokens_text = int(cfg.get("n_tokens_text", 0))
    if has_text:
        if n_tokens_text <= 0 and "pos_emb" in ms:
            n_tokens_text = infer_n_tokens_text_from_pos_emb(ms["pos_emb"], n_tokens_vision, n_tokens_state)
        if n_tokens_text <= 0:
            n_tokens_text = 2  # safe fallback

    model = HPTLikePolicy(
        vision_dim=vision_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        text_dim=512,
        d_model=d_model,
        n_tokens_vision=n_tokens_vision,
        n_tokens_state=n_tokens_state,
        n_tokens_text=n_tokens_text if has_text else 0,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        norm_first=norm_first,
    ).to(device)

    model.load_state_dict(ms, strict=True)
    model.eval()
    return model, has_text


def build_fedvla_from_ckpt(
    ck: dict,
    vision_dim: int,
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> Tuple[FedVLA_Model, bool]:
    cfg = ck.get("cfg", {}) or {}

    d_model = int(cfg.get("d_model", 512))
    n_tokens_vision = int(cfg.get("n_tokens_vision", 4))
    n_tokens_state = int(cfg.get("n_tokens_state", 2))
    n_layers = int(cfg.get("n_layers", 6))
    n_heads = int(cfg.get("n_heads", 8))
    dropout = float(cfg.get("dropout", 0.1))
    n_experts = int(cfg.get("n_experts", 4))
    d_ff = int(cfg.get("d_ff", 2048))
    lambda_scale = float(cfg.get("lambda_scale", 0.5))

    personal_by_taskid = ck["client_personal_state"]
    any_tid = int(next(iter(personal_by_taskid.keys())))
    any_personal = personal_by_taskid[any_tid]

    has_text = any(k.startswith("text_to_tokens.") for k in any_personal.keys())
    n_tokens_text = int(cfg.get("n_tokens_text", 0))
    if has_text:
        if n_tokens_text <= 0 and "pos_emb" in any_personal:
            n_tokens_text = infer_n_tokens_text_from_pos_emb(any_personal["pos_emb"], n_tokens_vision, n_tokens_state)
        if n_tokens_text <= 0:
            n_tokens_text = 2

    model = FedVLA_Model(
        vision_dim=vision_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        text_dim=512,
        d_model=d_model,
        n_tokens_vision=n_tokens_vision,
        n_tokens_state=n_tokens_state,
        n_tokens_text=n_tokens_text if has_text else 0,
        n_layers=n_layers,
        n_heads=n_heads,
        n_experts=n_experts,
        d_ff=d_ff,
        dropout=dropout,
        lambda_scale=lambda_scale,
    ).to(device)

    model.set_trunk_state(ck["global_trunk_state"])
    model.eval()
    return model, has_text


# -------------------------
# Text selection + caching
# -------------------------
def choose_prompt(task_name: str, text_bank: Dict[str, List[str]]) -> str:
    lst = text_bank.get(task_name, None)
    if lst and len(lst) > 0:
        return random.choice(lst)
    return f"Perform task: {task_name}"


def build_text_cache(
    tasks: List[str],
    text_bank: Dict[str, List[str]],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Pre-encode one prompt per task (deterministic with seed) to avoid re-encoding every episode.
    Returns:
      task_name -> tfeat (1,512)
    """
    rng = random.Random(seed)
    cache: Dict[str, torch.Tensor] = {}
    for task in tasks:
        prompts = text_bank.get(task, None)
        if prompts and len(prompts) > 0:
            prompt = prompts[rng.randrange(len(prompts))]
        else:
            prompt = f"Perform task: {task}"
        cache[task] = text_to_tfeat(prompt, clip_model, clip_processor, device)
        # store prompt text too as attribute-like (separately)
    return cache


# -------------------------
# Evaluation core
# -------------------------
def run_eval_for_model(
    model_tag: str,
    model_obj,
    tasks: list,
    episodes: int,
    max_steps: int,
    camera: str,
    out_dir: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    seed: int = 0,
    model_uses_text: bool = False,
    text_bank: Optional[Dict[str, List[str]]] = None,
    text_cache: Optional[Dict[str, torch.Tensor]] = None,
    fedvla_personal_by_taskid: dict = None,
    task_name_to_id: dict = None,
):
    random.seed(seed)
    np.random.seed(seed)

    results = {"model": model_tag, "tasks": {}}
    vid_root = os.path.join(out_dir, "videos", model_tag)
    ensure_dir(vid_root)

    text_bank = text_bank or {}
    text_cache = text_cache or {}

    for task_name in tasks:
        ml1, env = make_env(task_name)
        renderer = mujoco.Renderer(env.model, height=480, width=480)

        task_out_dir = os.path.join(vid_root, task_name)
        ensure_dir(task_out_dir)

        # FedVLA: set personal state per task
        if model_tag.lower() == "fedvla":
            if task_name_to_id is None or fedvla_personal_by_taskid is None:
                raise RuntimeError("FedVLA requires task_name_to_id and fedvla_personal_by_taskid.")
            tid = int(task_name_to_id[task_name])
            if tid not in fedvla_personal_by_taskid:
                raise KeyError(f"No personal state for task_id={tid} (task={task_name}).")
            model_obj.set_personal_state(fedvla_personal_by_taskid[tid])

        # Get cached text feature for task (if needed)
        task_tfeat = None
        task_prompt = None
        if model_uses_text:
            # deterministic per episode selection is OK, but we prefer stable per-task here
            # if cache exists use it, else encode one now
            if task_name in text_cache:
                task_tfeat = text_cache[task_name]
                # we still log a prompt chosen from bank (for trace)
                task_prompt = choose_prompt(task_name, text_bank)
            else:
                task_prompt = choose_prompt(task_name, text_bank)
                task_tfeat = text_to_tfeat(task_prompt, clip_model, clip_processor, device)

        successes = 0
        ep_summaries = []

        for ep in range(1, episodes + 1):
            env.set_task(random.choice(ml1.train_tasks))
            obs, _ = env.reset()

            video_path = os.path.join(task_out_dir, f"ep_{ep:03d}.mp4")
            vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (480, 480))
            if not vw.isOpened():
                raise RuntimeError(
                    f"cv2.VideoWriter failed to open: {video_path}. "
                    "Check OpenCV build / codec support."
                )

            ep_success = False
            min_dist = float("inf")

            for t in range(1, max_steps + 1):
                renderer.update_scene(env.data, camera=camera)
                frame = renderer.render()  # RGB
                vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                vfeat = frame_to_vfeat(frame, clip_model, clip_processor, device)
                st = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    if model_uses_text:
                        act = model_obj(vfeat, st, task_tfeat).detach().cpu().numpy()[0]
                    else:
                        act = model_obj(vfeat, st).detach().cpu().numpy()[0]

                obs, _, _, _, info = env.step(act)

                d = info.get("obj_to_target", None)
                if d is not None:
                    min_dist = min(min_dist, float(d))

                sflag = info.get("success", False)
                if isinstance(sflag, (int, float, np.number)):
                    sflag = (float(sflag) > 0.0)
                if bool(sflag):
                    ep_success = True
                    break

            vw.release()

            if ep_success:
                successes += 1

            ep_summaries.append(
                {
                    "episode": ep,
                    "success": bool(ep_success),
                    "min_dist": None if min_dist == float("inf") else float(min_dist),
                    "video": os.path.relpath(video_path, out_dir),
                    "prompt_used": task_prompt if model_uses_text else None,
                }
            )

            print(
                f"{model_tag} | {task_name} | ep {ep:03d} | success {int(ep_success)} | "
                f"min_dist {ep_summaries[-1]['min_dist']} | text={int(model_uses_text)}"
            )

        success_rate = (successes / episodes) * 100.0
        results["tasks"][task_name] = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": success_rate,
            "details": ep_summaries,
        }

        renderer.close()
        env.close()

    all_rates = [results["tasks"][t]["success_rate"] for t in results["tasks"]]
    results["overall"] = {"avg_success_rate": float(np.mean(all_rates)) if all_rates else 0.0, "tasks": len(all_rates)}
    return results


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_path", type=str, default="prepared/prepared_dataset.npz")

    ap.add_argument("--central_run", type=str, default=None)
    ap.add_argument("--fedavg_run", type=str, default=None)
    ap.add_argument("--fedvla_run", type=str, default=None)

    ap.add_argument("--central_ckpt", type=str, default=None)
    ap.add_argument("--fedavg_ckpt", type=str, default=None)
    ap.add_argument("--fedvla_ckpt", type=str, default=None)

    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=250)
    ap.add_argument("--camera", type=str, default="corner2")
    ap.add_argument("--tasks", type=str, default=None, help="comma-separated task names; default uses task_map from prepared")

    ap.add_argument("--text_inputs", type=str, default=None, help="Optional JSON mapping task_name -> list of prompts (fallback).")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    device = get_device()
    print("device:", device)

    # Output folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join("reports", f"sim_eval_{ts}")
    ensure_dir(out_dir)

    # Load prepared data (dims + task map + REAL text_inputs_json)
    if not os.path.exists(args.prepared_path):
        raise FileNotFoundError(f"prepared dataset not found: {args.prepared_path}")
    pdata = np.load(args.prepared_path, allow_pickle=True)

    V = pdata["v_feats"].astype(np.float32)
    S = pdata["states"].astype(np.float32)
    A = pdata["actions"].astype(np.float32)
    vision_dim = int(V.shape[1])
    state_dim = int(S.shape[1])
    action_dim = int(A.shape[1])

    task_name_to_id = {}
    if "task_map_json" in pdata.files:
        try:
            task_name_to_id = json.loads(str(pdata["task_map_json"][0]))
        except Exception:
            task_name_to_id = {}

    # Decide tasks
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = list(task_name_to_id.keys()) if task_name_to_id else ["door-lock-v3", "drawer-close-v3", "window-open-v3"]

    # Load REAL text bank (priority: prepared npz -> external file)
    text_bank = load_text_inputs_from_prepared_npz(args.prepared_path)
    if not text_bank:
        # fallback external file if provided
        if args.text_inputs:
            text_bank = load_text_inputs_json_file(args.text_inputs)
        else:
            text_bank = {}

    # Auto pick latest runs
    central_run = args.central_run or latest_run_dir("runs_centralized", "central")
    fedavg_run = args.fedavg_run or latest_run_dir("runs_fedavg", "fedavg")
    fedvla_run = args.fedvla_run or latest_run_dir("runs_fedvla", "fedvla")

    # Resolve checkpoints
    central_ckpt = args.central_ckpt or find_ckpt(central_run, "best_model.pt")
    fedavg_ckpt = args.fedavg_ckpt or find_ckpt(fedavg_run, "best_model.pt")
    fedvla_ckpt = args.fedvla_ckpt or find_ckpt(fedvla_run, "best_model.pt")

    meta = {
        "created_at": ts,
        "out_dir": out_dir,
        "prepared_path": args.prepared_path,
        "tasks": tasks,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "camera": args.camera,
        "central_run": central_run,
        "fedavg_run": fedavg_run,
        "fedvla_run": fedvla_run,
        "central_ckpt": central_ckpt,
        "fedavg_ckpt": fedavg_ckpt,
        "fedvla_ckpt": fedvla_ckpt,
        "text_bank_source": "prepared_npz:text_inputs_json" if load_text_inputs_from_prepared_npz(args.prepared_path) else ("external_file" if args.text_inputs else "none"),
        "text_bank_tasks": sorted(list(text_bank.keys()))[:50],
    }
    with open(os.path.join(out_dir, "eval_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Load CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    all_results = {}

    # -------------------------
    # Centralized
    # -------------------------
    if central_ckpt and os.path.exists(central_ckpt):
        ck = torch.load(central_ckpt, map_location="cpu")
        model, uses_text = build_hptlike_from_ckpt(ck, vision_dim, state_dim, action_dim, device)

        text_cache = build_text_cache(tasks, text_bank, clip_model, clip_processor, device, seed=args.seed) if uses_text else {}

        all_results["central"] = run_eval_for_model(
            model_tag="central",
            model_obj=model,
            tasks=tasks,
            episodes=args.episodes,
            max_steps=args.max_steps,
            camera=args.camera,
            out_dir=out_dir,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            seed=args.seed,
            model_uses_text=uses_text,
            text_bank=text_bank,
            text_cache=text_cache,
        )
    else:
        print("central checkpoint not found, skip.")

    # -------------------------
    # FedAvg
    # -------------------------
    if fedavg_ckpt and os.path.exists(fedavg_ckpt):
        ck = torch.load(fedavg_ckpt, map_location="cpu")
        model, uses_text = build_hptlike_from_ckpt(ck, vision_dim, state_dim, action_dim, device)

        text_cache = build_text_cache(tasks, text_bank, clip_model, clip_processor, device, seed=args.seed) if uses_text else {}

        all_results["fedavg"] = run_eval_for_model(
            model_tag="fedavg",
            model_obj=model,
            tasks=tasks,
            episodes=args.episodes,
            max_steps=args.max_steps,
            camera=args.camera,
            out_dir=out_dir,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            seed=args.seed,
            model_uses_text=uses_text,
            text_bank=text_bank,
            text_cache=text_cache,
        )
    else:
        print("fedavg checkpoint not found, skip.")

    # -------------------------
    # FedVLA
    # -------------------------
    if fedvla_ckpt and os.path.exists(fedvla_ckpt):
        ck = torch.load(fedvla_ckpt, map_location="cpu")
        model, uses_text = build_fedvla_from_ckpt(ck, vision_dim, state_dim, action_dim, device)

        personal_by_taskid = ck["client_personal_state"]
        fixed_personal = {int(k): v for k, v in personal_by_taskid.items()}

        if not task_name_to_id:
            raise RuntimeError("FedVLA evaluation needs task_map_json in prepared_dataset.npz to map task_name->task_id.")

        text_cache = build_text_cache(tasks, text_bank, clip_model, clip_processor, device, seed=args.seed) if uses_text else {}

        all_results["fedvla"] = run_eval_for_model(
            model_tag="fedvla",
            model_obj=model,
            tasks=tasks,
            episodes=args.episodes,
            max_steps=args.max_steps,
            camera=args.camera,
            out_dir=out_dir,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            seed=args.seed,
            model_uses_text=uses_text,
            text_bank=text_bank,
            text_cache=text_cache,
            fedvla_personal_by_taskid=fixed_personal,
            task_name_to_id=task_name_to_id,
        )
    else:
        print("fedvla checkpoint not found, skip.")

    # Save summary
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Success Rate Summary ===")
    for mtag, rr in all_results.items():
        overall = rr.get("overall", {})
        print(f"{mtag}: avg_success_rate={overall.get('avg_success_rate'):.2f}% over {overall.get('tasks')} tasks")
        for tname, td in rr.get("tasks", {}).items():
            print(f"  {tname}: {td.get('success_rate'):.1f}% ({td.get('successes')}/{td.get('episodes')})")

    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()