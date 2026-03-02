# 2_prepare_dataset.py

import os
import glob
import json
import random
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def set_seed(seed: int = 42):
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


@dataclass
class CFG:
    dataset_root: str = "./dataset"
    pattern: str = "**/dataset.npz"

    prepared_root: str = "./prepared"
    prepared_name: str = "prepared_dataset.npz"
    meta_name: str = "prepared_meta.json"

    text_inputs_json: str = "./text_inputs.json"
    seed: int = 42
    val_ratio: float = 0.1

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_batch_size_img: int = 64
    clip_batch_size_txt: int = 256

    image_limit_per_file: int = 0  # 0 = no limit, otherwise subsample per dataset file


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def split_indices(n: int, val_ratio: float, seed: int):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_val = int(n * val_ratio)
    return idx[n_val:], idx[:n_val]


def load_text_inputs(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"text_inputs.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("text_inputs.json must be a dict: {task_name: [list of strings]}")
    for k, v in obj.items():
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"text_inputs.json entry must be non-empty list for task: {k}")
    return obj


@torch.no_grad()
def encode_clip_images(images_uint8: np.ndarray, model, processor, device, batch_size: int):
    n = int(images_uint8.shape[0])
    out = []
    model.eval()

    for i in range(0, n, batch_size):
        batch = [Image.fromarray(im) for im in images_uint8[i:i + batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        vision_out = model.vision_model(**inputs).pooler_output
        v = model.visual_projection(vision_out)
        out.append(v.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(out, axis=0).astype(np.float32)


@torch.no_grad()
def encode_clip_texts(texts: list, model, processor, device, batch_size: int):
    n = len(texts)
    out = []
    model.eval()

    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Preferred path: tensor (B, 512)
        t = None
        if hasattr(model, "get_text_features"):
            t = model.get_text_features(**inputs)

        # Robust fallback: some versions may return output object
        if not torch.is_tensor(t):
            text_out = model.text_model(**inputs)  # BaseModelOutputWithPooling
            pooled = text_out.pooler_output        # (B, hidden)
            # CLIPModel has text_projection -> (B, 512)
            t = model.text_projection(pooled)

        out.append(t.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(out, axis=0).astype(np.float32)


def maybe_subsample(images, states, actions, limit: int, seed: int):
    if limit is None or int(limit) <= 0:
        return images, states, actions
    n = images.shape[0]
    if n <= limit:
        return images, states, actions
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=int(limit), replace=False)
    idx.sort()
    return images[idx], states[idx], actions[idx]


def main():
    cfg = CFG()
    set_seed(cfg.seed)

    dataset_glob = os.path.join(cfg.dataset_root, cfg.pattern)
    paths = sorted(glob.glob(dataset_glob, recursive=True))
    if len(paths) == 0:
        print("cwd:", os.getcwd())
        print("searched:", os.path.abspath(dataset_glob))
        raise FileNotFoundError("no dataset found; run 1_collect_dataset.py first")

    text_inputs = load_text_inputs(cfg.text_inputs_json)

    ensure_dir(cfg.prepared_root)
    device = get_device()

    clip_model = CLIPModel.from_pretrained(cfg.clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(cfg.clip_model_name)

    task_map = {}
    next_id = 0

    all_v, all_t, all_s, all_a, all_tid = [], [], [], [], []
    files_meta = []
    total_frames = 0

    for p in paths:
        d = np.load(p, allow_pickle=True)

        if "images" not in d.files or "states" not in d.files or "actions" not in d.files:
            raise KeyError(f"{p} missing keys. Found: {d.files}")

        task_name = str(d["task_name"][0]) if "task_name" in d.files else "unknown"
        if task_name not in task_map:
            task_map[task_name] = next_id
            next_id += 1
        tid = int(task_map[task_name])

        if task_name not in text_inputs:
            raise KeyError(
                f"text_inputs.json has no entry for task '{task_name}'. "
                f"Add it with 20 variants."
            )

        images = d["images"].astype(np.uint8)
        states = d["states"].astype(np.float32)
        actions = d["actions"].astype(np.float32)
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)

        images, states, actions = maybe_subsample(
            images, states, actions, limit=cfg.image_limit_per_file, seed=cfg.seed + tid
        )

        n = int(images.shape[0])
        total_frames += n

        v_feats = encode_clip_images(
            images_uint8=images,
            model=clip_model,
            processor=clip_processor,
            device=device,
            batch_size=cfg.clip_batch_size_img,
        )

        # sample one instruction per frame from the 20 variants
        instr_list = text_inputs[task_name]
        instr_per_sample = [random.choice(instr_list) for _ in range(n)]

        t_feats = encode_clip_texts(
            texts=instr_per_sample,
            model=clip_model,
            processor=clip_processor,
            device=device,
            batch_size=cfg.clip_batch_size_txt,
        )

        all_v.append(v_feats)
        all_t.append(t_feats)
        all_s.append(states)
        all_a.append(actions)
        all_tid.append(np.full((n,), tid, dtype=np.int64))

        files_meta.append(
            {
                "path": p,
                "task_name": task_name,
                "task_id": tid,
                "n": n,
            }
        )

    V = np.concatenate(all_v, axis=0).astype(np.float32)
    TT = np.concatenate(all_t, axis=0).astype(np.float32)
    S = np.concatenate(all_s, axis=0).astype(np.float32)
    A = np.concatenate(all_a, axis=0).astype(np.float32)
    TID = np.concatenate(all_tid, axis=0).astype(np.int64)

    tr_idx, va_idx = split_indices(V.shape[0], cfg.val_ratio, cfg.seed)

    out_npz = os.path.join(cfg.prepared_root, cfg.prepared_name)
    np.savez_compressed(
        out_npz,
        v_feats=V,
        t_feats=TT,
        states=S,
        actions=A,
        task_ids=TID,
        train_idx=tr_idx.astype(np.int64),
        val_idx=va_idx.astype(np.int64),
        task_map_json=np.array([json.dumps(task_map)], dtype=object),
        text_inputs_json=np.array([json.dumps(text_inputs)], dtype=object),
    )

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "dataset_glob": os.path.abspath(dataset_glob),
        "device": str(device),
        "clip_model": cfg.clip_model_name,
        "seed": cfg.seed,
        "val_ratio": cfg.val_ratio,
        "task_map": task_map,
        "num_total": int(V.shape[0]),
        "num_train": int(tr_idx.shape[0]),
        "num_val": int(va_idx.shape[0]),
        "files": files_meta,
        "shapes": {
            "v_feats": list(V.shape),
            "t_feats": list(TT.shape),
            "states": list(S.shape),
            "actions": list(A.shape),
            "task_ids": list(TID.shape),
        },
        "notes": {
            "instruction_sampling": "one random instruction per sample from text_inputs.json[task_name]",
            "actions_clipped": "np.clip(actions, -1, 1)",
            "image_limit_per_file": int(cfg.image_limit_per_file),
        },
    }

    meta_path = os.path.join(cfg.prepared_root, cfg.meta_name)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("prepared:", out_npz)
    print("meta:", meta_path)


if __name__ == "__main__":
    main()