# 6_eval_plot.py
#
# Plots for report-quality figures from:
# - runs_centralized/central_*/metrics_epoch.csv
# - runs_fedavg/fedavg_*/metrics_round.csv
# - runs_fedvla/fedvla_*/metrics_round.csv
#
# Saves .png into: reports/plots_<timestamp>/
#
# Usage examples:
#   python 6_eval_plot.py
#   python 6_eval_plot.py --central runs_centralized/central_20260301_120000 \
#                         --fedavg runs_fedavg/fedavg_20260301_121000 \
#                         --fedvla runs_fedvla/fedvla_20260301_122000
#   python 6_eval_plot.py --out reports/my_plots
#
# Notes:
# - This script uses matplotlib defaults (no manual color settings).
# - It is robust to missing optional columns (e.g., val_sat, avg_density).

import os
import glob
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def latest_run_dir(root: str, prefix: str):
    """
    Find latest run directory under root matching prefix_* by modification time.
    """
    if not os.path.isdir(root):
        return None
    candidates = []
    for d in os.listdir(root):
        if d.startswith(prefix + "_"):
            full = os.path.join(root, d)
            if os.path.isdir(full):
                candidates.append(full)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_json_if_exists(path: str):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_fig(path: str, dpi: int = 250):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def safe_read_csv(path: str):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return None


def find_file_in_dir(run_dir: str, filename: str):
    p = os.path.join(run_dir, filename)
    return p if os.path.exists(p) else None


def plot_line(
    x, ys, labels, title, xlabel, ylabel, out_path,
    ylog=False
):
    plt.figure()
    for y, lab in zip(ys, labels):
        if y is None:
            continue
        plt.plot(x, y, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylog:
        plt.yscale("log")
    if any(lab for lab in labels):
        plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig(out_path)


def plot_hist(values, title, xlabel, ylabel, out_path, bins=50):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    save_fig(out_path)


def add_text_box(text: str):
    plt.gca().text(
        0.01, 0.99, text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )


def rolling_mean(arr, win: int):
    if arr is None:
        return None
    s = pd.Series(arr)
    return s.rolling(win, min_periods=1).mean().values


# -------------------------
# Main plotting routines
# -------------------------
def plot_centralized(run_dir: str, out_dir: str):
    if not run_dir:
        return

    metrics_path = find_file_in_dir(run_dir, "metrics_epoch.csv")
    stats_path = find_file_in_dir(run_dir, "stats.json")
    cfg_path = find_file_in_dir(run_dir, "config.json")
    summary_path = find_file_in_dir(run_dir, "final_summary.json")

    df = safe_read_csv(metrics_path)
    stats = load_json_if_exists(stats_path)
    cfg = load_json_if_exists(cfg_path)
    summary = load_json_if_exists(summary_path)

    if df is None or df.empty:
        return

    # Detect columns (support both earlier and newer centralized versions)
    epoch_col = "epoch" if "epoch" in df.columns else None
    if epoch_col is None:
        return
    x = df[epoch_col].values

    # candidates
    train_mse = df["train_mse"].values if "train_mse" in df.columns else None
    val_mse = df["val_mse"].values if "val_mse" in df.columns else None
    train_loss = df["train_loss"].values if "train_loss" in df.columns else None
    val_loss = df["val_loss"].values if "val_loss" in df.columns else None
    train_mae = df["train_mae"].values if "train_mae" in df.columns else None
    val_mae = df["val_mae"].values if "val_mae" in df.columns else None

    # Prefer mse columns if present
    if train_mse is not None and val_mse is not None:
        plot_line(
            x=x,
            ys=[train_mse, val_mse],
            labels=["train_mse", "val_mse"],
            title="Centralized: MSE vs Epoch",
            xlabel="Epoch",
            ylabel="MSE",
            out_path=os.path.join(out_dir, "central_mse_vs_epoch.png"),
        )
        plot_line(
            x=x,
            ys=[rolling_mean(train_mse, 5), rolling_mean(val_mse, 5)],
            labels=["train_mse (roll5)", "val_mse (roll5)"],
            title="Centralized: MSE vs Epoch (Rolling Mean)",
            xlabel="Epoch",
            ylabel="MSE",
            out_path=os.path.join(out_dir, "central_mse_vs_epoch_roll5.png"),
        )
    elif train_loss is not None and val_loss is not None:
        plot_line(
            x=x,
            ys=[train_loss, val_loss],
            labels=["train_loss", "val_loss"],
            title="Centralized: Loss vs Epoch",
            xlabel="Epoch",
            ylabel="Loss",
            out_path=os.path.join(out_dir, "central_loss_vs_epoch.png"),
        )

    if train_mae is not None and val_mae is not None:
        plot_line(
            x=x,
            ys=[train_mae, val_mae],
            labels=["train_mae", "val_mae"],
            title="Centralized: MAE vs Epoch",
            xlabel="Epoch",
            ylabel="MAE",
            out_path=os.path.join(out_dir, "central_mae_vs_epoch.png"),
        )

    # Optional: grad_norm, sat
    if "grad_norm" in df.columns:
        plot_line(
            x=x,
            ys=[df["grad_norm"].values],
            labels=["grad_norm"],
            title="Centralized: Gradient Norm vs Epoch",
            xlabel="Epoch",
            ylabel="Grad Norm",
            out_path=os.path.join(out_dir, "central_gradnorm_vs_epoch.png"),
        )

    if "val_sat" in df.columns:
        plot_line(
            x=x,
            ys=[df["val_sat"].values],
            labels=["val_sat"],
            title="Centralized: Saturation Rate vs Epoch",
            xlabel="Epoch",
            ylabel="Saturation Rate",
            out_path=os.path.join(out_dir, "central_val_saturation_vs_epoch.png"),
        )

    # Dataset stats figure
    if stats is not None:
        plt.figure()
        plt.axis("off")
        txt = []
        txt.append("Centralized Run Summary")
        if summary:
            txt.append(f"best_val: {summary.get('best_val_mse', summary.get('best_val_loss', ''))}")
            txt.append(f"epochs: {summary.get('epochs', '')}")
            txt.append(f"num_train: {summary.get('num_train', '')}  num_val: {summary.get('num_val', '')}")
        txt.append("")
        txt.append("Dataset Stats")
        for k in ["V", "S", "A"]:
            if k in stats:
                v = stats[k]
                txt.append(f"{k}: shape={v.get('shape')} mean={v.get('mean'):.6f} std={v.get('std'):.6f} "
                           f"min={v.get('min'):.6f} max={v.get('max'):.6f}")
        if cfg:
            txt.append("")
            txt.append("Config (key)")
            for kk in ["lr", "batch_size", "epochs", "d_model", "n_layers", "n_heads", "n_tokens_vision", "n_tokens_state"]:
                if kk in cfg:
                    txt.append(f"{kk}: {cfg[kk]}")
        plt.text(0.01, 0.99, "\n".join(txt), va="top")
        save_fig(os.path.join(out_dir, "central_run_summary.png"))


def plot_fedavg_or_fedvla(run_dir: str, out_dir: str, tag: str):
    """
    tag: 'fedavg' or 'fedvla'
    """
    if not run_dir:
        return

    metrics_path = find_file_in_dir(run_dir, "metrics_round.csv")
    stats_path = find_file_in_dir(run_dir, "stats.json")
    cfg_path = find_file_in_dir(run_dir, "config.json")
    summary_path = find_file_in_dir(run_dir, "final_summary.json")

    df = safe_read_csv(metrics_path)
    stats = load_json_if_exists(stats_path)
    cfg = load_json_if_exists(cfg_path)
    summary = load_json_if_exists(summary_path)

    if df is None or df.empty:
        return

    # round col may be 'round'
    if "round" not in df.columns:
        return
    x = df["round"].values

    # Standard columns for fedavg
    val_mse = df["val_mse"].values if "val_mse" in df.columns else None
    val_mae = df["val_mae"].values if "val_mae" in df.columns else None
    val_sat = df["val_sat"].values if "val_sat" in df.columns else None

    # Standard columns for fedvla (avg across clients)
    val_mse_avg = df["val_mse_avg"].values if "val_mse_avg" in df.columns else None
    val_huber_avg = df["val_huber_avg"].values if "val_huber_avg" in df.columns else None
    val_sat_avg = df["val_sat_avg"].values if "val_sat_avg" in df.columns else None
    avg_density = df["avg_client_density"].values if "avg_client_density" in df.columns else None

    # Plot val curves
    if val_mse is not None:
        plot_line(
            x=x, ys=[val_mse],
            labels=["val_mse"],
            title=f"{tag.upper()}: Validation MSE vs Round",
            xlabel="Round", ylabel="Val MSE",
            out_path=os.path.join(out_dir, f"{tag}_val_mse_vs_round.png"),
        )
        plot_line(
            x=x, ys=[rolling_mean(val_mse, 5)],
            labels=["val_mse (roll5)"],
            title=f"{tag.upper()}: Validation MSE vs Round (Rolling Mean)",
            xlabel="Round", ylabel="Val MSE",
            out_path=os.path.join(out_dir, f"{tag}_val_mse_vs_round_roll5.png"),
        )
    if val_mse_avg is not None:
        plot_line(
            x=x, ys=[val_mse_avg],
            labels=["val_mse_avg"],
            title=f"{tag.upper()}: Validation MSE vs Round",
            xlabel="Round", ylabel="Val MSE (avg)",
            out_path=os.path.join(out_dir, f"{tag}_val_mseavg_vs_round.png"),
        )
        plot_line(
            x=x, ys=[rolling_mean(val_mse_avg, 5)],
            labels=["val_mse_avg (roll5)"],
            title=f"{tag.upper()}: Validation MSE vs Round (Rolling Mean)",
            xlabel="Round", ylabel="Val MSE (avg)",
            out_path=os.path.join(out_dir, f"{tag}_val_mseavg_vs_round_roll5.png"),
        )

    # Plot MAE/huber if present
    if val_mae is not None:
        plot_line(
            x=x, ys=[val_mae],
            labels=["val_mae"],
            title=f"{tag.upper()}: Validation MAE vs Round",
            xlabel="Round", ylabel="Val MAE",
            out_path=os.path.join(out_dir, f"{tag}_val_mae_vs_round.png"),
        )
    if val_huber_avg is not None:
        plot_line(
            x=x, ys=[val_huber_avg],
            labels=["val_huber_avg"],
            title=f"{tag.upper()}: Validation Huber vs Round",
            xlabel="Round", ylabel="Val Huber (avg)",
            out_path=os.path.join(out_dir, f"{tag}_val_huberavg_vs_round.png"),
        )

    # Saturation
    if val_sat is not None:
        plot_line(
            x=x, ys=[val_sat],
            labels=["val_sat"],
            title=f"{tag.upper()}: Saturation Rate vs Round",
            xlabel="Round", ylabel="Saturation Rate",
            out_path=os.path.join(out_dir, f"{tag}_val_sat_vs_round.png"),
        )
    if val_sat_avg is not None:
        plot_line(
            x=x, ys=[val_sat_avg],
            labels=["val_sat_avg"],
            title=f"{tag.upper()}: Saturation Rate vs Round",
            xlabel="Round", ylabel="Saturation Rate (avg)",
            out_path=os.path.join(out_dir, f"{tag}_val_satavg_vs_round.png"),
        )

    # DGMoE density (FedVLA)
    if avg_density is not None:
        plot_line(
            x=x, ys=[avg_density],
            labels=["avg_client_density"],
            title=f"{tag.upper()}: Avg Expert Density vs Round",
            xlabel="Round", ylabel="Avg Density",
            out_path=os.path.join(out_dir, f"{tag}_avg_density_vs_round.png"),
        )

    # Round timing (if present)
    if "time_sec_round" in df.columns:
        plot_line(
            x=x, ys=[df["time_sec_round"].values],
            labels=["time_sec_round"],
            title=f"{tag.upper()}: Round Time vs Round",
            xlabel="Round", ylabel="Seconds",
            out_path=os.path.join(out_dir, f"{tag}_time_vs_round.png"),
        )

    # Summary figure
    plt.figure()
    plt.axis("off")
    txt = []
    txt.append(f"{tag.upper()} Run Summary")
    if summary:
        for k, v in summary.items():
            txt.append(f"{k}: {v}")
    txt.append("")
    txt.append("Dataset/Config")
    if stats:
        txt.append(f"num_train: {stats.get('num_train','')}  num_val: {stats.get('num_val','')}")
        txt.append(f"num_clients_total: {stats.get('num_clients_total','')}")
        if "A" in stats:
            a = stats["A"]
            txt.append(f"A: mean={a.get('mean',0):.6f} std={a.get('std',0):.6f} min={a.get('min',0):.6f} max={a.get('max',0):.6f}")
    if cfg:
        for kk in ["lr", "rounds", "clients_per_round", "local_epochs", "batch_size", "n_layers", "n_experts"]:
            if kk in cfg:
                txt.append(f"{kk}: {cfg[kk]}")
    plt.text(0.01, 0.99, "\n".join(txt), va="top")
    save_fig(os.path.join(out_dir, f"{tag}_run_summary.png"))


def plot_comparison(central_dir: str, fedavg_dir: str, fedvla_dir: str, out_dir: str):
    # Load curves (best-effort)
    c_df = safe_read_csv(find_file_in_dir(central_dir, "metrics_epoch.csv")) if central_dir else None
    fa_df = safe_read_csv(find_file_in_dir(fedavg_dir, "metrics_round.csv")) if fedavg_dir else None
    fv_df = safe_read_csv(find_file_in_dir(fedvla_dir, "metrics_round.csv")) if fedvla_dir else None

    # Central val curve
    c_x, c_y = None, None
    if c_df is not None and "epoch" in c_df.columns:
        if "val_mse" in c_df.columns:
            c_x = c_df["epoch"].values
            c_y = c_df["val_mse"].values
        elif "val_loss" in c_df.columns:
            c_x = c_df["epoch"].values
            c_y = c_df["val_loss"].values

    # FedAvg val curve
    fa_x, fa_y = None, None
    if fa_df is not None and "round" in fa_df.columns and "val_mse" in fa_df.columns:
        fa_x = fa_df["round"].values
        fa_y = fa_df["val_mse"].values

    # FedVLA val curve
    fv_x, fv_y = None, None
    if fv_df is not None and "round" in fv_df.columns:
        if "val_mse_avg" in fv_df.columns:
            fv_x = fv_df["round"].values
            fv_y = fv_df["val_mse_avg"].values
        elif "val_mse" in fv_df.columns:
            fv_x = fv_df["round"].values
            fv_y = fv_df["val_mse"].values

    # Plot on same figure (normalize x-label names)
    plt.figure()
    if c_x is not None and c_y is not None:
        plt.plot(c_x, c_y, label="Centralized (val)")
    if fa_x is not None and fa_y is not None:
        plt.plot(fa_x, fa_y, label="FedAvg (val)")
    if fv_x is not None and fv_y is not None:
        plt.plot(fv_x, fv_y, label="FedVLA (val)")

    plt.title("Validation Curve Comparison")
    plt.xlabel("Epoch (Centralized) / Round (Federated)")
    plt.ylabel("Validation Metric (MSE or Loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig(os.path.join(out_dir, "compare_val_curve.png"))

    # Rolling comparison
    plt.figure()
    if c_x is not None and c_y is not None:
        plt.plot(c_x, rolling_mean(c_y, 5), label="Centralized (roll5)")
    if fa_x is not None and fa_y is not None:
        plt.plot(fa_x, rolling_mean(fa_y, 5), label="FedAvg (roll5)")
    if fv_x is not None and fv_y is not None:
        plt.plot(fv_x, rolling_mean(fv_y, 5), label="FedVLA (roll5)")

    plt.title("Validation Curve Comparison (Rolling Mean)")
    plt.xlabel("Epoch / Round")
    plt.ylabel("Validation Metric (Rolling)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig(os.path.join(out_dir, "compare_val_curve_roll5.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--central", type=str, default=None, help="Path to a centralized run dir (central_*)")
    ap.add_argument("--fedavg", type=str, default=None, help="Path to a fedavg run dir (fedavg_*)")
    ap.add_argument("--fedvla", type=str, default=None, help="Path to a fedvla run dir (fedvla_*)")
    ap.add_argument("--central_root", type=str, default="runs_centralized", help="Root folder for centralized runs")
    ap.add_argument("--fedavg_root", type=str, default="runs_fedavg", help="Root folder for fedavg runs")
    ap.add_argument("--fedvla_root", type=str, default="runs_fedvla", help="Root folder for fedvla runs")
    ap.add_argument("--out", type=str, default=None, help="Output folder for pngs")
    args = ap.parse_args()

    # Auto-pick latest runs if not specified
    central_dir = args.central or latest_run_dir(args.central_root, "central")
    fedavg_dir = args.fedavg or latest_run_dir(args.fedavg_root, "fedavg")
    fedvla_dir = args.fedvla or latest_run_dir(args.fedvla_root, "fedvla")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join("reports", f"plots_{ts}")
    ensure_dir(out_dir)

    # Save which runs used
    meta = {
        "central_dir": central_dir,
        "fedavg_dir": fedavg_dir,
        "fedvla_dir": fedvla_dir,
        "created_at": ts,
    }
    with open(os.path.join(out_dir, "plots_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Generate plots
    if central_dir:
        plot_centralized(central_dir, out_dir)
    if fedavg_dir:
        plot_fedavg_or_fedvla(fedavg_dir, out_dir, "fedavg")
    if fedvla_dir:
        plot_fedavg_or_fedvla(fedvla_dir, out_dir, "fedvla")

    # Comparisons
    if central_dir or fedavg_dir or fedvla_dir:
        plot_comparison(central_dir, fedavg_dir, fedvla_dir, out_dir)

    print("Saved plots to:", out_dir)
    print("Centralized:", central_dir)
    print("FedAvg:", fedavg_dir)
    print("FedVLA:", fedvla_dir)


if __name__ == "__main__":
    main()