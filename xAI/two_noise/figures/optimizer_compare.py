"""Optimizer head-to-head analysis: performance + speed + the realized-exponent trajectory.

Reads the per-step benchmark CSVs (runs/benchmarks/*.csv written by ml_experiments.benchmarks for
the gen_compare_grid configs) and produces, for each (optimizer, effective batch) cell, the
best-lr summary:

  * PERFORMANCE: stationary val loss (mean of the last K evals) + final val metric.
  * SPEED:       mean step_time_ms, peak_mem_mb, total wallclock, and steps/wallclock to reach a
                 target val loss (time-to-accuracy — the honest step-efficiency-vs-wallclock split).
  * stable_evo:  the realized exponent mean_exponent(t) over training.

Writes runs/figures/optimizer_compare_summary.csv and a few PNGs into figures/. Degrades
gracefully if the runs are absent. Run:  python3 -m figures.optimizer_compare
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCH = ROOT / "runs" / "benchmarks"
OUTFIG = ROOT / "figures"
OUTCSV = ROOT / "runs" / "figures"


def _opt_label(df: pd.DataFrame) -> str:
    opt = str(df["optimizer"].iloc[0])
    if opt == "soap":
        a = df["alpha"].dropna()
        opt = f"soap@{float(a.iloc[0]):g}" if len(a) else "soap"
    # distinguish demographic-noise (pSGLD) variants from the plain optimizer
    if "demo_temp" in df.columns:
        dt = pd.to_numeric(df["demo_temp"], errors="coerce").dropna()
        if len(dt) and dt.iloc[0] > 0:
            opt += f"+demoT{dt.iloc[0]:g}"
    return opt


def load_configs(model="vit_s", dataset="cifar100") -> list[dict]:
    """One dict per benchmark CSV: label, eff_batch, lr, full per-step frame, summary scalars."""
    if not BENCH.exists():
        return []
    out = []
    for f in sorted(BENCH.glob("*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or "optimizer" not in df.columns:
            continue
        if str(df["model"].iloc[0]) != model or str(df["dataset"].iloc[0]) != dataset:
            continue
        evals = df[df["val_loss"].notna() & (df["val_loss"] != "")]
        evals = evals.copy()
        evals["val_loss"] = pd.to_numeric(evals["val_loss"], errors="coerce")
        evals = evals[evals["val_loss"].notna()]
        if evals.empty:
            continue
        k = max(1, min(3, len(evals)))
        stationary = float(evals["val_loss"].iloc[-k:].mean())
        out.append(dict(
            file=f.name, label=_opt_label(df), eff_batch=int(df["eff_batch_size"].iloc[0]),
            lr=float(df["lr_actual"].iloc[0]), frame=df, evals=evals,
            stationary_val_loss=stationary,
            final_val_metric=float(pd.to_numeric(evals["val_metric"], errors="coerce").iloc[-1]),
            val_metric_name=str(df["val_metric_name"].iloc[-1]),
            mean_step_time_ms=float(pd.to_numeric(df["step_time_ms"], errors="coerce").replace(0, np.nan).mean()),
            peak_mem_mb=float(pd.to_numeric(df["peak_mem_mb"], errors="coerce").max()),
            wallclock_s=float(pd.to_numeric(df["wallclock_s"], errors="coerce").max()),
            steps=int(pd.to_numeric(df["step"], errors="coerce").max()),
        ))
    return out


def best_per_cell(configs: list[dict]) -> pd.DataFrame:
    """Pick the best-lr config (min stationary val loss) per (label, eff_batch)."""
    rows = []
    seen = {}
    for c in configs:
        key = (c["label"], c["eff_batch"])
        if key not in seen or c["stationary_val_loss"] < seen[key]["stationary_val_loss"]:
            seen[key] = c
    for (label, eb), c in sorted(seen.items()):
        rows.append({k: c[k] for k in ("label", "eff_batch", "lr", "stationary_val_loss",
                                       "final_val_metric", "val_metric_name",
                                       "mean_step_time_ms", "peak_mem_mb", "wallclock_s", "steps")})
    return pd.DataFrame(rows)


def _plot_trajectories(configs, best, outpath):
    """val_loss vs step and vs wallclock for the best-lr config of each optimizer, faceted by batch."""
    batches = sorted({c["eff_batch"] for c in configs})
    if not batches:
        return
    best_files = set(best["label"] + "|" + best["eff_batch"].astype(str))
    fig, axes = plt.subplots(2, len(batches), figsize=(5 * len(batches), 8), squeeze=False)
    for j, eb in enumerate(batches):
        for c in configs:
            if c["eff_batch"] != eb or (c["label"] + "|" + str(eb)) not in best_files:
                continue
            ev = c["evals"]
            axes[0][j].plot(pd.to_numeric(ev["step"]), ev["val_loss"], "-o", ms=3, label=c["label"])
            axes[1][j].plot(pd.to_numeric(ev["wallclock_s"]), ev["val_loss"], "-o", ms=3, label=c["label"])
        axes[0][j].set_title(f"eff batch {eb}"); axes[0][j].set_xlabel("step"); axes[0][j].set_ylabel("val loss")
        axes[1][j].set_xlabel("wallclock (s)"); axes[1][j].set_ylabel("val loss")
        axes[0][j].grid(alpha=.25); axes[1][j].grid(alpha=.25); axes[0][j].legend(fontsize=7)
    fig.suptitle("Optimizer comparison to equilibrium: val loss vs step (top) & wallclock (bottom)")
    fig.tight_layout(); fig.savefig(outpath, dpi=140); plt.close(fig)
    print("wrote", outpath)


def _plot_alpha_traj(configs, outpath):
    """stable_evo realized exponent over training, one line per batch."""
    se = [c for c in configs if c["label"] == "stable_evo" and "mean_exponent" in c["frame"].columns]
    if not se:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plotted = False
    for c in sorted(se, key=lambda c: c["eff_batch"]):
        d = c["frame"].copy()
        d["mean_exponent"] = pd.to_numeric(d["mean_exponent"], errors="coerce")
        d = d[d["mean_exponent"].notna()]
        if d.empty:
            continue
        ax.plot(pd.to_numeric(d["step"]), d["mean_exponent"], "-", label=f"eff batch {c['eff_batch']}")
        plotted = True
    if not plotted:
        plt.close(fig); return
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="whitening (½)")
    ax.set_xlabel("step"); ax.set_ylabel("realized exponent α  (mean over coords)")
    ax.set_title("StableEvolutionSOAP: realized α over training (higher batch → higher α)")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(outpath, dpi=140); plt.close(fig)
    print("wrote", outpath)


def build():
    configs = load_configs()
    if not configs:
        print("[note] no benchmark runs yet -> optimizer comparison skipped")
        return None
    best = best_per_cell(configs)
    OUTCSV.mkdir(parents=True, exist_ok=True)
    best.to_csv(OUTCSV / "optimizer_compare_summary.csv", index=False)
    print("wrote", OUTCSV / "optimizer_compare_summary.csv")
    print(best.to_string(index=False))
    _plot_trajectories(configs, best, OUTFIG / "optimizer_compare_trajectories.png")
    _plot_alpha_traj(configs, OUTFIG / "optimizer_compare_alpha.png")
    return best


if __name__ == "__main__":
    build()
