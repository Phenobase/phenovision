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

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# benchmark CSV dir; override with TN_BENCH_DIR to analyze an isolated run (e.g. the convergence run)
BENCH = Path(os.environ.get("TN_BENCH_DIR", str(ROOT / "runs" / "benchmarks")))
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

        def _last(col):
            if col not in df.columns:
                return float("nan")
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(s.iloc[-1]) if len(s) else float("nan")

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
            # final realized-exponent distribution (stable_evo; NaN for the others)
            exp_mean=_last("mean_exponent"), exp_std=_last("exp_std"), exp_max=_last("exp_max"),
            exp_frac_high=_last("exp_frac_high"), exp_frac_floor=_last("exp_frac_floor"),
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
                                       "mean_step_time_ms", "peak_mem_mb", "wallclock_s", "steps",
                                       "exp_mean", "exp_std", "exp_max",
                                       "exp_frac_high", "exp_frac_floor")})
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
    """stable_evo realized-exponent DISTRIBUTION over training: mean±std + max (left), and the
    fraction strongly leaning Newton (right) — the mean alone hides the spread. Color = batch
    (so the batch-dependence is visible); base solid, +demo dashed."""
    se = [c for c in configs if str(c["label"]).startswith("stable_evo")
          and "mean_exponent" in c["frame"].columns]
    if not se:
        return
    batches = sorted({c["eff_batch"] for c in se})
    cmap = {b: plt.cm.viridis(i / max(len(batches) - 1, 1)) for i, b in enumerate(batches)}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    plotted = False
    for c in sorted(se, key=lambda c: (c["eff_batch"], "demo" in str(c["label"]))):
        d = c["frame"].copy()
        for col in ("step", "mean_exponent", "exp_std", "exp_max", "exp_frac_high"):
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d[d["mean_exponent"].notna()]
        if d.empty:
            continue
        color = cmap[c["eff_batch"]]
        is_demo = "demo" in str(c["label"])
        ls = "--" if is_demo else "-"
        tag = f"B{c['eff_batch']}" + ("+demo" if is_demo else "")
        st = d["step"]
        axL.plot(st, d["mean_exponent"], ls, color=color, label=tag)
        if "exp_std" in d and d["exp_std"].notna().any():
            axL.fill_between(st, d["mean_exponent"] - d["exp_std"], d["mean_exponent"] + d["exp_std"],
                             color=color, alpha=0.12)
        if "exp_max" in d and d["exp_max"].notna().any():
            axL.plot(st, d["exp_max"], ls, color=color, alpha=0.4, lw=0.8)
        if "exp_frac_high" in d and d["exp_frac_high"].notna().any():
            axR.plot(st, d["exp_frac_high"], ls, color=color, label=tag)
        plotted = True
    if not plotted:
        plt.close(fig); return
    axL.axhline(0.5, color="grey", ls=":", lw=1)
    axL.set_xlabel("step"); axL.set_ylabel("realized α  (mean ± std; faint line = max)")
    axL.set_title("StableEvolutionSOAP: exponent distribution over training")
    axL.legend(fontsize=7, ncol=2); axL.grid(alpha=.25)
    axR.set_xlabel("step"); axR.set_ylabel("fraction strongly leaning Newton (lean > 0.8)")
    axR.set_title("How much is exploited vs held at whitening")
    axR.legend(fontsize=7); axR.grid(alpha=.25)
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
