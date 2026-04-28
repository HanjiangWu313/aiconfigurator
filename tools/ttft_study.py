#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TTFT Fairness Study — Two analyses for 6 deployment modes on 8 GPUs.

Study 1: Fixed-load TTFT comparison
    At the same concurrency level, which mode has the lowest TTFT?
    For each target concurrency, find the best (lowest-TTFT) config per mode
    and compare them side-by-side.

Study 2: TTFT vs Throughput Pareto frontier
    Scatter the full dataset of (TTFT, seq/s) for each mode.
    Overlay the Pareto frontier per mode to show the tradeoff envelope.
    This reveals whether one mode dominates another across the full range.

Reads from the CSVs produced by plot_8gpu_comparison.py.

Usage:
    cd /scratch1/hanjiang/aiconfigurator
    source aiconfigvenv/bin/activate
    python tools/ttft_study.py                       # default study
    python tools/ttft_study.py --concurrency 16,32,64,128,256   # custom fixed-load levels
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================================
# Configuration — must match plot_8gpu_comparison.py
# =========================================================================
MODEL_PATH = "Qwen/Qwen3-30B-A3B"
SYSTEM = "h100_sxm"
TOTAL_GPUS = 8
CSV_PREFIX = "plot_8gpu"

# Concurrency levels for the fixed-load study (can be overridden via CLI)
DEFAULT_CONCURRENCY_TARGETS = [8, 16, 32, 64, 128, 256, 512]

# Tolerance: when matching a target concurrency, accept rows whose concurrency
# falls within [target * (1 - TOL), target * (1 + TOL)].
CONCURRENCY_TOLERANCE = 0.25

# =========================================================================
# Mode style definitions (reuse palette from plot_8gpu_comparison.py)
# =========================================================================
MODE_STYLES = {
    "Agg Standard":    {"color": "#2196F3", "marker": "o",  "dash": "solid",   "label": "Agg (Standard)"},
    "Agg AFD":         {"color": "#FF9800", "marker": "s",  "dash": "dash",    "label": "Agg + AFD (M=1)"},
    "Agg AFD M=4":     {"color": "#FF5722", "marker": "P",  "dash": "dashdot", "label": "Agg + AFD M=4"},
    "Disagg Standard": {"color": "#4CAF50", "marker": "^",  "dash": "dot",     "label": "Disagg (Standard)"},
    "Disagg AFD":      {"color": "#E91E63", "marker": "D",  "dash": "longdash","label": "Disagg + AFD (M=1)"},
    "Disagg AFD M=4":  {"color": "#9C27B0", "marker": "*",  "dash": "longdashdot", "label": "Disagg + AFD M=4"},
}

PLOTLY_MARKERS = {
    "o": "circle", "s": "square", "P": "cross",
    "^": "triangle-up", "D": "diamond", "*": "star",
}


# =========================================================================
# Data loading
# =========================================================================
def load_csvs() -> dict[str, pd.DataFrame]:
    """Load the 6-mode CSVs produced by plot_8gpu_comparison.py."""
    mode_dfs = {}
    for mode_key in MODE_STYLES:
        safe_name = mode_key.replace(" ", "_").replace("=", "").lower()
        fname = f"{CSV_PREFIX}_{safe_name}_{SYSTEM}.csv"
        if os.path.exists(fname):
            mode_dfs[mode_key] = pd.read_csv(fname)
            print(f"  Loaded: {fname} ({len(mode_dfs[mode_key])} rows)")
        else:
            print(f"  Not found: {fname} — skipping {mode_key}")
            mode_dfs[mode_key] = pd.DataFrame()
    return mode_dfs


def _get_concurrency_col(df: pd.DataFrame) -> str:
    """Return the column name that represents effective concurrency."""
    return "concurrency"


# =========================================================================
# Study 1 — Fixed-load TTFT comparison
# =========================================================================
def _find_best_at_concurrency(
    df: pd.DataFrame,
    target_cc: float,
    tol: float = CONCURRENCY_TOLERANCE,
) -> Optional[pd.Series]:
    """Find the row with lowest TTFT near the target concurrency.

    Returns None if no row is within tolerance.
    """
    if df.empty:
        return None
    cc_col = _get_concurrency_col(df)
    lo, hi = target_cc * (1 - tol), target_cc * (1 + tol)
    mask = (df[cc_col] >= lo) & (df[cc_col] <= hi)
    subset = df.loc[mask]
    if subset.empty:
        return None
    best_idx = subset["ttft"].idxmin()
    return subset.loc[best_idx]


def study_fixed_load_ttft(
    mode_dfs: dict[str, pd.DataFrame],
    concurrency_targets: list[int],
):
    """Study 1: At the same concurrency, which mode has the lowest TTFT?"""
    print("\n" + "=" * 100)
    print("  STUDY 1 — Fixed-Load TTFT Comparison")
    print("  At the same concurrency, which mode has the lowest TTFT?")
    print("=" * 100)

    # Build a summary table: rows = concurrency targets, columns = modes
    records = []
    for target_cc in concurrency_targets:
        row = {"concurrency": target_cc}
        for mode_key, df in mode_dfs.items():
            best = _find_best_at_concurrency(df, target_cc)
            if best is not None:
                row[f"{mode_key} TTFT"] = round(best["ttft"], 1)
                row[f"{mode_key} seq/s"] = round(best["seq/s"], 3)
                row[f"{mode_key} tpot"] = round(best["tpot"], 1)
                row[f"{mode_key} actual_cc"] = int(best["concurrency"])
            else:
                row[f"{mode_key} TTFT"] = None
                row[f"{mode_key} seq/s"] = None
                row[f"{mode_key} tpot"] = None
                row[f"{mode_key} actual_cc"] = None
        records.append(row)

    summary = pd.DataFrame(records)

    # Print TTFT sub-table
    ttft_cols = ["concurrency"] + [f"{m} TTFT" for m in MODE_STYLES]
    print("\n  TTFT (ms) at each concurrency level (lower is better):")
    avail = [c for c in ttft_cols if c in summary.columns]
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.float_format", "{:.1f}".format):
        print(summary[avail].to_string(index=False))

    # Print seq/s sub-table
    seqs_cols = ["concurrency"] + [f"{m} seq/s" for m in MODE_STYLES]
    print("\n  seq/s at each concurrency level (higher is better):")
    avail = [c for c in seqs_cols if c in summary.columns]
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.float_format", "{:.3f}".format):
        print(summary[avail].to_string(index=False))

    # Identify winner per row
    print("\n  Winner (lowest TTFT) at each concurrency:")
    for _, r in summary.iterrows():
        cc = int(r["concurrency"])
        best_mode, best_ttft = None, float("inf")
        for mode_key in MODE_STYLES:
            v = r.get(f"{mode_key} TTFT")
            if v is not None and not pd.isna(v) and v < best_ttft:
                best_ttft = v
                best_mode = mode_key
        if best_mode:
            print(f"    cc={cc:>4d}:  {MODE_STYLES[best_mode]['label']}  (TTFT={best_ttft:.1f} ms)")
        else:
            print(f"    cc={cc:>4d}:  no data")

    # --- Save CSV ---
    csv_name = f"ttft_study1_fixed_load_{SYSTEM}.csv"
    summary.to_csv(csv_name, index=False)
    print(f"\n  Saved: {csv_name}")

    # --- Static plot ---
    _plot_fixed_load_static(mode_dfs, concurrency_targets)
    # --- Interactive plot ---
    _plot_fixed_load_interactive(mode_dfs, concurrency_targets)


def _plot_fixed_load_static(
    mode_dfs: dict[str, pd.DataFrame],
    concurrency_targets: list[int],
):
    """Bar chart: TTFT at each fixed concurrency, grouped by mode."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot A: TTFT bars
    ax = axes[0]
    n_modes = len(MODE_STYLES)
    n_targets = len(concurrency_targets)
    bar_width = 0.8 / n_modes
    x = np.arange(n_targets)

    for i, (mode_key, df) in enumerate(mode_dfs.items()):
        style = MODE_STYLES[mode_key]
        ttft_vals = []
        for cc in concurrency_targets:
            best = _find_best_at_concurrency(df, cc)
            ttft_vals.append(best["ttft"] if best is not None else 0)
        bars = ax.bar(
            x + i * bar_width, ttft_vals, bar_width,
            label=style["label"], color=style["color"],
            edgecolor="black", linewidth=0.5,
        )
        # annotate non-zero bars
        for bar, v in zip(bars, ttft_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + bar_width * (n_modes - 1) / 2)
    ax.set_xticklabels([str(c) for c in concurrency_targets])
    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel("TTFT (ms)", fontsize=12)
    ax.set_title(f"TTFT at Fixed Concurrency — {TOTAL_GPUS} GPUs ({MODEL_PATH})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Plot B: seq/s bars (throughput at the same concurrency for context)
    ax = axes[1]
    for i, (mode_key, df) in enumerate(mode_dfs.items()):
        style = MODE_STYLES[mode_key]
        seqs_vals = []
        for cc in concurrency_targets:
            best = _find_best_at_concurrency(df, cc)
            seqs_vals.append(best["seq/s"] if best is not None else 0)
        bars = ax.bar(
            x + i * bar_width, seqs_vals, bar_width,
            label=style["label"], color=style["color"],
            edgecolor="black", linewidth=0.5,
        )
        for bar, v in zip(bars, seqs_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + bar_width * (n_modes - 1) / 2)
    ax.set_xticklabels([str(c) for c in concurrency_targets])
    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel("seq/s", fontsize=12)
    ax.set_title(f"Throughput at Fixed Concurrency — {TOTAL_GPUS} GPUs ({MODEL_PATH})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fname = f"ttft_study1_fixed_load_{SYSTEM}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.close(fig)


def _plot_fixed_load_interactive(
    mode_dfs: dict[str, pd.DataFrame],
    concurrency_targets: list[int],
):
    """Interactive grouped bar chart with Plotly."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"TTFT at Fixed Concurrency — {TOTAL_GPUS} GPUs",
            f"Throughput at Fixed Concurrency — {TOTAL_GPUS} GPUs",
        ],
    )

    cc_labels = [str(c) for c in concurrency_targets]

    for mode_key, df in mode_dfs.items():
        style = MODE_STYLES[mode_key]
        ttft_vals, seqs_vals, hover_texts = [], [], []
        for cc in concurrency_targets:
            best = _find_best_at_concurrency(df, cc)
            if best is not None:
                ttft_vals.append(best["ttft"])
                seqs_vals.append(best["seq/s"])
                # Build hover with config details
                lines = [
                    f"<b>{style['label']}</b>",
                    f"target_cc={cc}, actual_cc={int(best['concurrency'])}",
                    f"TTFT={best['ttft']:.1f} ms",
                    f"TPOT={best['tpot']:.1f} ms",
                    f"seq/s={best['seq/s']:.3f}",
                    f"tokens/s={best['tokens/s']:.1f}",
                ]
                # Config details (agg vs disagg)
                if "(p)tp" in best.index:
                    lines.append(f"prefill: tp={_iv(best.get('(p)tp'))},dp={_iv(best.get('(p)dp'))},moe_ep={_iv(best.get('(p)moe_ep'))},workers={_iv(best.get('(p)workers'))}")
                    lines.append(f"decode:  tp={_iv(best.get('(d)tp'))},dp={_iv(best.get('(d)dp'))},moe_ep={_iv(best.get('(d)moe_ep'))},workers={_iv(best.get('(d)workers'))}")
                else:
                    lines.append(f"parallel: tp={_iv(best.get('tp'))},dp={_iv(best.get('dp'))},moe_tp={_iv(best.get('moe_tp'))},moe_ep={_iv(best.get('moe_ep'))}")
                    lines.append(f"bs={_iv(best.get('bs'))}, global_bs={_iv(best.get('global_bs'))}")
                    if not pd.isna(best.get("num_attn_gpus", float("nan"))):
                        lines.append(f"attn_gpus={_iv(best.get('num_attn_gpus'))}, ffn_gpus={_iv(best.get('num_ffn_gpus'))}")
                hover_texts.append("<br>".join(lines))
            else:
                ttft_vals.append(None)
                seqs_vals.append(None)
                hover_texts.append(f"<b>{style['label']}</b><br>No data at cc≈{cc}")

        fig.add_trace(
            go.Bar(
                x=cc_labels, y=ttft_vals, name=style["label"],
                marker_color=style["color"],
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                legendgroup=mode_key,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=cc_labels, y=seqs_vals, name=style["label"],
                marker_color=style["color"],
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
                legendgroup=mode_key,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        width=1600, height=700,
        title_text=f"Study 1: Fixed-Load TTFT Comparison — {TOTAL_GPUS} GPUs ({MODEL_PATH})",
        hoverlabel={"align": "left"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "xanchor": "left", "x": 0},
    )
    fig.update_yaxes(title_text="TTFT (ms)", row=1, col=1)
    fig.update_yaxes(title_text="seq/s", row=1, col=2)
    fig.update_xaxes(title_text="Concurrency", row=1, col=1)
    fig.update_xaxes(title_text="Concurrency", row=1, col=2)

    fname = f"ttft_study1_fixed_load_{SYSTEM}.html"
    fig.write_html(fname, include_plotlyjs=True, full_html=True)
    print(f"  Saved: {fname}")


def _iv(val) -> str:
    """Format a value for hover text (int-or-NA)."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or not np.isfinite(val))):
        return "N/A"
    try:
        return str(int(round(float(val))))
    except (TypeError, ValueError):
        return str(val)


# =========================================================================
# Study 2 — TTFT vs Throughput Pareto frontier
# =========================================================================
def _pareto_front_2d(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return indices of the Pareto-optimal points (minimize x, maximize y)."""
    n = len(xs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j has <= x AND >= y (with at least one strict)
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def study_ttft_vs_throughput(mode_dfs: dict[str, pd.DataFrame]):
    """Study 2: Scatter the full (TTFT, seq/s) space with Pareto frontiers."""
    print("\n" + "=" * 100)
    print("  STUDY 2 — TTFT vs Throughput Pareto Frontier")
    print("  Does one mode dominate another across the full operating range?")
    print("=" * 100)

    _plot_ttft_pareto_static(mode_dfs)
    _plot_ttft_pareto_interactive(mode_dfs)
    _print_pareto_summary(mode_dfs)


def _plot_ttft_pareto_static(mode_dfs: dict[str, pd.DataFrame]):
    """Static Matplotlib scatter + Pareto frontier lines."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # Left panel: full scatter with Pareto frontiers
    ax = axes[0]
    for mode_key, df in mode_dfs.items():
        if df.empty or "ttft" not in df.columns or "seq/s" not in df.columns:
            continue
        style = MODE_STYLES[mode_key]
        x, y = df["ttft"].values, df["seq/s"].values

        # scatter all points (faded)
        ax.scatter(x, y, c=style["color"], marker=style["marker"],
                   s=30, alpha=0.25, edgecolors="none")

        # Pareto frontier
        pidx = _pareto_front_2d(x, y)
        if len(pidx) > 0:
            px, py = x[pidx], y[pidx]
            order = np.argsort(px)
            ax.plot(px[order], py[order], color=style["color"], linewidth=2.5,
                    linestyle="-", label=style["label"], zorder=6)
            ax.scatter(px, py, c=style["color"], marker=style["marker"],
                       s=100, edgecolors="black", linewidths=0.6, zorder=7)

    ax.set_xlabel("TTFT (ms)", fontsize=12)
    ax.set_ylabel("seq/s", fontsize=12)
    ax.set_title(f"TTFT vs seq/s Pareto — {TOTAL_GPUS} GPUs ({MODEL_PATH})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Right panel: same but tokens/s
    ax = axes[1]
    for mode_key, df in mode_dfs.items():
        if df.empty or "ttft" not in df.columns or "tokens/s" not in df.columns:
            continue
        style = MODE_STYLES[mode_key]
        x, y = df["ttft"].values, df["tokens/s"].values

        ax.scatter(x, y, c=style["color"], marker=style["marker"],
                   s=30, alpha=0.25, edgecolors="none")

        pidx = _pareto_front_2d(x, y)
        if len(pidx) > 0:
            px, py = x[pidx], y[pidx]
            order = np.argsort(px)
            ax.plot(px[order], py[order], color=style["color"], linewidth=2.5,
                    linestyle="-", label=style["label"], zorder=6)
            ax.scatter(px, py, c=style["color"], marker=style["marker"],
                       s=100, edgecolors="black", linewidths=0.6, zorder=7)

    ax.set_xlabel("TTFT (ms)", fontsize=12)
    ax.set_ylabel("tokens/s", fontsize=12)
    ax.set_title(f"TTFT vs tokens/s Pareto — {TOTAL_GPUS} GPUs ({MODEL_PATH})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f"ttft_study2_pareto_{SYSTEM}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.close(fig)


def _build_hover_for_row(mode_key: str, row: pd.Series) -> str:
    """Build a detailed hover tooltip for one data point."""
    style = MODE_STYLES[mode_key]
    lines = [
        f"<b>{style['label']}</b>",
        f"TTFT={row['ttft']:.1f} ms",
        f"TPOT={row['tpot']:.1f} ms",
        f"seq/s={row['seq/s']:.3f}",
        f"tokens/s={row['tokens/s']:.1f}",
        f"request_latency={row.get('request_latency', 0):.1f} ms",
        f"concurrency={int(row['concurrency'])}",
    ]

    if "(p)tp" in row.index:
        # disagg mode
        lines.append(f"prefill: tp={_iv(row.get('(p)tp'))},dp={_iv(row.get('(p)dp'))},moe_ep={_iv(row.get('(p)moe_ep'))}, workers={_iv(row.get('(p)workers'))}, bs={_iv(row.get('(p)bs'))}")
        lines.append(f"decode:  tp={_iv(row.get('(d)tp'))},dp={_iv(row.get('(d)dp'))},moe_ep={_iv(row.get('(d)moe_ep'))}, workers={_iv(row.get('(d)workers'))}, bs={_iv(row.get('(d)bs'))}")
        if not pd.isna(row.get("(d)num_attn_gpus", float("nan"))):
            lines.append(f"decode attn_gpus={_iv(row.get('(d)num_attn_gpus'))}, ffn_gpus={_iv(row.get('(d)num_ffn_gpus'))}")
    else:
        # agg mode
        lines.append(f"parallel: tp={_iv(row.get('tp'))},dp={_iv(row.get('dp'))},moe_tp={_iv(row.get('moe_tp'))},moe_ep={_iv(row.get('moe_ep'))}")
        lines.append(f"bs={_iv(row.get('bs'))}, global_bs={_iv(row.get('global_bs'))}")
        if not pd.isna(row.get("num_attn_gpus", float("nan"))):
            lines.append(f"attn_gpus={_iv(row.get('num_attn_gpus'))}, ffn_gpus={_iv(row.get('num_ffn_gpus'))}")
        lines.append(f"replicas={_iv(row.get('num_replicas'))}, worker_gpus={_iv(row.get('worker_gpus'))}")

    return "<br>".join(lines)


def _plot_ttft_pareto_interactive(mode_dfs: dict[str, pd.DataFrame]):
    """Interactive Plotly scatter + Pareto frontiers for TTFT vs seq/s and TTFT vs tokens/s."""
    for y_metric, y_label, suffix in [
        ("seq/s", "seq/s", "seqs"),
        ("tokens/s", "tokens/s", "tokenss"),
    ]:
        fig = go.Figure()

        for mode_key, df in mode_dfs.items():
            if df.empty or "ttft" not in df.columns or y_metric not in df.columns:
                continue
            style = MODE_STYLES[mode_key]
            x = df["ttft"].values
            y = df[y_metric].values

            # All points (faded)
            hover_all = [_build_hover_for_row(mode_key, row) for _, row in df.iterrows()]
            fig.add_trace(go.Scatter(
                x=x.tolist(), y=y.tolist(),
                mode="markers",
                name=f"{style['label']} (all)",
                text=hover_all,
                hovertemplate="%{text}<extra></extra>",
                marker={
                    "size": 6,
                    "color": style["color"],
                    "opacity": 0.25,
                    "symbol": PLOTLY_MARKERS.get(style["marker"], "circle"),
                },
                legendgroup=mode_key,
                showlegend=False,
            ))

            # Pareto frontier
            pidx = _pareto_front_2d(x, y)
            if len(pidx) > 0:
                order = np.argsort(x[pidx])
                p_indices = pidx[order]
                px = x[p_indices]
                py = y[p_indices]
                hover_pareto = [_build_hover_for_row(mode_key, df.iloc[i]) for i in p_indices]

                fig.add_trace(go.Scatter(
                    x=px.tolist(), y=py.tolist(),
                    mode="lines+markers",
                    name=style["label"],
                    text=hover_pareto,
                    hovertemplate="%{text}<extra></extra>",
                    line={"color": style["color"], "width": 3},
                    marker={
                        "size": 12,
                        "color": style["color"],
                        "symbol": PLOTLY_MARKERS.get(style["marker"], "circle"),
                        "line": {"color": "black", "width": 1},
                    },
                    legendgroup=mode_key,
                ))

        fig.update_layout(
            title=f"TTFT vs {y_label} Pareto Frontier — {TOTAL_GPUS} GPUs ({MODEL_PATH})",
            template="plotly_white",
            width=1400, height=800,
            hoverlabel={"align": "left"},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        )
        fig.update_xaxes(title_text="TTFT (ms)", showgrid=True, gridcolor="rgba(0,0,0,0.12)")
        fig.update_yaxes(title_text=y_label, showgrid=True, gridcolor="rgba(0,0,0,0.12)")

        fname = f"ttft_study2_pareto_{suffix}_{SYSTEM}.html"
        fig.write_html(fname, include_plotlyjs=True, full_html=True)
        print(f"  Saved: {fname}")


def _print_pareto_summary(mode_dfs: dict[str, pd.DataFrame]):
    """Print a summary of the Pareto frontier: best seq/s and best TTFT per mode."""
    print("\n  Pareto frontier summary (TTFT vs seq/s):")
    print(f"  {'Mode':<25s} {'# Pareto pts':>12s} {'Best seq/s':>10s} {'@ TTFT':>8s}  {'Best TTFT':>10s} {'@ seq/s':>10s}")
    print("  " + "-" * 80)

    for mode_key, df in mode_dfs.items():
        if df.empty or "ttft" not in df.columns or "seq/s" not in df.columns:
            print(f"  {MODE_STYLES[mode_key]['label']:<25s}  {'no data':>12s}")
            continue
        x, y = df["ttft"].values, df["seq/s"].values
        pidx = _pareto_front_2d(x, y)
        if len(pidx) == 0:
            print(f"  {MODE_STYLES[mode_key]['label']:<25s}  {'no pareto':>12s}")
            continue

        px, py = x[pidx], y[pidx]
        best_seqs_i = np.argmax(py)
        best_ttft_i = np.argmin(px)
        print(
            f"  {MODE_STYLES[mode_key]['label']:<25s} "
            f"{len(pidx):>12d} "
            f"{py[best_seqs_i]:>10.3f} "
            f"{px[best_seqs_i]:>7.1f}ms "
            f" {px[best_ttft_i]:>9.1f}ms "
            f"{py[best_ttft_i]:>10.3f}"
        )

    # Dominance analysis: check if any mode's frontier dominates another's
    print("\n  Dominance analysis — does one mode's Pareto front dominate another?")
    print("  (A dominates B if, at every TTFT level on B's frontier, A achieves higher seq/s)")
    frontiers = {}
    for mode_key, df in mode_dfs.items():
        if df.empty or "ttft" not in df.columns or "seq/s" not in df.columns:
            continue
        x, y = df["ttft"].values, df["seq/s"].values
        pidx = _pareto_front_2d(x, y)
        if len(pidx) > 0:
            order = np.argsort(x[pidx])
            frontiers[mode_key] = (x[pidx][order], y[pidx][order])

    for a_key, (ax, ay) in frontiers.items():
        for b_key, (bx, by) in frontiers.items():
            if a_key == b_key:
                continue
            # Check if A dominates B: for each point on B, interpolate A's seq/s at that TTFT
            dominates = True
            for i in range(len(bx)):
                # A's seq/s at bx[i] (interpolate on A's frontier)
                a_interp = np.interp(bx[i], ax, ay, left=0, right=0)
                if a_interp < by[i]:
                    dominates = False
                    break
            if dominates and len(bx) > 0:
                print(f"    {MODE_STYLES[a_key]['label']} DOMINATES {MODE_STYLES[b_key]['label']}")


# =========================================================================
# Main
# =========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTFT Fairness Study for 8-GPU deployment modes.")
    parser.add_argument(
        "--concurrency", type=str, default=None,
        help="Comma-separated concurrency targets for Study 1 (default: 8,16,32,64,128,256,512)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cc_targets = DEFAULT_CONCURRENCY_TARGETS
    if args.concurrency:
        cc_targets = sorted(int(c.strip()) for c in args.concurrency.split(","))

    print(f"TTFT Fairness Study: {MODEL_PATH} on {SYSTEM} — {TOTAL_GPUS} GPUs")
    print(f"  Loading CSVs ...")
    mode_dfs = load_csvs()

    non_empty = sum(1 for df in mode_dfs.values() if not df.empty)
    if non_empty == 0:
        print("No data loaded. Run plot_8gpu_comparison.py first to generate CSVs.")
        sys.exit(1)
    print(f"  {non_empty}/{len(MODE_STYLES)} modes loaded.\n")

    # Study 1
    study_fixed_load_ttft(mode_dfs, cc_targets)

    # Study 2
    study_ttft_vs_throughput(mode_dfs)

    print("\n" + "=" * 60)
    print("  All outputs:")
    print(f"    ttft_study1_fixed_load_{SYSTEM}.csv")
    print(f"    ttft_study1_fixed_load_{SYSTEM}.png")
    print(f"    ttft_study1_fixed_load_{SYSTEM}.html")
    print(f"    ttft_study2_pareto_{SYSTEM}.png")
    print(f"    ttft_study2_pareto_seqs_{SYSTEM}.html")
    print(f"    ttft_study2_pareto_tokenss_{SYSTEM}.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
