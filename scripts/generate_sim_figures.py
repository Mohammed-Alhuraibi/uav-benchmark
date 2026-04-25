"""Generate report figures from a simulation CSV log.

Outputs:
  - fig13_error_path.png   — spec-error vs time, with envelope shaded and
                             phase background bands.
  - fig14_lock_timeline.png — Gantt-style phase timeline + 4s lock progress.
  - fig_simulation_hero.png — re-saved hero frame from the sim output.

Usage:
    python scripts/generate_sim_figures.py
        [--csv runs/sim/orbit.csv]
        [--video runs/sim/orbit.mp4]
        [--out report_figures/]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Phase color palette — matches the HUD palette so figures and video read
# the same.
PHASE_COLORS = {
    "SEARCHING": "#9ea0a3",
    "TRACKING":  "#ffcc33",
    "DROPOUT":   "#ff6b1a",
    "LOCKED":    "#23b85d",
}
PHASE_ORDER = ["SEARCHING", "TRACKING", "DROPOUT", "LOCKED"]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
    })


def load_log(csv_path: Path) -> dict:
    """Read the per-frame CSV and return columns as parallel lists."""
    cols: dict[str, list] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return cols


def to_floats(values: list[str]) -> np.ndarray:
    return np.array([float(v) if v else np.nan for v in values])


def to_bools(values: list[str]) -> np.ndarray:
    return np.array([v == "True" for v in values])


def shade_phases(ax, t: np.ndarray, phases: list[str], alpha: float = 0.15) -> None:
    """Color the axis background by phase. Each contiguous run = one rect."""
    if len(t) == 0:
        return
    start = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            x0 = t[start]
            x1 = t[i - 1] if i > start else x0
            color = PHASE_COLORS.get(phases[start], "#cccccc")
            ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0)
            start = i


def fig_error_path(cols: dict, out: Path) -> None:
    """Plot the AH-vs-target spec error with envelope and phase shading."""
    t = to_floats(cols["t_s"])
    t = t - t[0]  # zero-base time axis
    ex = to_floats(cols["err_x_px"])
    ey = to_floats(cols["err_y_px"])
    env_w = to_floats(cols["env_half_w_px"])
    env_h = to_floats(cols["env_half_h_px"])
    in_env = to_bools(cols["in_envelope"])
    phases = cols["phase"]

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

    for ax, err, env, label in [
        (ax_x, ex, env_w, "horizontal"),
        (ax_y, ey, env_h, "vertical"),
    ]:
        shade_phases(ax, t, phases)
        ax.fill_between(t, -env, env, color="#1f77b4", alpha=0.10,
                        label="spec envelope (±½ target dim)")
        ax.plot(t, err, color="#1f77b4", lw=1.4, label="AH center − target center")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_ylabel(f"{label} error (px)")

    ax_x.set_title("Spec-validity error: distance from AH centre to target centre")
    ax_y.set_xlabel("time (s)")

    # Fraction in envelope — informational, top-right
    n = len(in_env)
    pct = 100 * np.sum(in_env) / n if n else 0
    ax_x.text(0.99, 0.95, f"in envelope: {pct:.1f}% of frames",
              transform=ax_x.transAxes, ha="right", va="top",
              fontsize=9, color="#444",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, lw=0))

    # Phase legend (separate from line legend)
    phase_handles = [mpatches.Patch(facecolor=PHASE_COLORS[p], alpha=0.30, label=p)
                     for p in PHASE_ORDER if p in set(phases)]
    line_handles, line_labels = ax_x.get_legend_handles_labels()
    ax_x.legend(handles=line_handles + phase_handles,
                loc="lower right", ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_lock_timeline(cols: dict, out: Path) -> None:
    """Phase Gantt + 4s lock-progress curve overlay."""
    t = to_floats(cols["t_s"])
    t = t - t[0]
    elapsed = to_floats(cols["elapsed_s"])
    locked = to_bools(cols["locked"])
    phases = cols["phase"]

    fig, ax = plt.subplots(figsize=(10, 4.2))

    # Phase Gantt (one row of contiguous bars)
    y_bar = 0.5
    bar_h = 0.6
    start = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            x0 = t[start]
            x1 = t[i - 1] if i > start else x0
            color = PHASE_COLORS.get(phases[start], "#cccccc")
            ax.barh(y_bar, x1 - x0, left=x0, height=bar_h,
                    color=color, edgecolor="white", lw=0.3)
            start = i

    # Lock-engagement event markers (LOCKED rising edges)
    locked_starts = [t[i] for i in range(len(locked))
                     if locked[i] and (i == 0 or not locked[i - 1])]
    for x in locked_starts:
        ax.axvline(x, color=PHASE_COLORS["LOCKED"], lw=1.0, ls="--", alpha=0.6)
        ax.text(x, y_bar + bar_h / 2 + 0.02, "  LOCKED", color=PHASE_COLORS["LOCKED"],
                fontsize=8, va="bottom", ha="left")

    ax.set_yticks([y_bar])
    ax.set_yticklabels(["phase"])
    ax.set_ylim(-0.1, 1.65)

    # Overlay: lock-on progress (right axis)
    ax2 = ax.twinx()
    ax2.plot(t, elapsed, color="#444", lw=1.0, alpha=0.85,
             label="lock progress (s, resets on phase reset)")
    ax2.axhline(4.0, color="#23b85d", lw=1.0, ls=":", alpha=0.85,
                label="4 s lock threshold")
    ax2.set_ylabel("lock progress (s)")
    ax2.set_ylim(0, max(5.0, float(np.nanmax(elapsed)) + 1.0))
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    # Combined legend
    phase_handles = [mpatches.Patch(facecolor=PHASE_COLORS[p], label=p)
                     for p in PHASE_ORDER if p in set(phases)]
    line_h, line_l = ax2.get_legend_handles_labels()
    ax.legend(handles=phase_handles + line_h,
              loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=len(phase_handles) + len(line_h), fontsize=8)

    ax.set_xlabel("time (s)")
    ax.set_title("Lock-on state machine timeline")

    # Summary stats annotation
    n = len(phases)
    locked_n = int(np.sum(locked))
    cur_run = best_run = 0
    for v in locked:
        if v:
            cur_run += 1
            best_run = max(best_run, cur_run)
        else:
            cur_run = 0
    fps = (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 25.0
    ax.text(0.98, -0.30,
            f"locked frames: {locked_n} / {n} ({100 * locked_n / n:.1f}%)   "
            f"longest continuous lock: {best_run / fps:.2f} s",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#444")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def save_hero_frame(csv_path: Path, video_path: Path, out: Path,
                    src_fps: float = 25.0) -> None:
    """Pick the lock-acquisition frame (first locked=True of the longest run)."""
    rows = list(csv.DictReader(open(csv_path)))
    locked = [r["locked"] == "True" for r in rows]

    runs = []
    s = None
    for i, v in enumerate(locked):
        if v and s is None:
            s = i
        elif not v and s is not None:
            runs.append((s, i - s))
            s = None
    if s is not None:
        runs.append((s, len(locked) - s))
    if not runs:
        print("  no LOCKED frames — skipping hero figure")
        return
    longest = max(runs, key=lambda x: x[1])
    hero_idx = longest[0]  # 4 s lock-acquisition moment
    t_seek = hero_idx / src_fps
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{t_seek:.3f}", "-i", str(video_path),
        "-frames:v", "1", str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  ERROR: ffmpeg failed: {res.stderr[-300:]}")
        return
    print(f"  wrote {out}  (lock-acquisition frame {hero_idx}, t={t_seek:.2f}s)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "runs/sim/orbit.csv")
    parser.add_argument("--video", type=Path, default=ROOT / "runs/sim/orbit.mp4")
    parser.add_argument("--out", type=Path, default=ROOT / "report_figures")
    parser.add_argument("--src-fps", type=float, default=25.0,
                        help="Source video framerate (for hero-frame seek)")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    setup_style()
    args.out.mkdir(parents=True, exist_ok=True)

    cols = load_log(args.csv)
    print(f"  loaded {len(cols['frame_idx'])} rows from {args.csv}")

    fig_error_path(cols, args.out / "fig13_error_path.png")
    fig_lock_timeline(cols, args.out / "fig14_lock_timeline.png")

    if args.video.exists():
        save_hero_frame(args.csv, args.video, args.out / "fig_simulation_hero.png",
                        src_fps=args.src_fps)
    else:
        print(f"  WARNING: video not found ({args.video}); skipping hero frame")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
