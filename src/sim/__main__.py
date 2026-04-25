"""CLI entrypoint for the lock-on simulation.

Usage:
    python -m src.sim --video data/sim/clips/p51_dogfight.mp4

    # Headless, custom output paths
    python -m src.sim \\
        --video data/sim/clips/p51_dogfight.mp4 \\
        --out runs/sim/p51.mp4 \\
        --csv-out runs/sim/p51.csv \\
        --jsonl-out runs/sim/p51.jsonl \\
        --no-display

    # Stream telemetry to terminals (run sim in one, jq in others)
    python -m src.sim ... --jsonl-out logs/sim.jsonl
    tail -f logs/sim.jsonl | jq -r 'select(.ch=="lock") | "\\(.t)  \\(.from)->\\(.to)"'
    tail -f logs/sim.jsonl | jq -r 'select(.ch=="ctrl") | "\\(.t)  r=\\(.r) p=\\(.p) y=\\(.y) th=\\(.th)"'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.pipeline import RunConfig, SimulationPipeline, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the UAV lock-on simulation on a source video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", type=Path, required=True,
                        help="Source video (e.g. data/sim/clips/p51_dogfight.mp4)")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "runs" / "yolo11s_p2" / "weights" / "best.pt",
                        help="Model weights (.pt). Default: yolo11s_p2 best.")
    parser.add_argument("--deployment-config", type=Path,
                        default=ROOT / "configs" / "deployment.yaml",
                        help="Path to deployment.yaml (camera/competition rules).")
    parser.add_argument("--sim-config", type=Path,
                        default=ROOT / "configs" / "simulation.yaml",
                        help="Path to simulation.yaml (Kalman/controller/HUD).")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs" / "sim" / "sim_output.mp4",
                        help="Annotated video output path.")
    parser.add_argument("--no-out", action="store_true",
                        help="Skip writing the annotated video.")
    parser.add_argument("--csv-out", type=Path,
                        default=ROOT / "runs" / "sim" / "sim_log.csv",
                        help="Per-frame CSV output.")
    parser.add_argument("--no-csv-out", action="store_true",
                        help="Skip writing the CSV log.")
    parser.add_argument("--jsonl-out", type=Path,
                        default=ROOT / "runs" / "sim" / "sim_telemetry.jsonl",
                        help="JSONL telemetry output.")
    parser.add_argument("--no-jsonl-out", action="store_true",
                        help="Skip writing the JSONL telemetry.")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless — don't pop up an OpenCV window.")
    parser.add_argument("--simulate-camera", action="store_true",
                        help="Resize input to Pi Camera Module 3 resolution.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (for fast iteration).")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 1

    deployment_cfg = load_yaml(args.deployment_config)
    sim_cfg = load_yaml(args.sim_config)

    pipeline = SimulationPipeline(
        model_path=args.model,
        deployment_cfg=deployment_cfg,
        sim_cfg=sim_cfg,
    )

    run_cfg = RunConfig(
        video_path=args.video,
        out_video_path=None if args.no_out else args.out,
        csv_path=None if args.no_csv_out else args.csv_out,
        jsonl_path=None if args.no_jsonl_out else args.jsonl_out,
        display=not args.no_display,
        simulate_camera=args.simulate_camera,
        max_frames=args.max_frames,
    )

    print(f"  Source video:  {args.video}")
    print(f"  Model:         {args.model}")
    print(f"  Output video:  {run_cfg.out_video_path}")
    print(f"  CSV log:       {run_cfg.csv_path}")
    print(f"  JSONL log:     {run_cfg.jsonl_path}")
    print(f"  Display:       {'ON' if run_cfg.display else 'OFF'}")
    print(f"  Camera sim:    {'ON' if run_cfg.simulate_camera else 'OFF'}")
    print(f"  Max frames:    {run_cfg.max_frames or 'all'}")
    print()

    summary = pipeline.run(run_cfg)
    print()
    print("  Summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
