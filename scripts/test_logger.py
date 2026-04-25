"""Smoke test for src.sim.logger.

Synthesizes a 10-frame run that walks through SEARCHING -> TRACKING ->
DROPOUT -> TRACKING -> LOCKED, then verifies:

  - CSV has the right schema and row count.
  - JSONL contains expected channels.
  - 'lock' events fire only on phase transitions.
  - 'det' / 'ctrl' channels can be selectively disabled.
  - None-valued fields write as empty CSV cells (no crash).

Run:
    python scripts/test_logger.py
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim.controller import ControlCommand
from src.sim.logger import FRAME_COLUMNS, FrameLogger, LoggerConfig
from src.sim.state import HUDState


def make_state(idx: int, phase: str, t: float) -> HUDState:
    """Synthetic HUDState that exercises each phase."""
    common = dict(timestamp_s=t, frame_idx=idx, fps=58.0, inference_ms=15.0)

    if phase == "SEARCHING":
        return HUDState(**common, detections=[],
                        lock_state={"locked": False, "progress": 0.0, "elapsed_s": 0.0,
                                    "dropout_s": 0.0, "target_id": None, "in_zone": False})
    if phase == "TRACKING":
        det = {"bbox": [300, 320, 360, 360], "confidence": 0.85,
               "lockable": True, "meets_size": True, "track_id": 1}
        return HUDState(**common, detections=[det],
                        kalman_bbox=(298.0, 321.0, 358.0, 361.0),
                        lock_state={"locked": False, "progress": 0.5, "elapsed_s": 2.0,
                                    "dropout_s": 0.0, "target_id": 1, "in_zone": True},
                        control=ControlCommand(0.1, -0.05, 0.04, 0.6, 0.1, -0.02, 0.083),
                        spec_error_px=(2.0, -1.0),
                        spec_envelope_half_px=(30.0, 20.0),
                        in_envelope=True)
    if phase == "DROPOUT":
        return HUDState(**common, detections=[],
                        kalman_bbox=(305.0, 322.0, 365.0, 362.0),
                        lock_state={"locked": False, "progress": 0.65, "elapsed_s": 2.6,
                                    "dropout_s": 0.087, "target_id": 1, "in_zone": False},
                        control=ControlCommand(0.12, -0.05, 0.04, 0.62, 0.12, -0.02, 0.083),
                        spec_error_px=(7.0, -1.0),
                        spec_envelope_half_px=(30.0, 20.0),
                        in_envelope=True)
    if phase == "LOCKED":
        det = {"bbox": [300, 320, 360, 360], "confidence": 0.93,
               "lockable": True, "meets_size": True, "track_id": 1}
        return HUDState(**common, detections=[det],
                        kalman_bbox=(299.0, 320.0, 359.0, 360.0),
                        lock_state={"locked": True, "progress": 1.0, "elapsed_s": 4.05,
                                    "dropout_s": 0.0, "target_id": 1, "in_zone": True},
                        control=ControlCommand(0.02, 0.0, 0.01, 0.55, 0.02, 0.0, 0.083),
                        spec_error_px=(1.0, 0.0),
                        spec_envelope_half_px=(30.0, 20.0),
                        in_envelope=True)
    raise ValueError(phase)


# A 10-frame timeline: searching, then track, then a dropout in the middle,
# then lock.
TIMELINE = [
    (0, "SEARCHING"),
    (1, "SEARCHING"),
    (2, "TRACKING"),
    (3, "TRACKING"),
    (4, "DROPOUT"),
    (5, "TRACKING"),
    (6, "TRACKING"),
    (7, "TRACKING"),
    (8, "LOCKED"),
    (9, "LOCKED"),
]


def test_full_pipeline(tmpdir: Path) -> None:
    print("[test 1] full timeline write -> read back CSV + JSONL")
    csv_path = tmpdir / "frames.csv"
    jsonl_path = tmpdir / "telemetry.jsonl"

    with FrameLogger(LoggerConfig(csv_path=csv_path, jsonl_path=jsonl_path)) as log:
        log.write_meta(model="yolo11s_p2", source="test.mp4",
                       config={"lock_on_seconds": 4.0})
        for idx, phase in TIMELINE:
            log.write_frame(make_state(idx, phase, t=idx / 60.0))

    # CSV checks
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == FRAME_COLUMNS, "CSV schema mismatch"
        rows = list(reader)
    assert len(rows) == len(TIMELINE), f"expected {len(TIMELINE)} rows, got {len(rows)}"
    # First row is SEARCHING — has empty bbox cells (no detection)
    assert rows[0]["target_id"] == "" and rows[0]["target_x1"] == "", \
        "SEARCHING row should have blank target fields"
    # TRACKING row has populated kalman bbox
    assert rows[2]["kalman_x1"] != "", "TRACKING row should have kalman bbox"
    # LOCKED row has locked=True
    assert rows[8]["locked"] == "True"
    print(f"   CSV ok — {len(rows)} rows, {len(FRAME_COLUMNS)} columns")

    # JSONL checks
    with open(jsonl_path) as f:
        events = [json.loads(line) for line in f if line.strip()]
    by_ch: dict[str, list] = {}
    for e in events:
        by_ch.setdefault(e["ch"], []).append(e)

    print(f"   JSONL channels: " + ", ".join(
        f"{k}={len(v)}" for k, v in sorted(by_ch.items())
    ))

    assert "meta" in by_ch and len(by_ch["meta"]) == 1
    assert "frame" in by_ch and len(by_ch["frame"]) == len(TIMELINE)
    # Detections only on frames with detections — 5 TRACKING + 2 LOCKED = 7
    # (DROPOUT and SEARCHING frames carry empty detection lists by construction)
    assert len(by_ch["det"]) == 7, f"expected 7 det events, got {len(by_ch['det'])}"
    # Control on every frame except SEARCHING (no controller output yet) — 8 frames
    assert len(by_ch["ctrl"]) == 8, f"expected 8 ctrl events, got {len(by_ch['ctrl'])}"

    # Lock transitions: None->SEARCHING, SEARCHING->TRACKING, TRACKING->DROPOUT,
    # DROPOUT->TRACKING, TRACKING->LOCKED  =  5 transitions
    transitions = [(e["from"], e["to"]) for e in by_ch["lock"]]
    expected = [
        (None, "SEARCHING"),
        ("SEARCHING", "TRACKING"),
        ("TRACKING", "DROPOUT"),
        ("DROPOUT", "TRACKING"),
        ("TRACKING", "LOCKED"),
    ]
    assert transitions == expected, f"transitions wrong: {transitions}"
    print(f"   transitions: {transitions}")
    print("   OK")


def test_channel_disable(tmpdir: Path) -> None:
    print("[test 2] disable det+ctrl channels — only frame+lock+meta survive")
    jsonl_path = tmpdir / "filtered.jsonl"
    cfg = LoggerConfig(jsonl_path=jsonl_path,
                       channels=frozenset({"meta", "frame", "lock"}))
    with FrameLogger(cfg) as log:
        log.write_meta(model="x")
        for idx, phase in TIMELINE:
            log.write_frame(make_state(idx, phase, t=idx / 60.0))

    with open(jsonl_path) as f:
        channels = {json.loads(line)["ch"] for line in f if line.strip()}
    assert channels == {"meta", "frame", "lock"}, f"unexpected channels: {channels}"
    print(f"   channels seen: {sorted(channels)}")
    print("   OK")


def test_csv_only(tmpdir: Path) -> None:
    print("[test 3] csv-only mode — no jsonl file created")
    csv_path = tmpdir / "csv_only.csv"
    with FrameLogger(LoggerConfig(csv_path=csv_path)) as log:
        log.write_frame(make_state(0, "TRACKING", t=0.0))
    assert csv_path.exists()
    assert not (tmpdir / "csv_only.jsonl").exists()
    print(f"   csv exists, jsonl absent")
    print("   OK")


def test_no_csv_no_crash(tmpdir: Path) -> None:
    print("[test 4] jsonl-only mode — no csv file created, no crash")
    jsonl_path = tmpdir / "j_only.jsonl"
    with FrameLogger(LoggerConfig(jsonl_path=jsonl_path)) as log:
        log.write_frame(make_state(0, "SEARCHING", t=0.0))
    assert jsonl_path.exists()
    print("   OK")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_full_pipeline(tmp)
        test_channel_disable(tmp)
        test_csv_only(tmp)
        test_no_csv_no_crash(tmp)
    print("\nAll tests passed.")
