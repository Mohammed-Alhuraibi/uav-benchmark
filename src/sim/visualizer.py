"""HUD renderer for the lock-on simulation.

Composites a single annotated frame from:
  - the raw camera frame (with on-frame zone/detection/AH overlays drawn)
  - a fixed-width side panel showing lock status, control gauges, telemetry

This module is *drawing-only* — no inference / Kalman logic. It consumes
the HUDState defined in src.sim.state.

Color choices follow competition spec where applicable:
  - AH (Lockdown Quadrilateral) is drawn in BGR (0,0,255) = #FF0000.
  - Server time is rendered HH:MM:SS.mmm at top-right (ms precision).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.sim.state import HUDState, phase_from_lock_state


# ---------------------------------------------------------------------------
# Type alias — keeps drawing-helper signatures readable.
# ---------------------------------------------------------------------------
Color = tuple[int, int, int]


# ---------------------------------------------------------------------------
# Palette (BGR). Keep these named — magic tuples in drawing code are unreadable.
# ---------------------------------------------------------------------------
RED:        Color = (0, 0, 255)        # AH rectangle (spec mandate #FF0000)
ORANGE:     Color = (0, 100, 255)
YELLOW:     Color = (0, 255, 255)      # AV zone, mid-confidence detections
GREEN:      Color = (0, 255, 0)        # lockable detection, locked state
LIME:       Color = (50, 255, 50)
CYAN:       Color = (255, 200, 0)      # AK (Camera FOV) frame edge
GRAY:       Color = (160, 160, 160)
DARK_GRAY:  Color = (50, 50, 50)
PANEL_BG:   Color = (28, 28, 28)
WHITE:      Color = (240, 240, 240)
BLACK:      Color = (0, 0, 0)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass
class HUDLayout:
    """Fixed pixel sizes for the side panel."""
    panel_w: int = 320
    title_h: int = 40
    lock_h: int = 80
    bar_h: int = 36
    bar_pad: int = 8
    section_pad: int = 12

    font: int = cv2.FONT_HERSHEY_SIMPLEX
    label_scale: float = 0.50
    value_scale: float = 0.55
    title_scale: float = 0.65
    big_scale: float = 0.95


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class HUDRenderer:
    """Composites the full operator HUD — source frame + side panel."""

    def __init__(self, frame_w: int, frame_h: int, layout: HUDLayout | None = None):
        self.fw = frame_w
        self.fh = frame_h
        self.layout = layout or HUDLayout()

    @property
    def output_size(self) -> tuple[int, int]:
        """(width, height) of the composite frame this renderer produces."""
        return (self.fw + self.layout.panel_w, self.fh)

    def render(self, frame: np.ndarray, state: HUDState) -> np.ndarray:
        """Return a composite frame: annotated source on the left, panel on the right."""
        if frame.shape[:2] != (self.fh, self.fw):
            raise ValueError(
                f"frame shape {frame.shape[:2]} does not match renderer "
                f"({self.fh}, {self.fw})"
            )

        canvas = frame.copy()
        self._draw_ak(canvas)
        self._draw_av(canvas, state)
        self._draw_detections(canvas, state.detections)
        self._draw_ah(canvas, state.kalman_bbox)
        self._draw_top_overlay(canvas, state)
        self._draw_bottom_overlay(canvas, state)

        panel = self._render_side_panel(state)
        return np.hstack([canvas, panel])

    # ------------------------------------------------------------------
    # On-frame overlays
    # ------------------------------------------------------------------

    def _draw_ak(self, canvas: np.ndarray) -> None:
        """Camera Field of View — implicit at the frame edge, drawn faintly."""
        cv2.rectangle(canvas, (1, 1), (self.fw - 2, self.fh - 2), CYAN, 1)
        cv2.putText(canvas, "AK", (6, self.fh - 8), self.layout.font,
                    0.4, CYAN, 1, cv2.LINE_AA)

    def _draw_av(self, canvas: np.ndarray, state: HUDState) -> None:
        """Target Hit Area — yellow rectangle, 50% W × 80% H by default."""
        x1, y1, x2, y2 = self._frac_box(state.av_w_frac, state.av_h_frac)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), YELLOW, 1)
        cv2.putText(canvas, "AV", (x1 + 4, y1 + 16), self.layout.font,
                    0.45, YELLOW, 1, cv2.LINE_AA)

    def _draw_ah(
        self, canvas: np.ndarray,
        bbox: tuple[float, float, float, float] | None,
    ) -> None:
        """Lockdown Quadrilateral — red #FF0000, ≤2px line per spec.

        Drawn from the Kalman-smoothed bbox (so it doesn't jitter on raw
        YOLO noise). Skipped when the filter isn't initialized yet.
        """
        if bbox is None:
            return
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), RED, 2)
        cv2.putText(canvas, "AH", (x1 + 2, y1 - 6), self.layout.font,
                    0.45, RED, 1, cv2.LINE_AA)

    def _draw_detections(self, canvas: np.ndarray, dets: list[dict]) -> None:
        """Color-code raw YOLO detections.

        green  = lockable (size + conf + zone OK)
        yellow = detected but below lock threshold
        red    = too small (< min_bbox_ratio)
        """
        for d in dets:
            x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
            if d.get("lockable"):
                color = GREEN
                tag = "LOCK"
            elif d.get("meets_size", True):
                color = YELLOW
                tag = "DET"
            else:
                color = ORANGE
                tag = "SMALL"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
            tid = d.get("track_id")
            label = f"#{tid} {tag} {d.get('confidence', 0):.2f}" if tid is not None \
                else f"{tag} {d.get('confidence', 0):.2f}"
            # Prefer below-bbox; fall back to above when there's no room below.
            label_y = y2 + 14 if y2 + 14 <= self.fh - 4 else max(y1 - 6, 14)
            cv2.putText(canvas, label, (x1, label_y),
                        self.layout.font, 0.4, color, 1, cv2.LINE_AA)

    def _draw_top_overlay(self, canvas: np.ndarray, state: HUDState) -> None:
        """Top strip: FPS + frame idx + det count (left), server time (right)."""
        # Top-left
        n_lockable = sum(1 for d in state.detections if d.get("lockable"))
        left = (
            f"FPS {state.fps:5.1f}  frame {state.frame_idx:5d}  "
            f"det {len(state.detections)} (lock {n_lockable})  "
            f"inf {state.inference_ms:4.1f}ms"
        )
        self._text_with_shadow(canvas, left, (8, 22), self.layout.label_scale, WHITE)

        # Top-right (server time, ms precision)
        time_str = self._format_server_time(state.timestamp_s)
        (tw, _), _ = cv2.getTextSize(time_str, self.layout.font, 0.6, 2)
        self._text_with_shadow(canvas, time_str, (self.fw - tw - 10, 24), 0.6, WHITE)

    def _draw_bottom_overlay(self, canvas: np.ndarray, state: HUDState) -> None:
        """4s progress bar (centered) + dropout countdown + spec envelope readout."""
        ls = state.lock_state
        progress = float(ls.get("progress", 0.0))
        elapsed = float(ls.get("elapsed_s", 0.0))
        dropout_s = float(ls.get("dropout_s", 0.0))
        locked = bool(ls.get("locked", False))
        in_zone = bool(ls.get("in_zone", False))
        phase = phase_from_lock_state(ls)

        bar_w = min(int(self.fw * 0.6), 480)
        bar_h = 26
        bar_x = (self.fw - bar_w) // 2
        bar_y = self.fh - bar_h - 18

        # Semi-transparent backplate
        overlay = canvas.copy()
        cv2.rectangle(overlay, (bar_x - 4, bar_y - 24),
                      (bar_x + bar_w + 4, bar_y + bar_h + 26), BLACK, -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        # Bar fill color by phase
        fill = {"LOCKED": GREEN, "DROPOUT": ORANGE,
                "TRACKING": YELLOW, "SEARCHING": GRAY}[phase]

        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), DARK_GRAY, -1)
        fill_w = int(bar_w * max(0.0, min(progress, 1.0)))
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), fill, -1)
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), WHITE, 1)

        # Status text inside bar
        status = (f"{phase}  {elapsed:.2f}/4.00s"
                  if not locked else f"{phase}  {elapsed:.2f}s")
        cv2.putText(canvas, status, (bar_x + 8, bar_y + 18),
                    self.layout.font, 0.55, WHITE, 1, cv2.LINE_AA)

        # Zone status above bar
        zone_text = "IN AH ZONE" if in_zone else "OUT OF AH"
        zone_color = GREEN if in_zone else ORANGE
        self._text_with_shadow(canvas, zone_text,
                               (bar_x, bar_y - 6), 0.45, zone_color)

        # Bottom-left: dropout countdown when in gap
        if dropout_s > 0:
            self._text_with_shadow(
                canvas,
                f"DROPOUT {dropout_s * 1000:5.0f} / 200 ms",
                (10, self.fh - 12), 0.5, ORANGE,
            )

        # Bottom-right: spec envelope (AH-center vs target-center) per spec p.14
        if state.spec_error_px is not None and state.spec_envelope_half_px is not None:
            ex_px, ey_px = state.spec_error_px
            hw, hh = state.spec_envelope_half_px
            color = GREEN if state.in_envelope else RED
            # ASCII only — OpenCV HERSHEY fonts don't support Unicode (no Δ/±).
            text = (f"err ({ex_px:+5.1f},{ey_px:+5.1f})px"
                    f"  env (+/-{hw:.1f},+/-{hh:.1f})")
            (tw, _), _ = cv2.getTextSize(text, self.layout.font, 0.45, 1)
            self._text_with_shadow(
                canvas, text,
                (self.fw - tw - 10, self.fh - 12), 0.45, color,
            )

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------

    def _render_side_panel(self, state: HUDState) -> np.ndarray:
        """Build the right-side gauge panel as a fresh BGR canvas."""
        L = self.layout
        panel = np.full((self.fh, L.panel_w, 3), PANEL_BG, dtype=np.uint8)

        y = 0
        y = self._panel_title(panel, y, state)
        y = self._panel_lock_status(panel, y, state)
        y = self._panel_gauges(panel, y, state)
        y = self._panel_telemetry(panel, y, state)
        return panel

    def _panel_title(self, panel: np.ndarray, y: int, state: HUDState) -> int:
        L = self.layout
        cv2.rectangle(panel, (0, y), (L.panel_w, y + L.title_h), DARK_GRAY, -1)
        cv2.putText(panel, "OPERATOR HUD", (10, y + 26),
                    L.font, L.title_scale, WHITE, 2, cv2.LINE_AA)
        cv2.putText(panel, state.model_name,
                    (L.panel_w - 110, y + 24),
                    L.font, 0.45, GRAY, 1, cv2.LINE_AA)
        return y + L.title_h

    def _panel_lock_status(self, panel: np.ndarray, y: int, state: HUDState) -> int:
        L = self.layout
        ls = state.lock_state
        phase = phase_from_lock_state(ls)
        color = {"LOCKED": GREEN, "DROPOUT": ORANGE,
                 "TRACKING": YELLOW, "SEARCHING": GRAY}[phase]

        cv2.rectangle(panel, (8, y + 6), (L.panel_w - 8, y + L.lock_h - 6),
                      DARK_GRAY, -1)
        cv2.rectangle(panel, (8, y + 6), (L.panel_w - 8, y + L.lock_h - 6),
                      color, 2)
        cv2.putText(panel, phase, (20, y + 50),
                    L.font, L.big_scale, color, 2, cv2.LINE_AA)

        # Sub-line: progress fraction
        prog = ls.get("progress", 0.0)
        elapsed = ls.get("elapsed_s", 0.0)
        sub = f"{elapsed:.2f}s / 4.00s   {prog*100:5.1f}%"
        cv2.putText(panel, sub, (20, y + 70),
                    L.font, 0.45, WHITE, 1, cv2.LINE_AA)
        return y + L.lock_h + L.section_pad

    def _panel_gauges(self, panel: np.ndarray, y: int, state: HUDState) -> int:
        """Roll, pitch, yaw, throttle bars."""
        L = self.layout
        c = state.control

        # If no controller output yet, show zero bars in gray
        roll = pitch = yaw = thr = 0.0
        if c is not None:
            roll, pitch, yaw, thr = c.roll, c.pitch, c.yaw, c.throttle

        bars = [
            ("ROLL",  roll,  -1.0, +1.0, self._color_for_value(roll)),
            ("PITCH", pitch, -1.0, +1.0, self._color_for_value(pitch)),
            ("YAW",   yaw,   -1.0, +1.0, self._color_for_value(yaw)),
            ("THR",   thr,    0.0, +1.0, GREEN if 0.3 < thr < 0.85 else ORANGE),
        ]
        for label, val, vmin, vmax, color in bars:
            self._draw_horizontal_bar(panel, y, label, val, vmin, vmax, color)
            y += L.bar_h + L.bar_pad
        return y + L.section_pad

    def _panel_telemetry(self, panel: np.ndarray, y: int, state: HUDState) -> int:
        L = self.layout
        ls = state.lock_state
        c = state.control

        lines = []
        tid = ls.get("target_id")
        lines.append(f"Track ID:  {tid if tid is not None else '-'}")
        n_lock = sum(1 for d in state.detections if d.get("lockable"))
        lines.append(f"Detections: {len(state.detections)} (lockable {n_lock})")

        if c is not None:
            lines.append(f"Size ratio: {c.size_ratio*100:5.2f}%")
            lines.append(f"Err norm:   ({c.error_x_norm:+.2f}, {c.error_y_norm:+.2f})")
        else:
            lines.append("Size ratio: -")
            lines.append("Err norm:   -")

        env_text = "OK" if state.in_envelope else "OUT"
        env_color = GREEN if state.in_envelope else RED
        lines.append(f"Spec env:   {env_text}")

        # Background for telemetry block
        block_h = 18 * len(lines) + 14
        cv2.rectangle(panel, (8, y), (L.panel_w - 8, y + block_h), DARK_GRAY, -1)
        for i, line in enumerate(lines):
            color = env_color if line.startswith("Spec env") else WHITE
            cv2.putText(panel, line, (16, y + 18 + i * 18),
                        L.font, 0.45, color, 1, cv2.LINE_AA)
        return y + block_h + L.section_pad

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _draw_horizontal_bar(
        self, panel: np.ndarray, y: int, label: str,
        value: float, vmin: float, vmax: float, color: Color,
    ) -> None:
        L = self.layout
        x = 60
        w = L.panel_w - x - 60
        h = L.bar_h

        # Backplate
        cv2.rectangle(panel, (x, y), (x + w, y + h), DARK_GRAY, -1)
        cv2.rectangle(panel, (x, y), (x + w, y + h), GRAY, 1)

        # Zero line for signed bars
        if vmin < 0 < vmax:
            zero_x = x + int(round(w * (-vmin) / (vmax - vmin)))
            cv2.line(panel, (zero_x, y + 2), (zero_x, y + h - 2), WHITE, 1)

        # Fill
        v = max(vmin, min(vmax, value))
        if vmin < 0 < vmax:
            zero_x = x + int(round(w * (-vmin) / (vmax - vmin)))
            val_x = x + int(round(w * (v - vmin) / (vmax - vmin)))
            x0, x1 = sorted((zero_x, val_x))
            cv2.rectangle(panel, (x0, y + 3), (x1, y + h - 3), color, -1)
        else:
            val_x = x + int(round(w * (v - vmin) / (vmax - vmin)))
            cv2.rectangle(panel, (x + 1, y + 3), (val_x, y + h - 3), color, -1)

        # Label (left)
        cv2.putText(panel, label, (8, y + h // 2 + 5),
                    L.font, L.label_scale, WHITE, 1, cv2.LINE_AA)
        # Value (right)
        val_text = f"{value:+.2f}" if vmin < 0 else f"{value:.2f}"
        cv2.putText(panel, val_text, (x + w + 6, y + h // 2 + 5),
                    L.font, L.value_scale, WHITE, 1, cv2.LINE_AA)

    def _text_with_shadow(
        self, img: np.ndarray, text: str, org: tuple[int, int],
        scale: float, color: Color,
    ) -> None:
        """Anti-aliased text with 1px black shadow — keeps overlays readable on
        any background.
        """
        cv2.putText(img, text, (org[0] + 1, org[1] + 1),
                    self.layout.font, scale, BLACK, 2, cv2.LINE_AA)
        cv2.putText(img, text, org,
                    self.layout.font, scale, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _frac_box(self, w_frac: float, h_frac: float) -> tuple[int, int, int, int]:
        cx, cy = self.fw / 2, self.fh / 2
        bw, bh = self.fw * w_frac, self.fh * h_frac
        return (
            int(round(cx - bw / 2)),
            int(round(cy - bh / 2)),
            int(round(cx + bw / 2)),
            int(round(cy + bh / 2)),
        )

    @staticmethod
    def _format_server_time(timestamp_s: float) -> str:
        """HH:MM:SS.mmm. Wraps to 24h."""
        ts = max(0.0, timestamp_s)
        whole = int(ts)
        ms = int(round((ts - whole) * 1000))
        if ms == 1000:  # rounding edge
            whole += 1
            ms = 0
        h = (whole // 3600) % 24
        m = (whole // 60) % 60
        s = whole % 60
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    @staticmethod
    def _color_for_value(v: float) -> Color:
        """Green at zero, yellow at moderate, red at saturation."""
        a = abs(v)
        if a < 0.25:
            return GREEN
        if a < 0.7:
            return YELLOW
        return RED
