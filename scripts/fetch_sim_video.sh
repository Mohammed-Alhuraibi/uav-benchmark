#!/usr/bin/env bash
# Fetch and crop simulation source videos for the lock-on demo.
#
# Source clips are not committed to git (data/sim/raw, data/sim/clips are
# gitignored). This script makes them reproducible from URL + crop spec.
#
# Requires: yt-dlp (in .venv), ffmpeg (system).
#   source .venv/bin/activate && pip install -U yt-dlp
#
# Usage:
#   ./scripts/fetch_sim_video.sh                 # fetch all clips
#   ./scripts/fetch_sim_video.sh p51_dogfight    # fetch one by name

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/sim/raw"
CLIPS="$ROOT/data/sim/clips"
mkdir -p "$RAW" "$CLIPS"

# yt-dlp must come from the venv (not system python)
YTDLP="$ROOT/.venv/bin/yt-dlp"
if [[ ! -x "$YTDLP" ]]; then
  echo "ERROR: $YTDLP not found. Activate venv and run: pip install -U yt-dlp" >&2
  exit 1
fi

# Clip registry: name|url|yt-dlp format|crop (W:H:X:Y, empty for none)|trim (start-end, empty for full)
CLIPS_DEF=(
  # P-51D Mustang gun camera, WW2 (US gov public domain).
  # Vertical Short 720x1280; crop removes pilot info overlay (top 347px) and
  # black letterbox bar (bottom ~230px), leaving 720x680 of clean sky+target.
  "p51_dogfight|https://www.youtube.com/shorts/qR53ahoTNOQ|298+140|720:680:0:347|"

  # 8KM long-range FPV chase, ATOMRC Beluga (in-distribution color footage).
  # 720p25 horizontal; crop removes the lower picture-in-picture strip with
  # Mark's POV / pilots / Ben's POV, leaving 1280x454 of clean sky+target.
  "beluga_8km|https://www.youtube.com/watch?v=t7wFT61LVxU|136|1280:455:0:0|3:40-4:30"

  # Same source, 7:28-7:48: chase drone orbits the target. Target stays
  # large and roughly centered — the favored window for the lock-on demo.
  "beluga_orbit|https://www.youtube.com/watch?v=t7wFT61LVxU|136|1280:455:0:0|7:28-7:48"
)

fetch_one() {
  local name="$1" url="$2" fmt="$3" crop="$4" trim="$5"
  local raw="$RAW/$name.mp4"
  local out="$CLIPS/$name.mp4"

  if [[ -f "$out" ]]; then
    echo "[skip] $name (already exists at $out)"
    return 0
  fi

  echo "[fetch] $name <- $url"
  "$YTDLP" -f "$fmt" --merge-output-format mp4 -o "$RAW/$name.%(ext)s" "$url"

  local vf=""
  [[ -n "$crop" ]] && vf="crop=$crop"

  local ss_to=()
  if [[ -n "$trim" ]]; then
    local start="${trim%-*}" end="${trim#*-}"
    ss_to=(-ss "$start" -to "$end")
  fi

  echo "[crop]  $name -> $out"
  ffmpeg -y "${ss_to[@]}" -i "$raw" \
    ${vf:+-vf "$vf"} \
    -c:v libx264 -pix_fmt yuv420p -preset medium -crf 18 -an \
    "$out" 2>&1 | tail -1
}

target="${1:-}"
for entry in "${CLIPS_DEF[@]}"; do
  IFS='|' read -r name url fmt crop trim <<< "$entry"
  if [[ -z "$target" || "$target" == "$name" ]]; then
    fetch_one "$name" "$url" "$fmt" "$crop" "$trim"
  fi
done

echo
echo "Done. Clips ready in: $CLIPS"
ls -lh "$CLIPS"
