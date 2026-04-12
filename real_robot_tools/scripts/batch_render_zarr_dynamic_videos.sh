#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python"
OUTPUT_DIR="visualizations/multi_zarr"
EPISODE_IDX=0
FPS=30
MAX_FRAMES=0
SOURCES_FILE=""
SOURCE_ITEMS=()
EXTRA_FLAGS=()

usage() {
  cat <<'EOF'
Batch render multiple zarr sources into dynamic point-cloud videos.

Usage:
  bash real_robot_to_3dpolicy_tools/batch_render_zarr_dynamic_videos.sh \
    --source drawer_single=data/real_data/drawer_new_single-50.zarr \
    --source drawer_stereo=data/real_data/drawer_new-50.zarr \
    --output-dir visualizations/multi_zarr \
    --episode-idx 0 \
    --fps 30

Or provide a source list file:
  bash real_robot_to_3dpolicy_tools/batch_render_zarr_dynamic_videos.sh \
    --sources-file real_robot_to_3dpolicy_tools/data/source_paths.example.txt \
    --output-dir visualizations/multi_zarr

Source item format:
  name=/path/to/file.zarr

Optional:
  --max-frames 120
  --dynamic-axis
  --auto-crop
  --cleanup-frames
  --python-bin python3
EOF
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

safe_name() {
  local s="$1"
  printf '%s' "$s" | sed 's/[^A-Za-z0-9._-]/_/g'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_ITEMS+=("$2")
      shift 2
      ;;
    --sources-file)
      SOURCES_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --episode-idx)
      EPISODE_IDX="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dynamic-axis)
      EXTRA_FLAGS+=("--dynamic-axis")
      shift
      ;;
    --auto-crop)
      EXTRA_FLAGS+=("--auto-crop")
      shift
      ;;
    --cleanup-frames)
      EXTRA_FLAGS+=("--cleanup-frames")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "${SOURCES_FILE}" ]]; then
  if [[ ! -f "${SOURCES_FILE}" ]]; then
    echo "❌ sources file not found: ${SOURCES_FILE}" >&2
    exit 1
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(trim "$line")"
    [[ -z "$line" ]] && continue
    [[ "$line" == \#* ]] && continue
    SOURCE_ITEMS+=("$line")
  done < "${SOURCES_FILE}"
fi

if [[ "${#SOURCE_ITEMS[@]}" -eq 0 ]]; then
  echo "❌ No sources provided. Use --source name=path or --sources-file." >&2
  usage
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

processed=0
skipped=0

for item in "${SOURCE_ITEMS[@]}"; do
  if [[ "$item" != *=* ]]; then
    echo "⚠️  Skip invalid source item (expect name=path): $item"
    skipped=$((skipped + 1))
    continue
  fi

  source_name="$(trim "${item%%=*}")"
  zarr_path="$(trim "${item#*=}")"

  if [[ -z "${source_name}" || -z "${zarr_path}" ]]; then
    echo "⚠️  Skip invalid source item (empty name/path): $item"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ ! -e "${zarr_path}" ]]; then
    echo "⚠️  Skip ${source_name}: file not found -> ${zarr_path}"
    skipped=$((skipped + 1))
    continue
  fi

  output_name="$(safe_name "${source_name}")_ep${EPISODE_IDX}"
  cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/render_zarr_dynamic_video.py"
    --zarr-path "${zarr_path}"
    --output-dir "${OUTPUT_DIR}"
    --episode-idx "${EPISODE_IDX}"
    --fps "${FPS}"
    --output-name "${output_name}"
  )

  if [[ "${MAX_FRAMES}" -gt 0 ]]; then
    cmd+=(--max-frames "${MAX_FRAMES}")
  fi

  cmd+=("${EXTRA_FLAGS[@]}")

  echo "▶ ${source_name} -> ${zarr_path}"
  "${cmd[@]}"
  processed=$((processed + 1))
done

echo "✅ Batch rendering completed. processed=${processed}, skipped=${skipped}, output_dir=${OUTPUT_DIR}"
