#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./download.sh [options]

Downloads JSON metadata from Hugging Face, converts to dialogue,
and downloads videos with periodic uploads back to Hugging Face.

Options:
  --hf-repo NAME         Hugging Face dataset repo (required)
  --repo-type TYPE       Hugging Face repo type (default: dataset)
  --data-dir PATH        Base data directory (default: data)
  --raw-dir PATH         Where extracted JSON lives (default: data/raw)
  --converted-dir PATH   Where converted JSON is written (default: data/converted)
  --video-dir PATH       Where videos are stored (default: data/videos)
  --video-jobs N         Parallel video download workers (default: 8)
  --upload-threshold-gb N  Upload tar when size exceeds this GB (default: 8)
  --skip-convert         Skip JSON conversion
  --skip-videos          Skip video downloads
  -h, --help             Show this help
EOF
}

DATA_DIR="${DATA_DIR:-data}"
HF_REPO="${HF_REPO:-}"
REPO_TYPE="${REPO_TYPE:-dataset}"
RAW_DIR="${RAW_DIR:-}"
CONVERTED_DIR="${CONVERTED_DIR:-}"
VIDEO_DIR="${VIDEO_DIR:-}"
VIDEO_JOBS="${VIDEO_JOBS:-8}"
UPLOAD_THRESHOLD_GB="${UPLOAD_THRESHOLD_GB:-8}"
SKIP_CONVERT=0
SKIP_VIDEOS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-repo)
      HF_REPO="$2"
      shift 2
      ;;
    --repo-type)
      REPO_TYPE="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --raw-dir)
      RAW_DIR="$2"
      shift 2
      ;;
    --converted-dir)
      CONVERTED_DIR="$2"
      shift 2
      ;;
    --video-dir)
      VIDEO_DIR="$2"
      shift 2
      ;;
    --video-jobs)
      VIDEO_JOBS="$2"
      shift 2
      ;;
    --upload-threshold-gb)
      UPLOAD_THRESHOLD_GB="$2"
      shift 2
      ;;
    --skip-convert)
      SKIP_CONVERT=1
      shift
      ;;
    --skip-videos)
      SKIP_VIDEOS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

RAW_DIR="${RAW_DIR:-$DATA_DIR/raw}"
CONVERTED_DIR="${CONVERTED_DIR:-$DATA_DIR/converted}"
VIDEO_DIR="${VIDEO_DIR:-$DATA_DIR/videos}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

mkdir -p "$RAW_DIR"
if [[ "$SKIP_CONVERT" -eq 0 ]]; then
  mkdir -p "$CONVERTED_DIR"
fi
if [[ "$SKIP_VIDEOS" -eq 0 ]]; then
  mkdir -p "$VIDEO_DIR"
fi

if [[ -z "$HF_REPO" ]]; then
  echo "Missing required --hf-repo NAME."
  exit 1
fi

require_cmd python
echo "Downloading JSON metadata from Hugging Face..."
python scripts/hf_json_sync.py \
  --repo "$HF_REPO" \
  --repo-type "$REPO_TYPE" \
  --raw-dir "$RAW_DIR" \
  --download

if [[ "$SKIP_CONVERT" -eq 0 ]]; then
  echo "Converting Moment-10M part 0..."
  python scripts/convert_data.py \
    --source_path "$RAW_DIR/Moment-10M_0.json" \
    --target_path "$CONVERTED_DIR/Moment-10M_0_converted.json"

  echo "Converting Moment-10M part 1..."
  python scripts/convert_data.py \
    --source_path "$RAW_DIR/Moment-10M_1.json" \
    --target_path "$CONVERTED_DIR/Moment-10M_1_converted.json"

  echo "Converting GESM data..."
  python scripts/convert_data_gesm.py \
    --source_path "$RAW_DIR/GESM_data.json" \
    --target_path "$CONVERTED_DIR/GESM_data_converted.json"
fi

if [[ "$SKIP_VIDEOS" -eq 0 ]]; then
  echo "Downloading videos for Moment-10M part 0..."
  python scripts/download_videos.py \
    --source_path "$RAW_DIR/Moment-10M_0.json" \
    --video_path "$VIDEO_DIR" \
    --jobs "$VIDEO_JOBS" \
    --hf-repo "$HF_REPO" \
    --repo-type "$REPO_TYPE" \
    --upload-threshold-gb "$UPLOAD_THRESHOLD_GB"

  echo "Downloading videos for Moment-10M part 1..."
  python scripts/download_videos.py \
    --source_path "$RAW_DIR/Moment-10M_1.json" \
    --video_path "$VIDEO_DIR" \
    --jobs "$VIDEO_JOBS" \
    --hf-repo "$HF_REPO" \
    --repo-type "$REPO_TYPE" \
    --upload-threshold-gb "$UPLOAD_THRESHOLD_GB"
fi

echo "Done."
