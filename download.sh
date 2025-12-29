#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./download.sh [options]

Downloads videos using yt_index.json from Hugging Face.

Options:
  --hf-repo NAME         Hugging Face dataset repo (required)
  --repo-type TYPE       Hugging Face repo type (default: dataset)
  --video-dir PATH       Where videos are stored (default: data/videos)
  --video-jobs N         Parallel video download workers (default: 8)
  --upload-threshold-gb N  Upload tar when size exceeds this GB (default: 8)
  --upload-reference-gb N  Target GB size for each upload (default: 7)
  --dispatch-interval N  Seconds between dispatching downloads (default: 2)
  -h, --help             Show this help
EOF
}

DATA_DIR="${DATA_DIR:-data}"
HF_REPO="${HF_REPO:-}"
REPO_TYPE="${REPO_TYPE:-dataset}"
VIDEO_DIR="${VIDEO_DIR:-}"
VIDEO_JOBS="${VIDEO_JOBS:-8}"
UPLOAD_THRESHOLD_GB="${UPLOAD_THRESHOLD_GB:-8}"
UPLOAD_REFERENCE_GB="${UPLOAD_REFERENCE_GB:-7}"
DISPATCH_INTERVAL="${DISPATCH_INTERVAL:-2}"

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
    --upload-reference-gb)
      UPLOAD_REFERENCE_GB="$2"
      shift 2
      ;;
    --dispatch-interval)
      DISPATCH_INTERVAL="$2"
      shift 2
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

VIDEO_DIR="${VIDEO_DIR:-$DATA_DIR/videos}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

mkdir -p "$VIDEO_DIR"

if [[ -z "$HF_REPO" ]]; then
  echo "Missing required --hf-repo NAME."
  exit 1
fi

require_cmd python
echo "Downloading videos using yt_index.json..."
python scripts/download_videos.py \
  --video_path "$VIDEO_DIR" \
  --jobs "$VIDEO_JOBS" \
  --hf-repo "$HF_REPO" \
  --repo-type "$REPO_TYPE" \
  --upload-threshold-gb "$UPLOAD_THRESHOLD_GB" \
  --upload-reference-gb "$UPLOAD_REFERENCE_GB" \
  --dispatch-interval "$DISPATCH_INTERVAL"

echo "Done."
