#!/usr/bin/env bash
# Download dlib's 68-point facial landmark model into models/ (not stored in git).
# Equivalent to `python main.py --download-model`, but needs only curl + bunzip2.
# See DATA.md for the model's source and licence.
set -euo pipefail

URL="https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${FATIGUE_MODELS_DIR:-$ROOT/models}"
OUT="$MODELS_DIR/shape_predictor_68_face_landmarks.dat"

if [[ -f "$OUT" ]]; then echo "Already present: $OUT"; exit 0; fi
mkdir -p "$MODELS_DIR"
trap 'rm -f "$OUT.bz2.part" "$OUT.part"' EXIT

echo "Downloading $URL (~64 MB) ..."
curl -fL --retry 3 --progress-bar -o "$OUT.bz2.part" "$URL"
echo "Decompressing ..."
bunzip2 -c "$OUT.bz2.part" > "$OUT.part"
mv "$OUT.part" "$OUT"
echo "Ready: $OUT"
