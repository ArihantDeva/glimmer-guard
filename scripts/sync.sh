#!/usr/bin/env bash
# Push code + data to Amarel. Everything lands on /scratch: the home
# directory has a hard ~2GB quota and will silently evict large files.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${JEV_HOST:-amarel}"
SCRATCH="${JEV_SCRATCH:-/scratch/ad2416}"

echo "== syncing code -> $HOST:$SCRATCH/jevsrc =="
ssh "$HOST" "mkdir -p $SCRATCH/jevsrc $SCRATCH/jevdata"
rsync -az src/ "$HOST:$SCRATCH/jevsrc/"

echo "== syncing data -> $HOST:$SCRATCH/jevdata =="
rsync -az data/swe_*.jsonl data/noul_*.jsonl data/rjudge.jsonl "$HOST:$SCRATCH/jevdata/"

echo "== remote state =="
ssh "$HOST" "wc -l $SCRATCH/jevdata/*.jsonl"
