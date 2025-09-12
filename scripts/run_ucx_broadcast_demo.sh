#!/usr/bin/env bash
set -euo pipefail

# UCX broadcast demo: replicates a plan with 2 ranks over UCX.

PLAN=${1:-plan.json}
WORLD_SIZE=${WORLD_SIZE:-2}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-19999}

if [[ ! -f "$PLAN" ]]; then
  echo "Plan $PLAN not found. Run planning first." >&2
  exit 1
fi

export WORLD_SIZE MASTER_ADDR MASTER_PORT

echo "Starting rank 0 (listener)" >&2
RANK=0 hotweights replicate --plan "$PLAN" --ucx --verify &
PID0=$!
sleep 1
echo "Starting rank 1 (receiver)" >&2
RANK=1 hotweights replicate --plan "$PLAN" --ucx --verify

wait $PID0 || true
echo "UCX broadcast demo done."

