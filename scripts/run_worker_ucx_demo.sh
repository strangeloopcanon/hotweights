#!/usr/bin/env bash
set -euo pipefail

# Starts coordinator, builds a toy plan, and launches two workers with UCX.

ENDPOINT=${ENDPOINT:-tcp://127.0.0.1:5555}
WORLD_SIZE=${WORLD_SIZE:-2}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-20000}

export WORLD_SIZE MASTER_ADDR MASTER_PORT

echo "Starting coordinator at $ENDPOINT" >&2
(hotweights coord-serve --endpoint "$ENDPOINT" &) >/dev/null 2>&1 || true
sleep 1

echo "Preparing toy checkpoints and plan" >&2
rm -rf demo_ucx_ckpt_a demo_ucx_ckpt_b ucx_plan.json || true
mkdir -p demo_ucx_ckpt_a demo_ucx_ckpt_b
dd if=/dev/zero of=demo_ucx_ckpt_a/a.bin bs=1m count=8 >/dev/null 2>&1
cp demo_ucx_ckpt_a/a.bin demo_ucx_ckpt_b/a.bin
dd if=/dev/zero of=demo_ucx_ckpt_b/b.bin bs=1m count=4 >/dev/null 2>&1

hotweights publish --checkpoint demo_ucx_ckpt_a --version v0 --output m_prev.json
hotweights publish --checkpoint demo_ucx_ckpt_b --version v1 --output m_next.json
hotweights plan --prev m_prev.json --next m_next.json --bucket-mb 2 --output ucx_plan.json
hotweights coord-submit-plan --endpoint "$ENDPOINT" --plan ucx_plan.json
hotweights begin --endpoint "$ENDPOINT" --version v1

echo "Launching workers with UCX" >&2
(HOTWEIGHTS_COORD=$ENDPOINT hotweights worker --ucx --pinned --no-mpi --no-verify &) >/dev/null 2>&1 || true
(HOTWEIGHTS_COORD=$ENDPOINT hotweights worker --ucx --pinned --no-mpi --no-verify &) >/dev/null 2>&1 || true

echo "Workers started. Coordinator metrics at :9100."

