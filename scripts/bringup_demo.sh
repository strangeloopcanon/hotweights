#!/usr/bin/env bash
set -euo pipefail

# Bring-up: coordinator, sample manifests, plan, and replicate via UCX or local.

ENDPOINT=${ENDPOINT:-tcp://127.0.0.1:5555}

echo "Starting coordinator at $ENDPOINT" >&2
(hotweights coord-serve --endpoint "$ENDPOINT" &) >/dev/null 2>&1 || true
sleep 1

echo "Building sample checkpoints" >&2
rm -rf demo_ckpt_a demo_ckpt_b plan.json || true
mkdir -p demo_ckpt_a demo_ckpt_b
echo "hello" > demo_ckpt_a/a.bin
cp demo_ckpt_a/a.bin demo_ckpt_b/a.bin
echo "world" > demo_ckpt_b/b.bin

echo "Publishing manifests" >&2
hotweights publish --checkpoint demo_ckpt_a --version v0 --output m_prev.json
hotweights publish --checkpoint demo_ckpt_b --version v1 --output m_next.json

echo "Planning transfer" >&2
hotweights plan --prev m_prev.json --next m_next.json --bucket-mb 1 --output plan.json

echo "Submitting plan to coordinator" >&2
hotweights coord-submit-plan --endpoint "$ENDPOINT" --plan plan.json
hotweights begin --endpoint "$ENDPOINT" --version v1

echo "Replicating locally (no UCX/MPI)" >&2
hotweights replicate --plan plan.json --verify

echo "Done. Expose metrics at :9097 (replicate), :9100 (coord)."

