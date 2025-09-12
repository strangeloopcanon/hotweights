#!/usr/bin/env bash
set -euo pipefail

# Example: publish two manifests, generate a plan, and replicate locally.

CKPT_A=${CKPT_A:-./example_ckpt_a}
CKPT_B=${CKPT_B:-./example_ckpt_b}

mkdir -p "$CKPT_A" "$CKPT_B"
echo "hello" >"$CKPT_A/a.bin"
cp "$CKPT_A/a.bin" "$CKPT_B/a.bin"
echo "world" >"$CKPT_B/b.bin"

hotweights publish --checkpoint "$CKPT_A" --version v0 --model-id toy --output m_prev.json
hotweights publish --checkpoint "$CKPT_B" --version v1 --model-id toy --output m_next.json

hotweights plan --prev m_prev.json --next m_next.json --bucket-mb 1 --output plan.json
hotweights replicate --plan plan.json --verify --commit

echo "Done."
