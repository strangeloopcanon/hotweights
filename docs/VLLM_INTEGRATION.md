vLLM Integration Guide (Smoke-Level)

Overview

This guide shows two practical ways to integrate Hotweights with vLLM:

1) WorkerExtension path (CPU-friendly): call into Hotweights from your worker
   to fetch a plan from the coordinator and apply weights (CPU→GPU or CPU-only).
2) Auto-binder path (background thread): patch vLLM engines and let a binder
   manage pause/apply/commit on new versions (requires a coordinator and works
   best with CUDA-IPC).

Prerequisites

- Python 3.10+
- Torch installed (CUDA optional for the smoke)
- Optional: pyzmq installed for coordinator RPCs: pip install .[extras]
- Optional: vLLM installed with the version matching your environment

WorkerExtension Smoke (no CUDA required)

This mode uses a tiny nn.Linear as the “model” and a toy plan. It works with
or without a running coordinator.

Coordinator mode:

1) Start the HA coordinator in another shell:

   hotweights coord-serve --endpoint tcp://127.0.0.1:5555

2) Run the smoke demo (builds a toy plan, submits to coordinator, applies):

   python examples/vllm_smoke_demo.py --use-coord --endpoint tcp://127.0.0.1:5555

Local mode (no coordinator):

   python examples/vllm_smoke_demo.py

The script will print Applied weights match expected: True when successful.

Inside a vLLM worker, you can do the same with the WorkerExtension directly:

   from hotweights.adapters.vllm_extension import HotweightsWorkerExtension
   ext = HotweightsWorkerExtension(endpoint="tcp://127.0.0.1:5555")
   ext.bind_module(model_module)  # your torch.nn.Module
   ext.apply_update()

Auto-binder Path (vLLM engines)

This path patches vLLM engine constructors to start a background binder that
polls the coordinator and applies updates inside a pause/commit window.

   from hotweights.adapters.vllm_auto import install_autobind
   install_autobind(name_map=None, endpoint="tcp://127.0.0.1:5555")

Call this before constructing vLLM engines (or as early as possible). The
binder detects new versions via coordinator status + plan and applies them.

Notes

- Binder’s GPU-native path is enabled when HOTWEIGHTS_USE_IPC_AGENT=1 and a
  hotweights_agent is attached to your engine; otherwise it assumes a worker
  performs the apply. For CPU-only smoke, use the WorkerExtension approach.
- For CUDA-IPC intra-node + UCX/MPI inter-node setups, see docs/WALKTHROUGHS.md
  and docs/PRESETS.md to configure transport and topology knobs.

Troubleshooting

- Ensure pyzmq is installed for coordinator RPCs (pip install .[extras]).
- If running on macOS with MPS, the demo uses CPU-only safe paths.
- Use hotweights coord-status to inspect coordinator state and last events.

