Tensor-Parallel Group Mapping (HOTWEIGHTS_TP_GROUPS)

Overview

- HOTWEIGHTS_TP_GROUPS provides a mapping from a tensor-parallel group id to a list of world ranks that should consume buckets containing tensors belonging to that group.
- When group mapping is present, plan creation derives per-bucket `consumer_ranks` by inspecting each item’s `tensor` partitioning (`partitioning.tp_group`, `group`, or `group_id`).
- If no mapping is provided, the fallback auto mode uses HOTWEIGHTS_AUTO_TP=1 and HOTWEIGHTS_TP (or the maximum `partitioning.tp` found in the manifest) to assign a simple contiguous consumer set `[0..tp-1]`.

Schema

- HOTWEIGHTS_TP_GROUPS can be either:
  - a JSON string passed directly via environment variable
  - a path to a JSON file containing the mapping

Example (JSON string):

  HOTWEIGHTS_TP_GROUPS='{"0":[0,1,2,3], "1":[4,5,6,7]}'

Example (JSON file):

  $ cat groups.json
  {
    "0": [0, 1, 2, 3],
    "1": [4, 5, 6, 7]
  }
  
  $ HOTWEIGHTS_TP_GROUPS=groups.json hotweights plan --prev prev.json --next next.json --bucket-mb 512

Validation Rules

- Keys must be strings convertible to group ids (e.g., "0", "1").
- Values must be arrays of integers representing world ranks.
- Invalid mappings are ignored; plan generation falls back to auto mode when enabled.

Notes

- If `partitioning.tp_group` (or `group`/`group_id`) is missing on tensors, the mapping cannot be applied for those items.
- `consumer_ranks` affects broadcast and scatter; buckets are only transmitted to the listed ranks in MPI/UCX/CUDA-IPC paths.

