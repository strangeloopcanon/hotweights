"""
SOTA Tensor Mapping Wizard.

This module provides an interactive CLI tool to solve the tensor name mapping
problem, which is a major source of deployment errors.

Features:
- Interactive CLI to guide users through the mapping process.
- Multi-heuristic similarity analysis (Levenshtein distance, shape/dtype matching).
- Automatic suggestion of high-confidence mappings.
- User-friendly interface to confirm, reject, or manually map tensors.
- Generates a production-ready JSON mapping file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

# A simple text distance function for name similarity
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def analyze_similarity(
    manifest_tensor: Dict[str, Any],
    model_params: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Analyzes similarity and returns a sorted list of potential matches."""
    scores = []
    m_name = manifest_tensor["name"].lower().replace("/", ".").rsplit(".", 1)[0]
    m_shape = tuple(manifest_tensor.get("shape", []))
    m_dtype = manifest_tensor.get("dtype")

    for p_name, p_attrs in model_params.items():
        score = 0
        # Name similarity score (higher is better)
        dist = levenshtein_distance(m_name, p_name.lower())
        score += (1 - dist / max(len(m_name), len(p_name))) * 100

        # Shape and dtype match boosts score significantly
        if tuple(p_attrs.get("shape", [])) == m_shape:
            score += 50
        if p_attrs.get("dtype") == m_dtype:
            score += 20
        
        if score > 50: # Confidence threshold
            scores.append({"param_name": p_name, "score": score})
    
    return sorted(scores, key=lambda x: x["score"], reverse=True)

def run_wizard(manifest_path: str, model_path: str, output_path: str):
    """
    Runs the interactive mapping wizard.
    In a real scenario, `model_path` would be used to load the model and introspect it.
    Here, we will use a dummy model structure.
    """
    print("--- Hotweights Tensor Mapping Wizard ---")
    
    # 1. Load manifest tensors
    with Path(manifest_path).open("r") as f:
        manifest = json.load(f)
    manifest_tensors = manifest.get("tensors", [])
    print(f"Loaded {len(manifest_tensors)} tensors from manifest.")

    # 2. Load model parameters (dummy data for this example)
    # A real implementation would load a PyTorch model and call .named_parameters()
    dummy_model_params = {
        "model.layers.0.self_attn.q_proj.weight": {"shape": [4096, 4096], "dtype": "bfloat16"},
        "model.layers.0.self_attn.k_proj.weight": {"shape": [4096, 1024], "dtype": "bfloat16"},
        "model.layers.1.mlp.gate_proj.weight": {"shape": [14336, 4096], "dtype": "bfloat16"},
        "lm_head.weight": {"shape": [32000, 4096], "dtype": "bfloat16"},
    }
    print(f"Loaded {len(dummy_model_params)} parameters from model (dummy).")

    final_mapping = {}
    unmapped_tensors = []

    # 3. Iterate and map
    for tensor in manifest_tensors:
        tensor_key = f"{tensor['name']}:{tensor['shards'][0]['rank']}"
        print(f"\nProcessing tensor: {tensor_key} (Shape: {tensor.get('shape')})" )
        
        suggestions = analyze_similarity(tensor, dummy_model_params)
        
        if not suggestions or suggestions[0]["score"] < 80:
            print("--> No high-confidence match found.")
            unmapped_tensors.append(tensor)
            continue

        top_suggestion = suggestions[0]["param_name"]
        try:
            choice = input(f"--> Best match: '{top_suggestion}' (Score: {suggestions[0]['score']:.0f}). Use this? [Y/n/m(anual)]: ").lower()
        except EOFError:
            choice = "y"

        if choice == 'y' or choice == '':
            final_mapping[tensor_key] = top_suggestion
            print(f"    Mapping {tensor_key} -> {top_suggestion}")
        elif choice == 'm':
            manual_param = input("    Enter the exact parameter name: ")
            final_mapping[tensor_key] = manual_param
        else:
            unmapped_tensors.append(tensor)
            print("    Skipped.")

    # 4. Handle unmapped tensors
    if unmapped_tensors:
        print(f"\n--- Reviewing {len(unmapped_tensors)} Unmapped Tensors ---")
        for tensor in unmapped_tensors:
            tensor_key = f"{tensor['name']}:{tensor['shards'][0]['rank']}"
            manual_param = input(f"Enter mapping for {tensor_key} (or press Enter to skip): ")
            if manual_param:
                final_mapping[tensor_key] = manual_param

    # 5. Save the final mapping
    with Path(output_path).open("w") as f:
        json.dump(final_mapping, f, indent=2)
    
    print(f"\nMapping complete. Saved {len(final_mapping)} mappings to {output_path}.")
