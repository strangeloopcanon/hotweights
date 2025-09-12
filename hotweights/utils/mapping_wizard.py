"""Interactive tensor name mapping wizard for automated model compatibility.

Provides:
1. Interactive CLI tool for tensor name mapping
2. Automatic similarity detection using multiple heuristics
3. Rule-based mapping configuration
4. Confidence scoring and validation
5. Export to reusable mapping configurations

This eliminates the most common source of errors in production deployments.
"""
from __future__ import annotations

import re
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import difflib
import hashlib

try:  # optional
    import torch
except Exception:
    torch = None

from ..manifest import Manifest, TensorEntry
from ..telemetry.metrics import Timer


@dataclass
class MappingCandidate:
    """A candidate tensor mapping with confidence score."""
    source_name: str
    target_name: str
    confidence: float  # 0-1
    reasoning: str
    shape_match: bool = False
    dtype_match: bool = False
    name_similarity: float = 0.0
    structural_similarity: float = 0.0


@dataclass
class MappingRule:
    """A rule for tensor name mapping."""
    pattern: str  # Regex pattern
    replacement: str
    description: str
    priority: int = 0  # Higher priority rules applied first
    enabled: bool = True


@dataclass
class MappingConfiguration:
    """Complete mapping configuration."""
    name: str
    description: str
    version: str
    source_model_id: str
    target_model_id: str
    direct_mappings: Dict[str, str] = field(default_factory=dict)
    rules: List[MappingRule] = field(default_factory=list)
    confidence_threshold: float = 0.8
    auto_generated: bool = True
    created_at: str = field(default_factory=lambda: str(time.time()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MappingConfiguration':
        """Create from dictionary."""
        # Convert rules back to MappingRule objects
        if 'rules' in data:
            data['rules'] = [MappingRule(**rule) for rule in data['rules']]
        return cls(**data)


class TensorSimilarityAnalyzer:
    """Analyzes tensor similarity using multiple heuristics."""
    
    def __init__(self):
        self.heuristics = [
            self._name_similarity,
            self._shape_similarity,
            self._dtype_similarity,
            self._structural_similarity,
            self._positional_similarity
        ]
    
    def analyze_similarity(self, source_tensor: TensorEntry, 
                          target_tensor: TensorEntry,
                          source_context: Dict[str, Any],
                          target_context: Dict[str, Any]) -> MappingCandidate:
        """Analyze similarity between two tensors."""
        
        scores = []
        reasoning_parts = []
        
        for heuristic in self.heuristics:
            score, reasoning = heuristic(source_tensor, target_tensor, source_context, target_context)
            scores.append(score)
            if reasoning:
                reasoning_parts.append(reasoning)
        
        # Weighted combination of scores
        weights = [0.3, 0.25, 0.15, 0.2, 0.1]  # Name similarity is most important
        confidence = sum(score * weight for score, weight in zip(scores, weights))
        
        return MappingCandidate(
            source_name=source_tensor['name'],
            target_name=target_tensor['name'],
            confidence=confidence,
            reasoning="; ".join(reasoning_parts),
            shape_match=scores[1] > 0.9,
            dtype_match=scores[2] > 0.9,
            name_similarity=scores[0],
            structural_similarity=scores[3]
        )
    
    def _name_similarity(self, source: TensorEntry, target: TensorEntry,
                        source_ctx: Dict, target_ctx: Dict) -> Tuple[float, str]:
        """Calculate name-based similarity."""
        src_name = source['name']
        tgt_name = target['name']
        
        # Exact match
        if src_name == tgt_name:
            return 1.0, "exact name match"
        
        # Normalize names by removing common prefixes/suffixes
        src_normalized = self._normalize_name(src_name)
        tgt_normalized = self._normalize_name(tgt_name)
        
        if src_normalized == tgt_normalized:
            return 0.95, "normalized name match"
        
        # Sequence similarity
        similarity = difflib.SequenceMatcher(None, src_normalized, tgt_normalized).ratio()
        
        # Check for common patterns
        if self._check_pattern_match(src_name, tgt_name):
            similarity = max(similarity, 0.8)
        
        reasoning = f"name similarity: {similarity:.2f}" if similarity > 0.5 else ""
        return similarity, reasoning
    
    def _normalize_name(self, name: str) -> str:
        """Normalize tensor name for comparison."""
        # Remove common prefixes and suffixes
        patterns_to_remove = [
            r'^model\.',
            r'^transformer\.',
            r'^language_model\.',
            r'\.weight$',
            r'\.bias$',
            r'^_',
            r'_$'
        ]
        
        normalized = name
        for pattern in patterns_to_remove:
            normalized = re.sub(pattern, '', normalized)
        
        return normalized
    
    def _check_pattern_match(self, src_name: str, tgt_name: str) -> bool:
        """Check for common pattern matches."""
        # Common layer name mappings
        patterns = [
            (r'\.q_proj\.', r'.query_proj.'),
            (r'\.k_proj\.', r'.key_proj.'),
            (r'\.v_proj\.', r'.value_proj.'),
            (r'\.o_proj\.', r'.out_proj.'),
            (r'\.w_q\.', r'.q_proj.'),
            (r'\.w_k\.', r'.k_proj.'),
            (r'\.w_v\.', r'.v_proj.'),
            (r'\.w_o\.', r'.o_proj.'),
        ]
        
        for src_pattern, tgt_pattern in patterns:
            if re.search(src_pattern, src_name) and re.search(tgt_pattern, tgt_name):
                return True
            if re.search(tgt_pattern, src_name) and re.search(src_pattern, tgt_name):
                return True
        
        return False
    
    def _shape_similarity(self, source: TensorEntry, target: TensorEntry,
                         source_ctx: Dict, target_ctx: Dict) -> Tuple[float, str]:
        """Calculate shape-based similarity."""
        src_shape = source.get('shape', [])
        tgt_shape = target.get('shape', [])
        
        if not src_shape or not tgt_shape:
            return 0.0, "no shape info"
        
        if src_shape == tgt_shape:
            return 1.0, "exact shape match"
        
        # Check if shapes are compatible (same rank, similar dimensions)
        if len(src_shape) != len(tgt_shape):
            return 0.0, f"different ranks: {len(src_shape)} vs {len(tgt_shape)}"
        
        # Calculate dimension similarity
        dim_similarities = []
        for src_dim, tgt_dim in zip(src_shape, tgt_shape):
            if src_dim == tgt_dim:
                dim_similarities.append(1.0)
            elif src_dim > 0 and tgt_dim > 0:
                # Similarity based on relative difference
                similarity = 1.0 - abs(src_dim - tgt_dim) / max(src_dim, tgt_dim)
                dim_similarities.append(max(0.0, similarity))
            else:
                dim_similarities.append(0.0)
        
        avg_dim_similarity = sum(dim_similarities) / len(dim_similarities)
        reasoning = f"shape similarity: {avg_dim_similarity:.2f}"
        return avg_dim_similarity, reasoning
    
    def _dtype_similarity(self, source: TensorEntry, target: TensorEntry,
                         source_ctx: Dict, target_ctx: Dict) -> Tuple[float, str]:
        """Calculate dtype-based similarity."""
        src_dtype = source.get('dtype')
        tgt_dtype = target.get('dtype')
        
        if not src_dtype or not tgt_dtype:
            return 0.5, "no dtype info"
        
        if src_dtype == tgt_dtype:
            return 1.0, "exact dtype match"
        
        # Check if dtypes are compatible
        compatible_pairs = [
            ('float32', 'bfloat16'),
            ('float16', 'bfloat16'),
            ('int32', 'int64'),
            ('int16', 'int32'),
        ]
        
        if (src_dtype, tgt_dtype) in compatible_pairs or (tgt_dtype, src_dtype) in compatible_pairs:
            return 0.8, "compatible dtypes"
        
        return 0.0, f"incompatible dtypes: {src_dtype} vs {tgt_dtype}"
    
    def _structural_similarity(self, source: TensorEntry, target: TensorEntry,
                              source_ctx: Dict, target_ctx: Dict) -> Tuple[float, str]:
        """Calculate structural similarity based on layer hierarchy."""
        src_name = source['name']
        tgt_name = target['name']
        
        # Extract layer hierarchy
        src_parts = src_name.split('.')
        tgt_parts = tgt_name.split('.')
        
        # Check if they have similar structure
        if len(src_parts) == len(tgt_parts):
            # Same depth - check for structural similarity
            matches = 0
            for src_part, tgt_part in zip(src_parts[:-1], tgt_parts[:-1]):  # Exclude final tensor name
                if src_part == tgt_part:
                    matches += 1
            
            structural_similarity = matches / max(len(src_parts) - 1, 1)
            return structural_similarity, f"structural similarity: {structural_similarity:.2f}"
        
        return 0.0, "different structures"
    
    def _positional_similarity(self, source: TensorEntry, target: TensorEntry,
                              source_ctx: Dict, target_ctx: Dict) -> Tuple[float, str]:
        """Calculate positional similarity based on layer position."""
        src_name = source['name']
        tgt_name = target['name']
        
        # Get layer positions in model
        src_pos = source_ctx.get('layer_position', {}).get(src_name, -1)
        tgt_pos = target_ctx.get('layer_position', {}).get(tgt_name, -1)
        
        if src_pos >= 0 and tgt_pos >= 0:
            # Calculate position similarity
            max_pos = max(src_pos, tgt_pos)
            if max_pos > 0:
                position_similarity = 1.0 - abs(src_pos - tgt_pos) / max_pos
                return position_similarity, f"position similarity: {position_similarity:.2f}"
        
        return 0.5, "no position info"


class MappingWizard:
    """Interactive tensor name mapping wizard."""
    
    def __init__(self):
        self.similarity_analyzer = TensorSimilarityAnalyzer()
        self.mapping_rules: List[MappingRule] = []
        self.configurations: Dict[str, MappingConfiguration] = {}
    
    def analyze_manifests(self, source_manifest: Manifest, 
                         target_manifest: Manifest) -> List[MappingCandidate]:
        """Analyze two manifests and find mapping candidates."""
        
        # Build context for analysis
        source_context = self._build_context(source_manifest)
        target_context = self._build_context(target_manifest)
        
        # Get tensor lists
        source_tensors = source_manifest['tensors']
        target_tensors = target_manifest['tensors']
        
        candidates = []
        
        # For each source tensor, find best matches in target
        for source_tensor in source_tensors:
            best_candidate = None
            best_score = 0.0
            
            for target_tensor in target_tensors:
                candidate = self.similarity_analyzer.analyze_similarity(
                    source_tensor, target_tensor, source_context, target_context
                )
                
                if candidate.confidence > best_score:
                    best_score = candidate.confidence
                    best_candidate = candidate
            
            if best_candidate and best_candidate.confidence > 0.3:  # Minimum threshold
                candidates.append(best_candidate)
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return candidates
    
    def _build_context(self, manifest: Manifest) -> Dict[str, Any]:
        """Build analysis context from manifest."""
        context = {
            'tensor_positions': {},
            'layer_hierarchy': defaultdict(list),
            'shape_distribution': defaultdict(int),
            'dtype_distribution': defaultdict(int)
        }
        
        for i, tensor in enumerate(manifest['tensors']):
            tensor_name = tensor['name']
            
            # Track position
            context['tensor_positions'][tensor_name] = i
            
            # Build layer hierarchy
            parts = tensor_name.split('.')
            for j, part in enumerate(parts[:-1]):
                context['layer_hierarchy'][part].append(tensor_name)
            
            # Track shape and dtype distributions
            shape = tuple(tensor.get('shape', []))
            dtype = tensor.get('dtype', 'unknown')
            context['shape_distribution'][shape] += 1
            context['dtype_distribution'][dtype] += 1
        
        return context
    
    def interactive_mapping_session(self, source_manifest: Manifest,
                                   target_manifest: Manifest,
                                   confidence_threshold: float = 0.8) -> MappingConfiguration:
        """Run an interactive mapping session."""
        
        print("🔍 Tensor Name Mapping Wizard")
        print("=" * 50)
        print(f"Source: {source_manifest['model_id']} ({len(source_manifest['tensors'])} tensors)")
        print(f"Target: {target_manifest['model_id']} ({len(target_manifest['tensors'])} tensors)")
        print()
        
        # Analyze manifests
        candidates = self.analyze_manifests(source_manifest, target_manifest)
        
        # Categorize by confidence
        high_confidence = [c for c in candidates if c.confidence >= confidence_threshold]
        medium_confidence = [c for c in candidates if 0.5 <= c.confidence < confidence_threshold]
        low_confidence = [c for c in candidates if c.confidence < 0.5]
        
        print(f"Found {len(high_confidence)} high-confidence mappings")
        print(f"Found {len(medium_confidence)} medium-confidence mappings")
        print(f"Found {len(low_confidence)} low-confidence mappings")
        print()
        
        # Interactive review process
        approved_mappings = {}
        
        # Auto-approve high confidence mappings
        print("🟢 High-confidence mappings (auto-approved):")
        for candidate in high_confidence[:10]:  # Show first 10
            print(f"  {candidate.source_name} -> {candidate.target_name} (confidence: {candidate.confidence:.2f})")
            approved_mappings[candidate.source_name] = candidate.target_name
        
        if len(high_confidence) > 10:
            print(f"  ... and {len(high_confidence) - 10} more")
        
        print()
        
        # Review medium confidence mappings
        if medium_confidence:
            print("🟡 Medium-confidence mappings (need review):")
            for candidate in medium_confidence[:5]:  # Show first 5
                print(f"  {candidate.source_name} -> {candidate.target_name} (confidence: {candidate.confidence:.2f})")
                print(f"    Reasoning: {candidate.reasoning}")
                
                response = input("Approve this mapping? (y/n/s=skip): ").lower().strip()
                if response == 'y':
                    approved_mappings[candidate.source_name] = candidate.target_name
                elif response == 's':
                    break
                print()
        
        # Handle remaining tensors
        remaining_source = set(t['name'] for t in source_manifest['tensors']) - set(approved_mappings.keys())
        remaining_target = set(t['name'] for t in target_manifest['tensors']) - set(approved_mappings.values())
        
        if remaining_source or remaining_target:
            print("🔴 Manual mapping required for remaining tensors:")
            print(f"  {len(remaining_source)} source tensors need mapping")
            print(f"  {len(remaining_target)} target tensors available")
            
            # Show remaining source tensors
            if remaining_source:
                print("\nRemaining source tensors:")
                for tensor_name in sorted(remaining_source)[:10]:
                    print(f"  - {tensor_name}")
                if len(remaining_source) > 10:
                    print(f"  ... and {len(remaining_source) - 10} more")
            
            # Interactive manual mapping
            print("\nManual mapping mode:")
            for source_name in list(remaining_source)[:5]:  # Handle first 5 manually
                print(f"\nMap '{source_name}' to which target tensor?")
                
                # Show similar target tensors
                similar_targets = self._find_similar_names(source_name, list(remaining_target))
                for i, (target_name, similarity) in enumerate(similar_targets[:5]):
                    print(f"  {i+1}. {target_name} (similarity: {similarity:.2f})")
                print(f"  0. Skip this tensor")
                print(f"  c. Custom target name")
                
                response = input("Select option: ").strip().lower()
                
                if response == '0':
                    continue
                elif response == 'c':
                    custom_target = input("Enter custom target name: ").strip()
                    if custom_target:
                        approved_mappings[source_name] = custom_target
                        remaining_target.discard(custom_target)
                elif response.isdigit() and 1 <= int(response) <= len(similar_targets):
                    selected_target = similar_targets[int(response) - 1][0]
                    approved_mappings[source_name] = selected_target
                    remaining_target.discard(selected_target)
                    
        # Generate configuration
        config = MappingConfiguration(
            name=f"{source_manifest['model_id']}_to_{target_manifest['model_id']}",
            description=f"Auto-generated mapping from {source_manifest['model_id']} to {target_manifest['model_id']}",
            version="1.0.0",
            source_model_id=source_manifest['model_id'],
            target_model_id=target_manifest['model_id'],
            direct_mappings=approved_mappings,
            confidence_threshold=confidence_threshold
        )
        
        # Show summary
        print(f"\n✅ Mapping configuration created!")
        print(f"   Total mappings: {len(approved_mappings)}")
        print(f"   Coverage: {len(approved_mappings)}/{len(source_manifest['tensors'])} ({len(approved_mappings)/len(source_manifest['tensors'])*100:.1f}%)")
        
        return config
    
    def _find_similar_names(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """Find names similar to query."""
        similarities = [(candidate, difflib.SequenceMatcher(None, query, candidate).ratio()) 
                       for candidate in candidates]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def generate_mapping_rules(self, successful_mappings: Dict[str, str]) -> List[MappingRule]:
        """Generate mapping rules from successful mappings."""
        rules = []
        
        # Analyze patterns in successful mappings
        patterns = defaultdict(list)
        
        for source, target in successful_mappings.items():
            # Look for common patterns
            if '.weight' in source and '.weight' in target:
                patterns['weight_suffix'].append((source, target))
            if '.bias' in source and '.bias' in target:
                patterns['bias_suffix'].append((source, target))
            
            # Look for layer name transformations
            src_parts = source.split('.')
            tgt_parts = target.split('.')
            
            if len(src_parts) == len(tgt_parts):
                for i, (src_part, tgt_part) in enumerate(zip(src_parts, tgt_parts)):
                    if src_part != tgt_part:
                        patterns[f'part_{i}_replacement'].append((src_part, tgt_part))
        
        # Generate rules from patterns
        for pattern_name, mappings in patterns.items():
            if len(mappings) >= 3:  # Need at least 3 examples
                if pattern_name == 'weight_suffix':
                    rules.append(MappingRule(
                        pattern=r'^(.*)\.weight$',
                        replacement=r'\1.weight',
                        description="Preserve .weight suffix",
                        priority=10
                    ))
                elif pattern_name == 'bias_suffix':
                    rules.append(MappingRule(
                        pattern=r'^(.*)\.bias$',
                        replacement=r'\1.bias',
                        description="Preserve .bias suffix", 
                        priority=10
                    ))
                elif pattern_name.startswith('part_'):
                    # Find most common replacement
                    replacements = defaultdict(int)
                    for src_part, tgt_part in mappings:
                        replacements[(src_part, tgt_part)] += 1
                    
                    if replacements:
                        most_common = max(replacements.items(), key=lambda x: x[1])
                        src_part, tgt_part = most_common[0]
                        
                        if replacements[most_common[0]] >= 3:  # At least 3 occurrences
                            part_idx = int(pattern_name.split('_')[1])
                            pattern_parts = ['([^\.]+)'] * 10  # Support up to 10 parts
                            pattern_parts[part_idx] = re.escape(src_part)
                            pattern = '^' + '\\.'.join(pattern_parts) + '$'
                            
                            replacement_parts = [f'\\{i+1}' for i in range(10)]
                            replacement_parts[part_idx] = tgt_part
                            replacement = '\\.'.join(replacement_parts)
                            
                            rules.append(MappingRule(
                                pattern=pattern,
                                replacement=replacement,
                                description=f"Replace '{src_part}' with '{tgt_part}' in position {part_idx}",
                                priority=5
                            ))
        
        return rules
    
    def save_configuration(self, config: MappingConfiguration, filepath: Path) -> None:
        """Save mapping configuration to file."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(config, f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        print(f"✅ Configuration saved to {filepath}")
    
    def load_configuration(self, filepath: Path) -> MappingConfiguration:
        """Load mapping configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return MappingConfiguration.from_dict(data)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def apply_mapping(self, tensor_name: str, config: MappingConfiguration) -> Optional[str]:
        """Apply mapping configuration to a tensor name."""
        # Check direct mappings first
        if tensor_name in config.direct_mappings:
            return config.direct_mappings[tensor_name]
        
        # Apply rules in priority order
        sorted_rules = sorted(config.rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
                
            try:
                match = re.match(rule.pattern, tensor_name)
                if match:
                    # Apply replacement
                    result = match.expand(rule.replacement)
                    return result
            except re.error:
                continue
        
        # No mapping found
        return None
    
    def validate_mapping(self, source_manifest: Manifest, target_manifest: Manifest,
                        config: MappingConfiguration) -> Dict[str, any]:
        """Validate a mapping configuration."""
        source_tensors = {t['name']: t for t in source_manifest['tensors']}
        target_tensors = {t['name']: t for t in target_manifest['tensors']}
        
        results = {
            'total_source_tensors': len(source_tensors),
            'total_target_tensors': len(target_tensors),
            'successful_mappings': {},
            'failed_mappings': [],
            'shape_mismatches': [],
            'dtype_mismatches': [],
            'coverage': 0.0
        }
        
        for source_name, source_tensor in source_tensors.items():
            mapped_name = self.apply_mapping(source_name, config)
            
            if mapped_name:
                if mapped_name in target_tensors:
                    target_tensor = target_tensors[mapped_name]
                    
                    # Check shape compatibility
                    src_shape = source_tensor.get('shape')
                    tgt_shape = target_tensor.get('shape')
                    shape_ok = (src_shape == tgt_shape) or (not src_shape or not tgt_shape)
                    
                    # Check dtype compatibility
                    src_dtype = source_tensor.get('dtype')
                    tgt_dtype = target_tensor.get('dtype')
                    dtype_ok = (src_dtype == tgt_dtype) or (not src_dtype or not tgt_dtype)
                    
                    if shape_ok and dtype_ok:
                        results['successful_mappings'][source_name] = mapped_name
                    else:
                        if not shape_ok:
                            results['shape_mismatches'].append({
                                'source': source_name,
                                'target': mapped_name,
                                'source_shape': src_shape,
                                'target_shape': tgt_shape
                            })
                        if not dtype_ok:
                            results['dtype_mismatches'].append({
                                'source': source_name,
                                'target': mapped_name,
                                'source_dtype': src_dtype,
                                'target_dtype': tgt_dtype
                            })
                else:
                    results['failed_mappings'].append({
                        'source': source_name,
                        'mapped_target': mapped_name,
                        'error': 'target tensor not found'
                    })
            else:
                results['failed_mappings'].append({
                    'source': source_name,
                    'error': 'no mapping found'
                })
        
        # Calculate coverage
        results['coverage'] = len(results['successful_mappings']) / len(source_tensors)
        results['success_rate'] = len(results['successful_mappings']) / (len(results['successful_mappings']) + len(results['failed_mappings']))
        
        return results


def run_mapping_wizard_cli():
    """Run the mapping wizard as a CLI tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tensor Name Mapping Wizard")
    parser.add_argument("source_manifest", help="Source manifest file")
    parser.add_argument("target_manifest", help="Target manifest file")
    parser.add_argument("--output", "-o", help="Output configuration file", default="mapping_config.json")
    parser.add_argument("--confidence", "-c", type=float, default=0.8, help="Confidence threshold")
    parser.add_argument("--validate", action="store_true", help="Validate the generated mapping")
    
    args = parser.parse_args()
    
    # Load manifests
    from ..manifest import load_manifest
    
    source_manifest = load_manifest(args.source_manifest)
    target_manifest = load_manifest(args.target_manifest)
    
    # Run wizard
    wizard = MappingWizard()
    config = wizard.interactive_mapping_session(
        source_manifest, target_manifest, args.confidence
    )
    
    # Save configuration
    wizard.save_configuration(config, Path(args.output))
    
    # Validate if requested
    if args.validate:
        print("\n🔍 Validating mapping...")
        validation_results = wizard.validate_mapping(source_manifest, target_manifest, config)
        
        print(f"\nValidation Results:")
        print(f"  Coverage: {validation_results['coverage']:.1%}")
        print(f"  Success Rate: {validation_results['success_rate']:.1%}")
        print(f"  Successful Mappings: {len(validation_results['successful_mappings'])}")
        print(f"  Failed Mappings: {len(validation_results['failed_mappings'])}")
        
        if validation_results['shape_mismatches']:
            print(f"  Shape Mismatches: {len(validation_results['shape_mismatches'])}")
        if validation_results['dtype_mismatches']:
            print(f"  Dtype Mismatches: {len(validation_results['dtype_mismatches'])}")


if __name__ == "__main__":
    run_mapping_wizard_cli()
