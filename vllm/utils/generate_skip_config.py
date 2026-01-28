# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate skip heads configuration from collected statistics.

This script aggregates skip softmax statistics files and generates a
configuration file for runtime head skipping.

Usage:
    python -m vllm.utils.generate_skip_config \
        --stats-dir /path/to/stats/ \
        --output /path/to/skip_heads_config.json \
        --threshold 2.0 \
        --min-sparsity 98.0
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def load_stats_files(stats_dir: str) -> list[dict]:
    """Load all statistics JSON files from a directory.

    Args:
        stats_dir: Directory containing skip_softmax_stats_*.json files.

    Returns:
        List of loaded statistics dictionaries.
    """
    stats_files = []
    stats_path = Path(stats_dir)

    if not stats_path.exists():
        raise FileNotFoundError(f"Stats directory not found: {stats_dir}")

    # Find all JSON files matching various patterns
    patterns = [
        "skip_softmax_stats_*.json",  # Timestamped stats files
        "skip_softmax_*.json",         # Alternative naming pattern
    ]

    loaded_files = set()
    for pattern in patterns:
        for file_path in stats_path.glob(pattern):
            if file_path.name in loaded_files:
                continue
            try:
                with open(file_path, "r") as f:
                    stats = json.load(f)
                    # Verify it has the expected structure
                    if "per_layer_head" in stats:
                        stats_files.append(stats)
                        loaded_files.add(file_path.name)
                        print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

    # Also check for single file (non-timestamped)
    single_file = stats_path / "skip_softmax_stats.json"
    if single_file.exists() and single_file.name not in loaded_files:
        try:
            with open(single_file, "r") as f:
                stats = json.load(f)
                if "per_layer_head" in stats:
                    stats_files.append(stats)
                    print(f"Loaded: {single_file.name}")
        except Exception as e:
            print(f"Warning: Failed to load {single_file}: {e}")

    return stats_files


def aggregate_stats(
    stats_files: list[dict],
    threshold: float,
) -> dict[tuple[int, int], dict]:
    """Aggregate statistics across multiple files.

    Args:
        stats_files: List of statistics dictionaries.
        threshold: Threshold value to use for sparsity calculation.

    Returns:
        Dictionary mapping (layer, head) to aggregated stats.
    """
    # Aggregated data: (layer, head) -> {"total": int, "skippable": int}
    aggregated: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"total": 0, "skippable": 0}
    )

    threshold_key = str(threshold)

    for stats in stats_files:
        per_layer_head = stats.get("per_layer_head", {})

        for key, data in per_layer_head.items():
            layer = data.get("layer")
            head = data.get("head")

            if layer is None or head is None:
                continue

            total_blocks = data.get("total_blocks", 0)
            at_threshold = data.get("at_threshold", {})

            # Get skippable count at the specified threshold
            threshold_data = at_threshold.get(threshold_key, {})
            skippable = threshold_data.get("skippable", 0)

            aggregated[(layer, head)]["total"] += total_blocks
            aggregated[(layer, head)]["skippable"] += skippable

    return dict(aggregated)


def compute_sparsity(aggregated: dict[tuple[int, int], dict]) -> list[dict]:
    """Compute sparsity percentage for each (layer, head).

    Args:
        aggregated: Aggregated stats from aggregate_stats().

    Returns:
        List of dicts with layer, head, total, skippable, sparsity_pct.
    """
    results = []

    for (layer, head), data in aggregated.items():
        total = data["total"]
        skippable = data["skippable"]

        if total > 0:
            sparsity_pct = 100.0 * skippable / total
        else:
            sparsity_pct = 0.0

        results.append({
            "layer": layer,
            "head": head,
            "total_blocks": total,
            "skippable_blocks": skippable,
            "sparsity_pct": round(sparsity_pct, 2),
        })

    # Sort by sparsity (descending), then by layer, then by head
    results.sort(key=lambda x: (-x["sparsity_pct"], x["layer"], x["head"]))

    return results


def generate_skip_config(
    stats_dir: str,
    output_path: str,
    threshold: float = 2.0,
    min_sparsity: float = 98.0,
) -> dict:
    """Generate skip heads configuration from statistics files.

    Args:
        stats_dir: Directory containing statistics JSON files.
        output_path: Path to write output configuration.
        threshold: Threshold value to use for sparsity calculation.
        min_sparsity: Minimum sparsity percentage to include head.

    Returns:
        Generated configuration dictionary.
    """
    # Load all stats files
    stats_files = load_stats_files(stats_dir)

    if not stats_files:
        raise ValueError(f"No stats files found in {stats_dir}")

    print(f"\nLoaded {len(stats_files)} statistics file(s)")

    # Aggregate stats
    aggregated = aggregate_stats(stats_files, threshold)
    print(f"Found {len(aggregated)} unique (layer, head) combinations")

    # Compute sparsity
    all_heads = compute_sparsity(aggregated)

    # Filter heads that meet the minimum sparsity threshold
    skip_heads = [h for h in all_heads if h["sparsity_pct"] >= min_sparsity]

    print(f"\nHeads with sparsity >= {min_sparsity}%: {len(skip_heads)}")

    # Build configuration
    config = {
        "threshold_used": threshold,
        "min_sparsity": min_sparsity,
        "total_heads_analyzed": len(all_heads),
        "heads_to_skip": len(skip_heads),
        "skip_heads": [
            {
                "layer": h["layer"],
                "head": h["head"],
                f"sparsity_t{threshold}": h["sparsity_pct"],
            }
            for h in skip_heads
        ],
        # Also provide compact format for easy loading
        "skip_mask": {},
    }

    # Build compact skip_mask format: layer -> list of heads
    for h in skip_heads:
        layer_str = str(h["layer"])
        if layer_str not in config["skip_mask"]:
            config["skip_mask"][layer_str] = []
        config["skip_mask"][layer_str].append(h["head"])

    # Sort head lists
    for layer_str in config["skip_mask"]:
        config["skip_mask"][layer_str].sort()

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nWrote configuration to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Threshold: {threshold}")
    print(f"Min sparsity: {min_sparsity}%")
    print(f"Total heads analyzed: {len(all_heads)}")
    print(f"Heads to skip: {len(skip_heads)} ({100.0 * len(skip_heads) / max(len(all_heads), 1):.1f}%)")

    # Print per-layer summary
    layers_with_skips = sorted(set(h["layer"] for h in skip_heads))
    print(f"\nLayers with skipped heads: {layers_with_skips}")

    for layer in layers_with_skips:
        layer_heads = [h for h in skip_heads if h["layer"] == layer]
        heads_str = ", ".join(str(h["head"]) for h in layer_heads[:10])
        if len(layer_heads) > 10:
            heads_str += f"... ({len(layer_heads)} total)"
        print(f"  Layer {layer}: {len(layer_heads)} heads - [{heads_str}]")

    # Print top 10 most sparse heads
    print("\nTop 10 most sparse heads:")
    for h in skip_heads[:10]:
        print(f"  L{h['layer']}H{h['head']}: {h['sparsity_pct']:.1f}%")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate skip heads configuration from statistics files"
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        required=True,
        help="Directory containing skip_softmax_stats_*.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for skip_heads_config.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Threshold value to use for sparsity (default: 2.0)",
    )
    parser.add_argument(
        "--min-sparsity",
        type=float,
        default=98.0,
        help="Minimum sparsity percentage to include head (default: 98.0)",
    )

    args = parser.parse_args()

    generate_skip_config(
        stats_dir=args.stats_dir,
        output_path=args.output,
        threshold=args.threshold,
        min_sparsity=args.min_sparsity,
    )


if __name__ == "__main__":
    main()
