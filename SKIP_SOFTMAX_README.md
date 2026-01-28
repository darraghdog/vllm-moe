# Skip Softmax Expert Usage Tracking

This feature enables vLLM to track and report per-(layer, head) attention sparsity statistics, useful for identifying candidates for TensorRT-LLM style Skip Softmax optimization.

## Overview

Skip Softmax is an optimization technique that skips attention computation for KV blocks where the attention scores would be negligible. For each KV block, we compute:

```
m_local = max(Q * K[block]^T)
```

If `(m_global - m_local) > threshold`, the block can be skipped as its softmax contribution would be negligible.

This implementation tracks which (layer, head) combinations have the highest sparsity (most skippable blocks), helping identify optimization targets.

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VLLM_SKIP_SOFTMAX_BLOCK_ANALYSIS` | bool | `0` | Enable skip softmax block analysis |
| `VLLM_SKIP_SOFTMAX_THRESHOLD` | float | `2.0` | Threshold for (m_global - m_local) to consider a block skippable |
| `VLLM_SKIP_SOFTMAX_SAMPLE_RATE` | float | `1.0` | Sampling rate (0.0-1.0) to reduce overhead |
| `VLLM_SKIP_SOFTMAX_OUTPUT_FILE` | str | `None` | Path to write full JSON statistics (all layer/head combos) |
| `VLLM_BLOCK_USAGE_LOG_INTERVAL` | float | `30.0` | Logging interval in seconds (increase for less frequent logging) |
| `VLLM_BLOCK_USAGE_STATS` | bool | `0` | Enable KV block usage statistics |
| `VLLM_ATTENTION_SPARSITY_STATS` | bool | `0` | Enable attention sparsity statistics |
| `VLLM_ATTENTION_SPARSITY_THRESHOLD` | float | `30.0` | Threshold for attention sparsity analysis |

## Usage

Set environment variables before starting the vLLM server:

```python
import os
os.environ["VLLM_SKIP_SOFTMAX_BLOCK_ANALYSIS"] = "1"
os.environ["VLLM_SKIP_SOFTMAX_THRESHOLD"] = "2.0"
os.environ["VLLM_SKIP_SOFTMAX_SAMPLE_RATE"] = "0.1"  # 10% sampling to reduce overhead
os.environ["VLLM_BLOCK_USAGE_LOG_INTERVAL"] = "300"  # Log every 5 minutes (less overhead)
os.environ["VLLM_SKIP_SOFTMAX_OUTPUT_FILE"] = "/tmp/skip_softmax_stats.json"  # Write ALL stats to file
```

Or via command line:

```bash
VLLM_SKIP_SOFTMAX_BLOCK_ANALYSIS=1 \
VLLM_SKIP_SOFTMAX_THRESHOLD=2.0 \
VLLM_SKIP_SOFTMAX_SAMPLE_RATE=0.1 \
VLLM_BLOCK_USAGE_LOG_INTERVAL=300 \
VLLM_SKIP_SOFTMAX_OUTPUT_FILE=/tmp/skip_softmax_stats.json \
python -m vllm.entrypoints.openai.api_server --model <model_name>
```

## Expected Output

When enabled, the logs will periodically show statistics like:

```
Skip Softmax Block Analysis: total_kv_blocks=1000, skippable_blocks=420, sparsity=42.0%
Most sparse (layer, head): {L31H40: 62.0%, L30H39: 58.0%, L29H38: 55.0%, ...}
Least sparse (layer, head): {L0H0: 38.0%, L1H1: 40.0%, L2H2: 41.0%, ...}
```

This helps identify:
- **Most sparse**: Attention heads that could benefit most from Skip Softmax optimization
- **Least sparse**: Attention heads where Skip Softmax would provide minimal benefit

## JSON Output Format

When `VLLM_SKIP_SOFTMAX_OUTPUT_FILE` is set, statistics are written with **multi-threshold tracking**:

```json
{
  "timestamp": 1706356281.123,
  "primary_threshold": 2.0,
  "thresholds": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
  "sample_rate": 0.1,
  "overall": {
    "total_blocks": 100000000,
    "skippable_blocks": 82000000,
    "sparsity_pct": 82.0,
    "total_calls": 10000,
    "sampled_calls": 1000,
    "at_threshold": {
      "1.0": {"skippable": 90000000, "sparsity_pct": 90.0},
      "2.0": {"skippable": 82000000, "sparsity_pct": 82.0},
      "3.0": {"skippable": 70000000, "sparsity_pct": 70.0}
    }
  },
  "per_layer_head": {
    "L0H40": {
      "layer": 0,
      "head": 40,
      "total_blocks": 100000,
      "readings": 500,
      "at_threshold": {
        "1.0": {"skippable": 99000, "sparsity_pct": 99.0},
        "2.0": {"skippable": 98000, "sparsity_pct": 98.0},
        "3.0": {"skippable": 95000, "sparsity_pct": 95.0}
      }
    }
  }
}
```

Key fields:
- **thresholds**: All thresholds tracked (enables post-hoc analysis)
- **readings**: Number of samples per (layer, head) for proper averaging
- **at_threshold**: Sparsity stats at each threshold level

This matches TensorRT-LLM's approach where threshold scales with sequence length.

## Files

| File | Description |
|------|-------------|
| `vllm/attention/ops/triton_skip_softmax_analysis.py` | Triton kernel (`skip_softmax_block_analysis`), `compute_skip_softmax_sparsity_multi_threshold()` |
| `vllm/v1/core/kv_cache_metrics.py` | `SkipSoftmaxBlockAnalyzer` class with multi-threshold tracking |
| `vllm/v1/attention/backends/flash_attn.py` | Integration with FlashAttention backend |
| `vllm/envs.py` | Environment variable definitions |

## Performance Considerations

To minimize impact on inference throughput:

1. **Reduce sample rate**: `VLLM_SKIP_SOFTMAX_SAMPLE_RATE=0.01` (1% sampling recommended)
2. **Increase log interval**: `VLLM_BLOCK_USAGE_LOG_INTERVAL=90` (90 seconds)
3. **Use file output**: Set `VLLM_SKIP_SOFTMAX_OUTPUT_FILE` to get ALL stats without verbose logging

Minimal overhead configuration:
```bash
VLLM_SKIP_SOFTMAX_BLOCK_ANALYSIS=1 \
VLLM_SKIP_SOFTMAX_SAMPLE_RATE=0.01 \
VLLM_BLOCK_USAGE_LOG_INTERVAL=90 \
VLLM_SKIP_SOFTMAX_OUTPUT_FILE=/tmp/skip_softmax_stats.json \
python -m vllm.entrypoints.openai.api_server --model <model>
```
4. **Disable after profiling**: Set `VLLM_SKIP_SOFTMAX_BLOCK_ANALYSIS=0` for production inference

The Triton kernel runs asynchronously but still adds overhead proportional to the sample rate.

## Phase 2 TODO: Head Skipping Implementation

### Analysis Results (Updated 2025-01-27)

Based on collected statistics from 30 stats files (threshold=2.0, 1% sampling, 53,090 sampled calls):

| Metric | Value |
|--------|-------|
| Total (layer, head) combinations | 2,240 (35 layers x 64 heads) |
| Total blocks analyzed | 28,037,791,808 |
| Heads with >= 99% sparsity | 551 (24.6%) - safe to skip entirely |
| Heads with >= 98% sparsity | 888 (39.6%) - can skip KV cache |
| Heads with >= 95% sparsity | 1,337 (59.7%) |
| Overall sparsity | 86.1% |

**Key Findings**:
- **VERY SPARSE layers** (≥98% - can skip most KV): L7, L9, L13, L19, L21, L25, L27, L29, L31, L35, L36
- **DENSE layers** (<70% - must keep all KV): L12 (68.9%), L28 (64.7%)
- Missing layers in analysis: L11, L15 (not sampled)

### Per-Layer Sparsity Summary

| Layer | ≥99% Heads | ≥98% Heads | Sparsity | Assessment |
|-------|------------|------------|----------|------------|
| L7 | 51 | 62 | 99.3% | SKIP KV |
| L9 | 52 | 61 | 99.4% | SKIP KV |
| L21 | 49 | 56 | 99.0% | SKIP KV |
| L13 | 23 | 56 | 98.8% | SKIP KV |
| L27 | 45 | 53 | 98.8% | SKIP KV |
| L19 | 29 | 53 | 98.4% | SKIP KV |
| L31 | 23 | 52 | 98.5% | SKIP KV |
| L25 | 30 | 50 | 98.4% | SKIP KV |
| L12 | 3 | 4 | 68.9% | DENSE - KEEP |
| L28 | 0 | 1 | 64.7% | DENSE - KEEP |

### Heads to Skip KV Cache (≥98% sparse)

```python
SKIP_KV_HEADS = {
    7: [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63],  # 62/64
    9: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,51,52,53,54,55,56,57,58,59,60,61,62,63],  # 61/64
    21: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,41,43,45,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63],  # 56/64
    13: [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55,56,57,60,61,62,63],  # 56/64
    27: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41,42,44,45,46,47,55,56,57,58,59,60,61,62,63],  # 53/64
    19: [0,1,2,3,4,5,6,7,8,9,12,13,14,19,20,21,22,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,59,60,62,63],  # 53/64
    31: [0,1,3,4,5,6,7,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49,50,51,52,53,54,55,56,57,58,62,63],  # 52/64
    25: [3,4,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,61],  # 50/64
    29: [0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,24,26,28,30,31,32,33,34,35,36,37,38,39,40,43,45,48,49,53,54,55,56,57,58,59,60,61,62,63],  # 45/64
    35: [0,1,2,3,4,5,6,7,9,11,13,14,15,17,19,20,21,22,23,26,27,32,33,34,40,41,42,43,44,45,46,47,48,49,51,53,54,55,56,57,58,60,61,62],  # 44/64
    3: [0,3,4,5,7,8,10,12,13,14,15,16,17,18,19,20,21,22,23,26,29,30,31,32,33,34,35,36,37,38,39,41,42,46,50,51,52,56,57,58,59,60,62,63],  # 44/64
    5: [1,4,7,8,9,10,11,12,13,14,16,17,18,20,21,24,25,26,28,31,32,37,38,39,42,43,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62],  # 42/64
    36: [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],  # 40/64
}
# Total: 888 heads (39.6% of all heads)
```

### TensorRT-LLM Reference

TRT-LLM's Skip Softmax implementation (from `/notebooks/pkgs/TensorRT-LLM/`):

1. **Threshold calculation**: `threshold = scale_factor / sequence_length` (dynamic per-sequence)
2. **Skip condition**: `exp(local_max - global_max) < threshold`
3. **No per-head masking**: TRT-LLM does NOT support static head-level skipping
4. **Sparse attention**: Uses token-level sparsity (RocketKV, DSA), not head-level

### Implementation Plan

1. **Config File Format**: Create a JSON config specifying which (layer, head) combinations to skip:
   ```json
   {
     "skip_kv_heads": {
       "7": [0,1,2,4,5,...],
       "9": [0,1,2,3,4,...],
       ...
     },
     "threshold_used": 2.0,
     "min_sparsity": 98.0
   }
   ```

2. **Modify FlashAttention Backend**:
   - Skip KV cache allocation for specified heads
   - Skip attention computation for those heads
   - Output zeros for skipped head positions
   - Add `skip_head_mask` tensor to attention metadata

3. **Runtime Toggle**:
   - `VLLM_SKIP_KV_HEADS_CONFIG=/path/to/config.json`
   - `VLLM_SKIP_KV_HEADS_ENABLED=1`

### Expected Benefits

| Optimization | Estimated Impact |
|--------------|------------------|
| KV cache memory savings | ~40% (888/2240 heads) |
| Attention compute reduction | ~40% for skipped heads |
| Overall decode speedup | 12-20% (memory-bound) |
| Overall prefill speedup | 16-24% (compute-bound) |
| Realistic end-to-end speedup | **15-25%** |

### Comparison with TRT-LLM Methods

| Method | Speedup | Notes |
|--------|---------|-------|
| TRT-LLM Skip Softmax | 1.3-1.6x | Dynamic block skipping |
| TRT-LLM RocketKV | 1.5-2.5x | Token-level KV reduction |
| Static Head Skipping (this) | 1.15-1.25x | 40% head reduction |

### Files to Modify

- `vllm/v1/attention/backends/flash_attn.py` - Core attention changes
- `vllm/v1/core/kv_cache_manager.py` - KV cache allocation changes
- `vllm/v1/core/kv_cache_metrics.py` - Generate skip config from stats
- New: `vllm/config/skip_heads_config.py` - Config loading and validation
