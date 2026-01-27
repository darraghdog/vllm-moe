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

### Analysis Results

Based on collected statistics from `skip_softmax_stats.json` (threshold=2.0, 10% sampling, ~20K sampled calls):

| Metric | Value |
|--------|-------|
| Total (layer, head) combinations | 1088 |
| Heads with >= 99% sparsity | 359 (33.0%) - safe to skip |
| Heads with >= 98% sparsity | 656 (60.3%) |
| Overall sparsity | 84.2% |

**Key Finding**: All 64 heads in Layer 3 (L3H0-L3H63) have 0% sparsity - these should never be skipped. This layer appears critical for attention.

### Implementation Plan

1. **Config File Format**: Create a JSON/YAML config specifying which (layer, head) combinations to skip:
   ```json
   {
     "skip_heads": [
       {"layer": 0, "head": 40},
       {"layer": 0, "head": 18},
       ...
     ],
     "min_sparsity_threshold": 99.0
   }
   ```

2. **Modify FlashAttention Backend**:
   - Skip KV cache writes for specified heads
   - Skip attention computation for those heads
   - Output zeros for skipped head positions
   - Add `skip_head_mask` tensor to attention call

3. **Runtime Toggle**: Environment variable to enable/disable head skipping

### Expected Benefits

| Optimization | Estimated Impact |
|--------------|------------------|
| Attention compute reduction | ~33% (if skipping >= 99% sparsity heads) |
| KV cache memory savings | ~33% for skipped heads |
| Overall inference speedup | 10-15% estimated |

### Files to Modify

- `vllm/v1/attention/backends/flash_attn.py` - Core attention changes
- `vllm/v1/core/kv_cache_manager.py` - KV cache allocation changes
- New: `vllm/config/skip_heads.py` - Config loading and validation
