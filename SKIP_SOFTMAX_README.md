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

When `VLLM_SKIP_SOFTMAX_OUTPUT_FILE` is set, ALL layer/head statistics are written to a JSON file:

```json
{
  "timestamp": 1706356281.123,
  "threshold": 2.0,
  "sample_rate": 0.1,
  "overall": {
    "total_blocks": 100000,
    "skippable_blocks": 42000,
    "sparsity_pct": 42.0,
    "total_calls": 1000,
    "sampled_calls": 100
  },
  "per_layer_head": {
    "L36H46": {"layer": 36, "head": 46, "sparsity_pct": 99.7},
    "L36H12": {"layer": 36, "head": 12, "sparsity_pct": 99.7},
    ...
  }
}
```

The `per_layer_head` section is sorted by sparsity (highest first), making it easy to identify the most sparse (rarely used) attention heads.

## Files

| File | Description |
|------|-------------|
| `vllm/attention/ops/triton_skip_softmax_analysis.py` | Triton kernel for per-KV-block max logit computation |
| `vllm/v1/core/kv_cache_metrics.py` | `SkipSoftmaxBlockAnalyzer`, `BlockUsageCollector`, `AttentionSparsityCollector` classes |
| `vllm/v1/attention/backends/flash_attn.py` | Integration with FlashAttention backend |
| `vllm/envs.py` | Environment variable definitions |

## Performance Considerations

To minimize impact on inference throughput:

1. **Reduce sample rate**: `VLLM_SKIP_SOFTMAX_SAMPLE_RATE=0.1` (10% sampling)
2. **Increase log interval**: `VLLM_BLOCK_USAGE_LOG_INTERVAL=300` (5 minutes instead of 30 seconds)
3. **Use file output**: Set `VLLM_SKIP_SOFTMAX_OUTPUT_FILE` to get ALL stats without verbose logging
4. **Disable after profiling**: Set `VLLM_SKIP_SOFTMAX_BLOCK_ANALYSIS=0` for production inference

The Triton kernel runs asynchronously but still adds overhead proportional to the sample rate.
