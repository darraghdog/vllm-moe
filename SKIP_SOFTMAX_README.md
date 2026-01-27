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
| `VLLM_BLOCK_USAGE_STATS` | bool | `0` | Enable KV block usage statistics |
| `VLLM_BLOCK_USAGE_LOG_INTERVAL` | float | `30.0` | Logging interval in seconds |
| `VLLM_ATTENTION_SPARSITY_STATS` | bool | `0` | Enable attention sparsity statistics |
| `VLLM_ATTENTION_SPARSITY_THRESHOLD` | float | `30.0` | Threshold for attention sparsity analysis |

## Usage

Set environment variables before starting the vLLM server:

```python
import os
os.environ["VLLM_SKIP_SOFTMAX_BLOCK_ANALYSIS"] = "1"
os.environ["VLLM_SKIP_SOFTMAX_THRESHOLD"] = "2.0"
os.environ["VLLM_SKIP_SOFTMAX_SAMPLE_RATE"] = "0.1"  # 10% sampling to reduce overhead
```

Or via command line:

```bash
VLLM_SKIP_SOFTMAX_BLOCK_ANALYSIS=1 \
VLLM_SKIP_SOFTMAX_THRESHOLD=2.0 \
VLLM_SKIP_SOFTMAX_SAMPLE_RATE=0.1 \
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

## Files

| File | Description |
|------|-------------|
| `vllm/attention/ops/triton_skip_softmax_analysis.py` | Triton kernel for per-KV-block max logit computation |
| `vllm/v1/core/kv_cache_metrics.py` | `SkipSoftmaxBlockAnalyzer`, `BlockUsageCollector`, `AttentionSparsityCollector` classes |
| `vllm/v1/attention/backends/flash_attn.py` | Integration with FlashAttention backend |
| `vllm/envs.py` | Environment variable definitions |

## Performance Considerations

- Use `VLLM_SKIP_SOFTMAX_SAMPLE_RATE=0.1` (10% sampling) to reduce overhead during production profiling
- The Triton kernel runs asynchronously but still adds some overhead
- Disable analysis (`VLLM_SKIP_SOFTMAX_BLOCK_ANALYSIS=0`) for production inference after profiling is complete
