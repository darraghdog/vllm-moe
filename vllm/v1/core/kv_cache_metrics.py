# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache metrics tracking."""

import json
import random
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

from vllm.v1.metrics.stats import KVCacheEvictionEvent

logger = init_logger(__name__)


class BlockMetricsState:
    """Tracks lifecycle metrics for a single KV cache block."""

    def __init__(self):
        now_ns = time.monotonic_ns()
        self.birth_time_ns = now_ns
        self.last_access_ns = now_ns
        # Bounded to prevent unbounded growth if a block is accessed many times.
        self.access_history: deque[int] = deque(maxlen=4)

    def record_access(self) -> None:
        now_ns = time.monotonic_ns()
        self.last_access_ns = now_ns
        self.access_history.append(now_ns)

    def get_lifetime_seconds(self) -> float:
        now_ns = time.monotonic_ns()
        return (now_ns - self.birth_time_ns) / 1e9

    def get_idle_time_seconds(self) -> float:
        now_ns = time.monotonic_ns()
        return (now_ns - self.last_access_ns) / 1e9

    def get_reuse_gaps_seconds(self) -> list[float]:
        if len(self.access_history) < 2:
            return []
        history = list(self.access_history)
        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]


class KVCacheMetricsCollector:
    """Collects KV cache residency metrics with sampling."""

    def __init__(self, sample_rate: float = 0.01):
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self.sample_rate = sample_rate

        self.block_metrics: dict[int, BlockMetricsState] = {}

        self._eviction_events: list[KVCacheEvictionEvent] = []

    def should_sample_block(self) -> bool:
        return random.random() < self.sample_rate

    def on_block_allocated(self, block: "KVCacheBlock") -> None:
        if self.should_sample_block():
            self.block_metrics[block.block_id] = BlockMetricsState()

    def on_block_accessed(self, block: "KVCacheBlock") -> None:
        metrics = self.block_metrics.get(block.block_id)
        if metrics:
            metrics.record_access()

    def on_block_evicted(self, block: "KVCacheBlock") -> None:
        metrics = self.block_metrics.pop(block.block_id, None)
        if not metrics:
            return

        lifetime = metrics.get_lifetime_seconds()
        idle_time = metrics.get_idle_time_seconds()
        reuse_gaps = tuple(metrics.get_reuse_gaps_seconds())

        self._eviction_events.append(
            KVCacheEvictionEvent(
                lifetime_seconds=lifetime,
                idle_seconds=idle_time,
                reuse_gaps_seconds=reuse_gaps,
            )
        )

    def reset(self) -> None:
        """Clear all state on cache reset."""
        self.block_metrics.clear()
        self._eviction_events.clear()

    def drain_events(self) -> list[KVCacheEvictionEvent]:
        events = self._eviction_events
        self._eviction_events = []
        return events


class BlockUsageCollector:
    """Collects block usage statistics during inference.

    This class tracks which KV cache blocks are accessed during attention
    computation, inspired by TensorRT-LLM's Skip Softmax block tracking.
    It provides insights into cache utilization patterns that can inform
    sparse attention optimizations.
    """

    def __init__(self, enabled: bool = False, log_interval: float = 30.0):
        self.enabled = enabled
        self.log_interval = log_interval
        self._block_access_counts: dict[int, int] = defaultdict(int)
        self._total_requests = 0
        self._total_blocks_accessed = 0
        self._last_log_time = time.monotonic()

    def record_batch_blocks(
        self,
        block_table: torch.Tensor,
        num_reqs: int,
        seq_lens: torch.Tensor,
        block_size: int = 16,
    ) -> None:
        """Record which blocks are accessed in this batch.

        Args:
            block_table: Tensor of shape [num_reqs, max_blocks] containing
                block IDs for each request.
            num_reqs: Number of requests in this batch.
            seq_lens: Tensor of shape [num_reqs] containing sequence lengths.
            block_size: Size of each KV cache block (default 16).
        """
        if not self.enabled:
            return

        block_table_cpu = block_table.cpu()
        seq_lens_cpu = seq_lens.cpu()

        for req_idx in range(num_reqs):
            seq_len = int(seq_lens_cpu[req_idx].item())
            num_blocks = (seq_len + block_size - 1) // block_size
            for b in range(num_blocks):
                block_id = int(block_table_cpu[req_idx, b].item())
                self._block_access_counts[block_id] += 1
            self._total_blocks_accessed += num_blocks
        self._total_requests += num_reqs

        self._maybe_log()

    def _maybe_log(self) -> None:
        """Log stats if log interval has elapsed."""
        now = time.monotonic()
        if now - self._last_log_time >= self.log_interval:
            self._log_stats()
            self._last_log_time = now

    def _log_stats(self) -> None:
        """Log current block usage statistics."""
        unique_blocks = len(self._block_access_counts)
        total_accesses = sum(self._block_access_counts.values())

        # Build access frequency histogram
        hist: dict[str, int] = defaultdict(int)
        for count in self._block_access_counts.values():
            if count == 1:
                hist["1x"] += 1
            elif count <= 5:
                hist["2-5x"] += 1
            elif count <= 10:
                hist["6-10x"] += 1
            else:
                hist[">10x"] += 1

        avg_blocks_per_req = (
            self._total_blocks_accessed / max(self._total_requests, 1)
        )

        logger.info(
            "Block Usage: unique_blocks=%d, total_accesses=%d, "
            "total_requests=%d, avg_blocks/req=%.1f",
            unique_blocks,
            total_accesses,
            self._total_requests,
            avg_blocks_per_req,
        )
        logger.info("Block access histogram: %s", dict(hist))

    def force_log(self) -> None:
        """Force logging of current stats."""
        if self.enabled:
            self._log_stats()

    def reset(self) -> None:
        """Reset all statistics."""
        self._block_access_counts.clear()
        self._total_requests = 0
        self._total_blocks_accessed = 0
        self._last_log_time = time.monotonic()


class AttentionSparsityCollector:
    """Collects attention sparsity statistics for Skip Softmax analysis.

    Tracks which attention blocks have low scores relative to the global max,
    indicating they could potentially be skipped (TensorRT-LLM Skip Softmax style).

    Uses LSE (log-sum-exp) from FlashAttention as a proxy for max logits.
    """

    def __init__(self, enabled: bool = False, log_interval: float = 30.0,
                 threshold: float = 30.0):
        self.enabled = enabled
        self.log_interval = log_interval
        self.threshold = threshold
        self._total_blocks = 0
        self._skippable_blocks = 0
        self._total_calls = 0
        self._last_log_time = time.monotonic()
        # Per-head statistics
        self._per_head_total: dict[int, int] = defaultdict(int)
        self._per_head_skippable: dict[int, int] = defaultdict(int)

    def record_attention_lse(self, lse: torch.Tensor, num_blocks: int) -> None:
        """Record sparsity from LSE values.

        Args:
            lse: LSE tensor of shape [num_heads, batch_size] or
                [batch_size, num_heads]
            num_blocks: Number of KV blocks in this attention computation
        """
        if not self.enabled or lse is None:
            return

        # Ensure shape is [num_heads, batch]
        if lse.dim() == 2 and lse.shape[0] < lse.shape[1]:
            lse = lse.transpose(0, 1)

        lse_cpu = lse.float().cpu()
        num_heads = lse_cpu.shape[0]
        batch_size = lse_cpu.shape[1]

        # Compute m_global (max across all positions) per head
        m_global = lse_cpu.max(dim=1)[0]  # [num_heads]

        # Count skippable blocks: where (m_global - m_local) > threshold
        for h in range(num_heads):
            diff = m_global[h] - lse_cpu[h]  # [batch_size]
            skippable = (diff > self.threshold).sum().item()
            self._per_head_skippable[h] += int(skippable)
            self._per_head_total[h] += batch_size

        self._total_blocks += num_heads * batch_size
        self._skippable_blocks += sum(
            (m_global[h] - lse_cpu[h] > self.threshold).sum().item()
            for h in range(num_heads)
        )
        self._total_calls += 1

        self._maybe_log()

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log_time >= self.log_interval:
            self._log_stats()
            self._last_log_time = now

    def _log_stats(self) -> None:
        if self._total_blocks == 0:
            return

        sparsity_pct = 100.0 * self._skippable_blocks / self._total_blocks

        logger.info(
            "Attention Sparsity: total_blocks=%d, skippable_blocks=%d, "
            "sparsity=%.1f%%, threshold=%.1f, calls=%d",
            self._total_blocks,
            self._skippable_blocks,
            sparsity_pct,
            self.threshold,
            self._total_calls,
        )

        # Per-head breakdown (top 3 and bottom 3 by sparsity)
        head_sparsity = {}
        for h in self._per_head_total:
            if self._per_head_total[h] > 0:
                head_sparsity[h] = (
                    100.0 * self._per_head_skippable[h] / self._per_head_total[h]
                )

        if head_sparsity:
            sorted_heads = sorted(
                head_sparsity.items(), key=lambda x: x[1], reverse=True
            )
            top_sparse = sorted_heads[:3]
            low_sparse = sorted_heads[-3:]
            logger.info(
                "Per-head sparsity: highest=%s, lowest=%s",
                {h: f"{s:.1f}%" for h, s in top_sparse},
                {h: f"{s:.1f}%" for h, s in low_sparse},
            )

    def force_log(self) -> None:
        if self.enabled:
            self._log_stats()

    def reset(self) -> None:
        self._total_blocks = 0
        self._skippable_blocks = 0
        self._total_calls = 0
        self._per_head_total.clear()
        self._per_head_skippable.clear()
        self._last_log_time = time.monotonic()


class SkipSoftmaxBlockAnalyzer:
    """True per-KV-block Skip Softmax analysis (TensorRT-LLM style).

    Unlike AttentionSparsityCollector which uses LSE (one value per query
    position), this analyzer computes per-KV-block max logits using a
    custom Triton kernel. This matches TensorRT-LLM's Skip Softmax granularity.

    Tracks statistics per (layer, head) at multiple thresholds to identify
    which specific attention heads in which layers are most sparse.

    The TensorRT-LLM algorithm:
    1. For each KV block, compute m_local = max(Q * K[block]^T)
    2. Track m_global = max across all KV blocks
    3. If (m_global - m_local) > threshold, the block can be skipped
    """

    # Default thresholds for multi-threshold analysis
    DEFAULT_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    def __init__(
        self,
        enabled: bool = False,
        threshold: float = 2.0,
        log_interval: float = 30.0,
        sample_rate: float = 1.0,
        output_file: str | None = None,
        thresholds: list[float] | None = None,
    ):
        """Initialize the Skip Softmax block analyzer.

        Args:
            enabled: Whether analysis is enabled
            threshold: Primary skip threshold for logging (default 2.0)
            log_interval: Seconds between log messages
            sample_rate: Fraction of attention calls to analyze (0.0-1.0).
                Use < 1.0 to reduce overhead while still collecting stats.
            output_file: Path to write full statistics JSON. If set, all
                layer/head stats are written to this file on each log interval.
            thresholds: List of thresholds to track (default: DEFAULT_THRESHOLDS).
                Stats are collected at each threshold for post-hoc analysis.
        """
        self.enabled = enabled
        self.threshold = threshold
        self.log_interval = log_interval
        self.sample_rate = sample_rate
        self.output_file = output_file
        self.thresholds = thresholds if thresholds is not None else self.DEFAULT_THRESHOLDS

        self._total_calls = 0
        self._sampled_calls = 0
        self._last_log_time = time.monotonic()

        # Per-(layer, head) statistics with multi-threshold support
        # Key: (layer_name, head_idx)
        # Value: total blocks for this (layer, head)
        self._per_layer_head_total: dict[tuple[str, int], int] = defaultdict(int)
        # Value: number of readings/samples for this (layer, head)
        self._per_layer_head_readings: dict[tuple[str, int], int] = defaultdict(int)
        # Value: dict of threshold -> skippable block count
        self._per_layer_head_at_threshold: dict[tuple[str, int], dict[float, int]] = defaultdict(
            lambda: {t: 0 for t in self.thresholds}
        )

        # Per-layer statistics (aggregated across heads)
        self._per_layer_total: dict[str, int] = defaultdict(int)
        self._per_layer_readings: dict[str, int] = defaultdict(int)

    def analyze_attention(
        self,
        query: torch.Tensor,  # [num_tokens, num_heads, head_dim]
        key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: torch.Tensor,  # [num_seqs, max_blocks]
        seq_lens: torch.Tensor,  # [num_seqs]
        query_start_loc: torch.Tensor,  # [num_seqs + 1]
        scale: float,
        block_size: int = 16,
        layer_name: str = "unknown",
    ) -> None:
        """Run Skip Softmax analysis on the attention computation.

        This runs a separate Triton kernel to compute per-KV-block max logits,
        then counts how many blocks could potentially be skipped.

        Args:
            query: Query tensor [num_tokens, num_heads, head_dim]
            key_cache: KV cache tensor [num_blocks, block_size, num_kv_heads, head_dim]
            block_table: Block table [num_seqs, max_blocks_per_seq]
            seq_lens: Sequence lengths [num_seqs]
            query_start_loc: Cumulative query positions [num_seqs + 1]
            scale: Softmax scale factor
            block_size: KV cache block size
            layer_name: Name of the attention layer (e.g., "model.layers.5.self_attn")
        """
        if not self.enabled:
            return

        self._total_calls += 1

        # Sample to reduce overhead
        if self.sample_rate < 1.0 and random.random() > self.sample_rate:
            return

        self._sampled_calls += 1

        # Import here to avoid circular imports
        from vllm.attention.ops.triton_skip_softmax_analysis import (
            compute_skip_softmax_sparsity_multi_threshold,
            skip_softmax_block_analysis,
        )

        # Run the analysis kernel
        block_max = skip_softmax_block_analysis(
            query=query,
            key_cache=key_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            scale=scale,
            block_size=block_size,
        )

        # Compute sparsity statistics at multiple thresholds
        per_head_total, per_head_at_threshold = compute_skip_softmax_sparsity_multi_threshold(
            block_max=block_max,
            seq_lens=seq_lens,
            thresholds=self.thresholds,
            block_size=block_size,
        )

        # Update per-layer stats
        layer_total = sum(per_head_total.values())
        self._per_layer_total[layer_name] += layer_total
        self._per_layer_readings[layer_name] += 1

        # Update per-(layer, head) stats with multi-threshold data
        for h, head_total in per_head_total.items():
            key = (layer_name, h)
            self._per_layer_head_total[key] += head_total
            self._per_layer_head_readings[key] += 1
            # Update skippable counts at each threshold
            for t, skippable in per_head_at_threshold[h].items():
                self._per_layer_head_at_threshold[key][t] += skippable

        self._maybe_log()

    def _maybe_log(self) -> None:
        """Log stats if log interval has elapsed."""
        now = time.monotonic()
        if now - self._last_log_time >= self.log_interval:
            self._log_stats()
            self._last_log_time = now

    def _extract_layer_idx(self, layer_name: str) -> int | None:
        """Extract layer index from layer name like 'model.layers.5.self_attn'."""
        import re
        match = re.search(r'layers\.(\d+)', layer_name)
        if match:
            return int(match.group(1))
        return None

    def _log_stats(self) -> None:
        """Log current Skip Softmax block analysis statistics."""
        total_blocks = sum(self._per_layer_head_total.values())
        if total_blocks == 0:
            return

        # Compute overall sparsity at primary threshold
        total_skippable = sum(
            self._per_layer_head_at_threshold[key].get(self.threshold, 0)
            for key in self._per_layer_head_total
        )
        sparsity_pct = 100.0 * total_skippable / total_blocks

        sample_info = ""
        if self.sample_rate < 1.0:
            sample_info = f", sample_rate={self.sample_rate:.1%}, sampled={self._sampled_calls}/{self._total_calls}"

        logger.info(
            "Skip Softmax Block Analysis: total_kv_blocks=%d, "
            "skippable_blocks=%d, sparsity=%.1f%%, threshold=%.1f%s",
            total_blocks,
            total_skippable,
            sparsity_pct,
            self.threshold,
            sample_info,
        )

        # Per-(layer, head) breakdown at primary threshold
        layer_head_sparsity = {}
        layer_head_data = {}  # Full data for file output
        for (layer_name, head), total in self._per_layer_head_total.items():
            if total > 0:
                layer_idx = self._extract_layer_idx(layer_name)
                readings = self._per_layer_head_readings[(layer_name, head)]
                thresh_data = self._per_layer_head_at_threshold[(layer_name, head)]

                # Sparsity at primary threshold for logging
                skippable = thresh_data.get(self.threshold, 0)
                sparsity = 100.0 * skippable / total
                layer_head_sparsity[(layer_idx, head)] = sparsity

                # Full data for file output
                layer_head_data[(layer_idx, head)] = {
                    "total_blocks": total,
                    "readings": readings,
                    "at_threshold": thresh_data,
                }

        if layer_head_sparsity:
            sorted_lh = sorted(
                layer_head_sparsity.items(), key=lambda x: x[1], reverse=True
            )
            top_sparse_lh = sorted_lh[:10]
            low_sparse_lh = sorted_lh[-10:]

            logger.info(
                "Most sparse (layer, head) at threshold=%.1f: %s",
                self.threshold,
                {f"L{l}H{h}": f"{s:.1f}%" for (l, h), s in top_sparse_lh},
            )
            logger.info(
                "Least sparse (layer, head) at threshold=%.1f: %s",
                self.threshold,
                {f"L{l}H{h}": f"{s:.1f}%" for (l, h), s in low_sparse_lh},
            )

            # Write full statistics to file if output_file is set
            if self.output_file:
                self._write_stats_to_file(layer_head_data, total_blocks, total_skippable)

    def _write_stats_to_file(
        self,
        layer_head_data: dict[tuple[int | None, int], dict],
        total_blocks: int,
        total_skippable: int,
    ) -> None:
        """Write full statistics to JSON file with multi-threshold data.

        Args:
            layer_head_data: Dict mapping (layer_idx, head_idx) to stats dict
            total_blocks: Total blocks across all layer/heads
            total_skippable: Total skippable blocks at primary threshold
        """
        try:
            overall_sparsity = 100.0 * total_skippable / total_blocks if total_blocks > 0 else 0.0

            # Compute overall stats at each threshold
            overall_at_threshold = {t: 0 for t in self.thresholds}
            for data in layer_head_data.values():
                for t, count in data["at_threshold"].items():
                    overall_at_threshold[t] += count

            # Build full statistics dict
            stats = {
                "timestamp": time.time(),
                "primary_threshold": self.threshold,
                "thresholds": self.thresholds,
                "sample_rate": self.sample_rate,
                "overall": {
                    "total_blocks": total_blocks,
                    "skippable_blocks": total_skippable,
                    "sparsity_pct": round(overall_sparsity, 2),
                    "total_calls": self._total_calls,
                    "sampled_calls": self._sampled_calls,
                    "at_threshold": {
                        str(t): {
                            "skippable": overall_at_threshold[t],
                            "sparsity_pct": round(100.0 * overall_at_threshold[t] / total_blocks, 2) if total_blocks > 0 else 0.0,
                        }
                        for t in self.thresholds
                    },
                },
                "per_layer_head": {},
            }

            # Add all layer/head combinations sorted by sparsity at primary threshold (descending)
            sorted_lh = sorted(
                layer_head_data.items(),
                key=lambda x: x[1]["at_threshold"].get(self.threshold, 0) / max(x[1]["total_blocks"], 1),
                reverse=True,
            )
            for (layer_idx, head_idx), data in sorted_lh:
                key = f"L{layer_idx}H{head_idx}"
                head_total = data["total_blocks"]
                stats["per_layer_head"][key] = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "total_blocks": head_total,
                    "readings": data["readings"],
                    "at_threshold": {
                        str(t): {
                            "skippable": data["at_threshold"][t],
                            "sparsity_pct": round(100.0 * data["at_threshold"][t] / head_total, 2) if head_total > 0 else 0.0,
                        }
                        for t in self.thresholds
                    },
                }

            # Write to file (overwrite each time)
            with open(self.output_file, "w") as f:
                json.dump(stats, f, indent=2)

            logger.info("Wrote skip softmax stats to %s", self.output_file)
        except Exception as e:
            logger.warning("Failed to write skip softmax stats to file: %s", e)

    def force_log(self) -> None:
        """Force logging of current stats."""
        if self.enabled:
            self._log_stats()

    def reset(self) -> None:
        """Reset all statistics."""
        self._total_calls = 0
        self._sampled_calls = 0
        self._per_layer_head_total.clear()
        self._per_layer_head_readings.clear()
        self._per_layer_head_at_threshold.clear()
        self._per_layer_total.clear()
        self._per_layer_readings.clear()
        self._last_log_time = time.monotonic()
