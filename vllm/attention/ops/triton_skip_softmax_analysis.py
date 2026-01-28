# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton kernel for Skip Softmax block analysis.

This kernel computes per-KV-block max logits for TensorRT-LLM style
Skip Softmax analysis. Unlike FlashAttention's LSE output (which gives
one value per query position), this kernel computes per-KV-block max
logits for true block-level sparsity analysis.

The key insight from TensorRT-LLM's Skip Softmax:
- For each KV block, compute m_local = max(Q * K[block]^T)
- Compare against m_global (max across all blocks)
- Blocks where (m_global - m_local) > threshold can be skipped

This kernel runs as a separate analysis pass after the main attention,
allowing us to measure potential Skip Softmax sparsity without modifying
the production attention kernel.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _skip_softmax_block_analysis_kernel(
    # Inputs
    Q_ptr,  # [num_tokens, num_heads, head_dim]
    K_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    query_start_loc_ptr,  # [num_seqs + 1]
    # Output
    block_max_ptr,  # [num_seqs, max_num_blocks, num_heads]
    # Dimensions
    num_seqs,
    max_num_blocks_per_seq,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    scale,
    # Strides
    stride_q_token,
    stride_q_head,
    stride_k_block,
    stride_k_pos,
    stride_k_head,
    stride_bt_seq,
    stride_out_seq,
    stride_out_block,
    # Block dimensions
    BLOCK_M: tl.constexpr,  # Query block size
    BLOCK_D: tl.constexpr,  # Head dimension block (rounded up)
):
    """Compute per-KV-block max logits for Skip Softmax analysis.

    Each program handles one (seq_idx, kv_block_idx, head_idx) tuple.
    For each KV block, we compute max(Q @ K[block]^T) across all query
    positions in the sequence.
    """
    # Program IDs
    seq_idx = tl.program_id(0)
    kv_block_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # Bounds check - exit early if this block doesn't exist for this sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_kv_blocks = (seq_len + block_size - 1) // block_size

    if kv_block_idx >= num_kv_blocks:
        # Store -inf for invalid blocks
        out_idx = (
            seq_idx * stride_out_seq + kv_block_idx * stride_out_block + head_idx
        )
        tl.store(block_max_ptr + out_idx, float("-inf"))
        return

    # Get physical block ID from block table
    physical_block = tl.load(
        block_table_ptr + seq_idx * stride_bt_seq + kv_block_idx
    )

    # Get query range for this sequence
    q_start = tl.load(query_start_loc_ptr + seq_idx)
    q_end = tl.load(query_start_loc_ptr + seq_idx + 1)
    num_query_tokens = q_end - q_start

    # Handle GQA: map query head to KV head
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # Initialize running max
    block_max = float("-inf")

    # Determine valid positions in this KV block
    kv_block_start = kv_block_idx * block_size
    kv_block_end = tl.minimum(kv_block_start + block_size, seq_len)
    valid_kv_positions = kv_block_end - kv_block_start

    # Offsets for head dimension
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    # Load K block for this head: iterate over valid positions in the block
    # K shape in cache: [num_blocks, block_size, num_kv_heads, head_dim]
    for kv_pos in range(valid_kv_positions):
        # Load one K vector: [head_dim] and cast to float32 for FP8 support
        k_ptr = (
            K_cache_ptr
            + physical_block * stride_k_block
            + kv_pos * stride_k_pos
            + kv_head_idx * stride_k_head
            + offs_d
        )
        k_vec = tl.load(k_ptr, mask=mask_d, other=0.0).to(tl.float32)

        # Iterate over query positions in chunks
        for q_offset in range(0, num_query_tokens, BLOCK_M):
            # Determine how many queries in this chunk
            q_chunk_size = tl.minimum(BLOCK_M, num_query_tokens - q_offset)

            # Load Q vectors for this chunk
            offs_q = tl.arange(0, BLOCK_M)
            mask_q = offs_q < q_chunk_size

            # For each query in the chunk, compute dot product with k_vec
            for q_local in range(BLOCK_M):
                if q_local < q_chunk_size:
                    q_token_idx = q_start + q_offset + q_local
                    # Load Q vector and cast to float32 for FP8 support
                    q_ptr = (
                        Q_ptr
                        + q_token_idx * stride_q_token
                        + head_idx * stride_q_head
                        + offs_d
                    )
                    q_vec = tl.load(q_ptr, mask=mask_d, other=0.0).to(tl.float32)

                    # Compute dot product
                    qk = tl.sum(q_vec * k_vec)
                    qk = qk * scale

                    # Update running max
                    block_max = tl.maximum(block_max, qk)

    # Store result
    out_idx = seq_idx * stride_out_seq + kv_block_idx * stride_out_block + head_idx
    tl.store(block_max_ptr + out_idx, block_max)


@triton.jit
def _skip_softmax_block_analysis_kernel_v2(
    # Inputs
    Q_ptr,  # [num_tokens, num_heads, head_dim]
    K_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    query_start_loc_ptr,  # [num_seqs + 1]
    # Output
    block_max_ptr,  # [num_seqs, max_num_blocks, num_heads]
    # Dimensions
    num_seqs,
    max_num_blocks_per_seq,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    scale,
    # Strides
    stride_q_token,
    stride_q_head,
    stride_k_block,
    stride_k_pos,
    stride_k_head,
    stride_bt_seq,
    stride_out_seq,
    stride_out_block,
    # Block dimensions
    BLOCK_D: tl.constexpr,  # Head dimension block (rounded up)
):
    """Optimized kernel that processes one Q token at a time against whole KV block.

    Each program handles one (seq_idx, kv_block_idx, head_idx) tuple.
    """
    # Program IDs
    seq_idx = tl.program_id(0)
    kv_block_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # Bounds check
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_kv_blocks = (seq_len + block_size - 1) // block_size

    if kv_block_idx >= num_kv_blocks:
        out_idx = (
            seq_idx * stride_out_seq + kv_block_idx * stride_out_block + head_idx
        )
        tl.store(block_max_ptr + out_idx, float("-inf"))
        return

    # Get physical block ID
    physical_block = tl.load(
        block_table_ptr + seq_idx * stride_bt_seq + kv_block_idx
    )

    # Get query range
    q_start = tl.load(query_start_loc_ptr + seq_idx)
    q_end = tl.load(query_start_loc_ptr + seq_idx + 1)
    num_query_tokens = q_end - q_start

    # Handle GQA
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # Valid KV positions
    kv_block_end = tl.minimum((kv_block_idx + 1) * block_size, seq_len)
    valid_kv_positions = kv_block_end - kv_block_idx * block_size

    # Initialize
    block_max = float("-inf")
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    # Pre-load all K vectors for this block into registers
    # This is more efficient for small block sizes
    # Process each query token
    for q_local in range(num_query_tokens):
        q_token_idx = q_start + q_local

        # Load Q vector and cast to float32 (handles FP8 KV cache)
        q_ptr = (
            Q_ptr
            + q_token_idx * stride_q_token
            + head_idx * stride_q_head
            + offs_d
        )
        q_vec = tl.load(q_ptr, mask=mask_d, other=0.0).to(tl.float32)

        # Compute dot product with each K in the block
        for kv_pos in range(block_size):
            if kv_pos < valid_kv_positions:
                k_ptr = (
                    K_cache_ptr
                    + physical_block * stride_k_block
                    + kv_pos * stride_k_pos
                    + kv_head_idx * stride_k_head
                    + offs_d
                )
                # Cast to float32 to handle FP8 KV cache
                k_vec = tl.load(k_ptr, mask=mask_d, other=0.0).to(tl.float32)

                qk = tl.sum(q_vec * k_vec) * scale
                block_max = tl.maximum(block_max, qk)

    # Store result
    out_idx = seq_idx * stride_out_seq + kv_block_idx * stride_out_block + head_idx
    tl.store(block_max_ptr + out_idx, block_max)


def skip_softmax_block_analysis(
    query: torch.Tensor,  # [num_tokens, num_heads, head_dim]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table: torch.Tensor,  # [num_seqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_seqs]
    query_start_loc: torch.Tensor,  # [num_seqs + 1]
    scale: float,
    block_size: int = 16,
) -> torch.Tensor:
    """Compute per-KV-block max logits for Skip Softmax analysis.

    This function computes the maximum attention logit (Q @ K^T) for each
    KV cache block, which is used to determine which blocks could potentially
    be skipped during attention computation (TensorRT-LLM Skip Softmax style).

    Args:
        query: Query tensor of shape [num_tokens, num_heads, head_dim]
        key_cache: KV cache tensor of shape [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: Block table mapping sequences to physical blocks
                     Shape: [num_seqs, max_blocks_per_seq]
        seq_lens: Sequence lengths tensor of shape [num_seqs]
        query_start_loc: Cumulative query lengths of shape [num_seqs + 1]
        scale: Softmax scale factor (typically 1/sqrt(head_dim))
        block_size: Size of each KV cache block

    Returns:
        block_max: Tensor of shape [num_seqs, max_num_blocks, num_heads]
                   containing the maximum attention logit for each KV block.
                   Invalid blocks (beyond sequence length) contain -inf.
    """
    num_seqs = seq_lens.shape[0]
    max_num_blocks = block_table.shape[1]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = key_cache.shape[2]

    # Output tensor
    block_max = torch.full(
        (num_seqs, max_num_blocks, num_heads),
        float("-inf"),
        device=query.device,
        dtype=torch.float32,
    )

    # Grid: one program per (seq, kv_block, head)
    grid = (num_seqs, max_num_blocks, num_heads)

    # Round up head_dim to power of 2 for Triton
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Launch kernel
    _skip_softmax_block_analysis_kernel_v2[grid](
        query,
        key_cache,
        block_table,
        seq_lens,
        query_start_loc,
        block_max,
        num_seqs,
        max_num_blocks,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        scale,
        # Strides for Q: [num_tokens, num_heads, head_dim]
        query.stride(0),
        query.stride(1),
        # Strides for K cache: [num_blocks, block_size, num_kv_heads, head_dim]
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        # Stride for block table: [num_seqs, max_blocks]
        block_table.stride(0),
        # Strides for output: [num_seqs, max_blocks, num_heads]
        block_max.stride(0),
        block_max.stride(1),
        # Block dimensions
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    return block_max


# Default thresholds for multi-threshold analysis
DEFAULT_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def compute_skip_softmax_sparsity(
    block_max: torch.Tensor,  # [num_seqs, max_num_blocks, num_heads]
    seq_lens: torch.Tensor,  # [num_seqs]
    threshold: float,
    block_size: int = 16,
) -> tuple[int, int, dict[int, tuple[int, int]]]:
    """Compute Skip Softmax sparsity from per-block max logits.

    Args:
        block_max: Per-block max logits from skip_softmax_block_analysis
        seq_lens: Sequence lengths
        threshold: Skip threshold (blocks with m_global - m_local > threshold are skippable)
        block_size: KV cache block size

    Returns:
        Tuple of (total_blocks, skippable_blocks, per_head_stats)
        where per_head_stats is a dict mapping head_idx to (total, skippable)
    """
    num_seqs, max_num_blocks, num_heads = block_max.shape

    total_blocks = 0
    skippable_blocks = 0
    per_head_stats: dict[int, tuple[int, int]] = {}

    # Process on CPU for accurate counting
    block_max_cpu = block_max.cpu().float()
    seq_lens_cpu = seq_lens.cpu()

    for h in range(num_heads):
        head_total = 0
        head_skippable = 0

        for s in range(num_seqs):
            seq_len = int(seq_lens_cpu[s].item())
            num_blocks = (seq_len + block_size - 1) // block_size

            if num_blocks == 0:
                continue

            # Get valid block max values for this sequence and head
            valid_block_max = block_max_cpu[s, :num_blocks, h]

            # Compute global max for this sequence/head
            m_global = valid_block_max.max().item()

            # Count skippable blocks
            diff = m_global - valid_block_max
            skippable = (diff > threshold).sum().item()

            head_total += num_blocks
            head_skippable += int(skippable)

        per_head_stats[h] = (head_total, head_skippable)
        total_blocks += head_total
        skippable_blocks += head_skippable

    return total_blocks, skippable_blocks, per_head_stats


def compute_skip_softmax_sparsity_multi_threshold(
    block_max: torch.Tensor,  # [num_seqs, max_num_blocks, num_heads]
    seq_lens: torch.Tensor,  # [num_seqs]
    thresholds: list[float] | None = None,
    block_size: int = 16,
) -> tuple[int, dict[int, int], dict[int, dict[float, int]]]:
    """Compute Skip Softmax sparsity at multiple thresholds.

    Args:
        block_max: Per-block max logits from skip_softmax_block_analysis
        seq_lens: Sequence lengths
        thresholds: List of thresholds to evaluate (default: DEFAULT_THRESHOLDS)
        block_size: KV cache block size

    Returns:
        Tuple of (total_blocks_per_head, per_head_total, per_head_skippable_at_threshold)
        - total_blocks_per_head: dict mapping head_idx to total block count
        - per_head_skippable_at_threshold: dict mapping head_idx to dict of threshold -> skippable count
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    num_seqs, max_num_blocks, num_heads = block_max.shape

    # Per-head stats: head -> total blocks
    per_head_total: dict[int, int] = {}
    # Per-head stats at each threshold: head -> {threshold -> skippable count}
    per_head_at_threshold: dict[int, dict[float, int]] = {}

    # Process on CPU for accurate counting
    block_max_cpu = block_max.cpu().float()
    seq_lens_cpu = seq_lens.cpu()

    for h in range(num_heads):
        head_total = 0
        head_skippable_at_thresh = {t: 0 for t in thresholds}

        for s in range(num_seqs):
            seq_len = int(seq_lens_cpu[s].item())
            num_blocks = (seq_len + block_size - 1) // block_size

            if num_blocks == 0:
                continue

            # Get valid block max values for this sequence and head
            valid_block_max = block_max_cpu[s, :num_blocks, h]

            # Compute global max for this sequence/head
            m_global = valid_block_max.max().item()

            # Compute diff once, then count at each threshold
            diff = m_global - valid_block_max

            head_total += num_blocks
            for t in thresholds:
                skippable = (diff > t).sum().item()
                head_skippable_at_thresh[t] += int(skippable)

        per_head_total[h] = head_total
        per_head_at_threshold[h] = head_skippable_at_thresh

    return per_head_total, per_head_at_threshold


def compute_block_skip_mask(
    block_max_scores: torch.Tensor,  # [num_seqs, max_num_blocks, num_heads]
    seq_lens: torch.Tensor,  # [num_seqs]
    scale_factor: float,
    block_size: int = 16,
) -> torch.Tensor:
    """Compute dynamic block skip mask using BLASST/TRT-LLM threshold formula.

    The threshold formula is: threshold = scale_factor / seq_len
    A block is skipped if: local_max < running_global_max - ln(threshold)

    This implements the online softmax style decision where we track the running
    maximum and decide whether to skip based on how far the local max is from
    the current global max.

    Args:
        block_max_scores: Per-block max QK scores [num_seqs, max_num_blocks, num_heads]
        seq_lens: Sequence lengths [num_seqs]
        scale_factor: Scale factor for threshold (higher = more aggressive skipping)
        block_size: KV cache block size

    Returns:
        skip_mask: Boolean tensor [num_seqs, max_num_blocks, num_heads]
                   True = skip this block, False = compute this block
    """
    num_seqs, max_num_blocks, num_heads = block_max_scores.shape
    device = block_max_scores.device

    # Compute dynamic threshold per sequence: threshold = scale_factor / seq_len
    # ln(threshold) = ln(scale_factor) - ln(seq_len)
    seq_lens_float = seq_lens.float().clamp(min=1.0)  # Avoid div by zero
    thresholds = scale_factor / seq_lens_float  # [num_seqs]
    ln_threshold = torch.log(thresholds).view(num_seqs, 1, 1)  # [num_seqs, 1, 1]

    # Compute running max across KV blocks (simulates online softmax)
    # For each position, the running max is the max of all blocks up to that point
    # cummax returns (values, indices), we only need values
    running_max, _ = torch.cummax(block_max_scores, dim=1)  # [num_seqs, max_num_blocks, num_heads]

    # For the skip decision, we compare current block's max against the previous running max
    # Shift running_max right by 1 (pad with -inf at start)
    prev_running_max = torch.full_like(running_max, float('-inf'))
    prev_running_max[:, 1:, :] = running_max[:, :-1, :]

    # Skip condition: block_max < prev_running_max - ln(threshold)
    # This means the block's contribution to softmax is negligible
    skip_threshold = prev_running_max - ln_threshold
    skip_mask = block_max_scores < skip_threshold

    # Mask out invalid blocks (beyond sequence length)
    # Create a mask for valid blocks
    block_indices = torch.arange(max_num_blocks, device=device).view(1, -1, 1)
    num_blocks_per_seq = (seq_lens.view(-1, 1, 1) + block_size - 1) // block_size
    valid_mask = block_indices < num_blocks_per_seq

    # Only consider skip for valid blocks; invalid blocks marked as skip
    # (they won't be processed anyway)
    skip_mask = skip_mask | ~valid_mask

    # Never skip the first block (it sets the initial running max)
    skip_mask[:, 0, :] = False

    return skip_mask


def compute_block_skip_mask_with_stats(
    block_max_scores: torch.Tensor,  # [num_seqs, max_num_blocks, num_heads]
    seq_lens: torch.Tensor,  # [num_seqs]
    scale_factor: float,
    block_size: int = 16,
) -> tuple[torch.Tensor, dict]:
    """Compute block skip mask and return sparsity statistics.

    Same as compute_block_skip_mask but also returns statistics about
    how many blocks are being skipped.

    Args:
        block_max_scores: Per-block max QK scores [num_seqs, max_num_blocks, num_heads]
        seq_lens: Sequence lengths [num_seqs]
        scale_factor: Scale factor for threshold
        block_size: KV cache block size

    Returns:
        skip_mask: Boolean tensor [num_seqs, max_num_blocks, num_heads]
        stats: Dictionary with sparsity statistics
    """
    skip_mask = compute_block_skip_mask(
        block_max_scores, seq_lens, scale_factor, block_size
    )

    num_seqs, max_num_blocks, num_heads = block_max_scores.shape
    device = block_max_scores.device

    # Count valid blocks per sequence
    block_indices = torch.arange(max_num_blocks, device=device).view(1, -1, 1)
    num_blocks_per_seq = (seq_lens.view(-1, 1, 1) + block_size - 1) // block_size
    valid_mask = block_indices < num_blocks_per_seq

    # Count total valid blocks and skipped blocks
    total_valid_blocks = valid_mask.sum().item()
    skipped_blocks = (skip_mask & valid_mask).sum().item()
    computed_blocks = total_valid_blocks - skipped_blocks

    sparsity_pct = (skipped_blocks / total_valid_blocks * 100) if total_valid_blocks > 0 else 0.0

    stats = {
        'total_blocks': total_valid_blocks,
        'skipped_blocks': skipped_blocks,
        'computed_blocks': computed_blocks,
        'sparsity_pct': sparsity_pct,
        'scale_factor': scale_factor,
        'num_seqs': num_seqs,
        'num_heads': num_heads,
    }

    return skip_mask, stats


def find_contiguous_ranges(
    skip_mask: torch.Tensor,  # [max_num_blocks] boolean
) -> list[tuple[int, int]]:
    """Find contiguous ranges of non-skipped blocks.

    Args:
        skip_mask: Boolean tensor where True = skip, False = compute

    Returns:
        List of (start, end) tuples for contiguous non-skipped ranges.
        Each range is [start, end) (end is exclusive).
    """
    ranges = []
    n = skip_mask.shape[0]

    # Convert to CPU for iteration
    mask_cpu = skip_mask.cpu().numpy()

    in_range = False
    start = 0

    for i in range(n):
        if not mask_cpu[i]:  # Not skipped
            if not in_range:
                start = i
                in_range = True
        else:  # Skipped
            if in_range:
                ranges.append((start, i))
                in_range = False

    # Handle case where range extends to end
    if in_range:
        ranges.append((start, n))

    return ranges


def merge_small_ranges(
    ranges: list[tuple[int, int]],
    min_gap: int = 1,
) -> list[tuple[int, int]]:
    """Merge ranges that are separated by small gaps.

    When gaps between ranges are small, the overhead of multiple kernel
    launches may exceed the benefit of skipping those blocks. This function
    merges ranges separated by gaps smaller than min_gap.

    Args:
        ranges: List of (start, end) tuples
        min_gap: Minimum gap size to keep ranges separate

    Returns:
        Merged list of ranges
    """
    if len(ranges) <= 1:
        return ranges

    merged = [ranges[0]]

    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end

        if gap <= min_gap:
            # Merge with previous range
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged
