# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import os

import torch

from vllm.logger import init_logger

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax, tl_rand32


logger = init_logger(__name__)


@triton.jit
def _compute_max_and_sumexp(logits):
    max = tl.max(logits, axis=0)
    sumexp = tl.where(
        max > float("-inf"),
        tl.sum(tl.exp(logits - max)),
        0.0,
    )
    return max, sumexp


@triton.jit
def _compute_global_logsumexp(
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    maxes = tl.load(
        local_max_ptr + logit_idx * local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    sumexps = tl.load(
        local_sumexp_ptr + logit_idx * local_sumexp_stride + blocks,
        mask=blocks_mask,
        other=0.0,
    )
    global_max = tl.max(maxes, axis=0)
    global_lse = global_max + tl.log(tl.sum(sumexps * tl.exp(maxes - global_max)))
    return global_lse


@triton.jit
def _compute_global_residual_mass(
    local_residual_mass_ptr,
    local_residual_mass_stride,
    prefix_joint_ratio,
    target_logits_ptr,
    target_logits_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_sampled_ptr,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    if HAS_DRAFT_LOGITS:
        blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
        mask = blocks < vocab_num_blocks
        partials = tl.load(
            local_residual_mass_ptr + logit_idx * local_residual_mass_stride + blocks,
            mask=mask,
            other=0.0,
        )
        return tl.sum(partials, axis=0)
    else:
        # One-hot draft. M_s is a point mass at this draft token
        # so the residual mass reduces to the closed form:
        #   p * (1 - M_b(draft_token)).
        draft_token = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        target_lse = _compute_global_logsumexp(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            logit_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        target_logit = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + draft_token,
        ).to(tl.float32)
        m_b = tl.exp(target_logit - target_lse)
        return prefix_joint_ratio * (1.0 - m_b)


@triton.jit
def _compute_global_target_argmax(
    target_local_max_ptr,
    target_local_max_stride,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    local_max = tl.load(
        target_local_max_ptr + logit_idx * target_local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    max_block_idx = tl.argmax(local_max, axis=0)
    return tl.load(
        target_local_argmax_ptr + logit_idx * target_local_argmax_stride + max_block_idx
    ).to(tl.int64)


@triton.jit
def _compute_global_logprobs_and_logsumexp(
    token,
    mask,
    logit_idx,
    req_state_idx,
    draft_step,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    target_logit = tl.load(
        target_logits_ptr + logit_idx * target_logits_stride + token,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    target_lse = _compute_global_logsumexp(
        target_local_max_ptr,
        target_local_max_stride,
        target_local_sumexp_ptr,
        target_local_sumexp_stride,
        logit_idx,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
    )
    target_log_prob = target_logit - target_lse
    if HAS_DRAFT_LOGITS:
        draft_logit = tl.load(
            draft_logits_ptr
            + req_state_idx * draft_logits_stride_0
            + draft_step * draft_logits_stride_1
            + token,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        draft_lse = _compute_global_logsumexp(
            draft_local_max_ptr,
            draft_local_max_stride,
            draft_local_sumexp_ptr,
            draft_local_sumexp_stride,
            logit_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        draft_log_prob = draft_logit - draft_lse
    else:
        # One-hot draft: q(token) = 1, log_q = 0.
        draft_log_prob = 0.0
        draft_lse = 0.0
    return target_log_prob, draft_log_prob, target_lse, draft_lse


@triton.jit
def _compute_local_logits_stats_kernel(
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)

    if draft_step_idx >= num_speculative_steps:
        # Bonus token. Max/argmax and summed exponentials are not needed.
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if temp == 0.0:
        # Greedy sampling. Only the target max/argmax are needed.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(
            target_local_argmax_ptr
            + logit_idx * target_local_argmax_stride
            + block_idx,
            token_id,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            value,
        )
    else:
        # Get local target max and summed exponentials.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        target_max, target_sumexp = _compute_max_and_sumexp(target_logits)
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            target_max,
        )
        tl.store(
            target_local_sumexp_ptr
            + logit_idx * target_local_sumexp_stride
            + block_idx,
            target_sumexp,
        )
        if HAS_DRAFT_LOGITS:
            # Get local draft max and summed exponentials.
            draft_logits = tl.load(
                draft_logits_ptr
                + req_state_idx * draft_logits_stride_0
                + draft_step_idx * draft_logits_stride_1
                + block_offsets,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            draft_max, draft_sumexp = _compute_max_and_sumexp(draft_logits)
            tl.store(
                draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
                draft_max,
            )
            tl.store(
                draft_local_sumexp_ptr
                + logit_idx * draft_local_sumexp_stride
                + block_idx,
                draft_sumexp,
            )


@triton.jit
def _compute_cumulative_log_p_kernel(
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp == 0.0:
        return

    log_p = 0.0
    for step in range(num_draft_tokens):
        logit_idx = start_idx + step
        draft_token = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        target_logprob, draft_logprob, _, _ = _compute_global_logprobs_and_logsumexp(
            draft_token,
            True,  # mask
            logit_idx,
            req_state_idx,
            step,
            target_logits_ptr,
            target_logits_stride,
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            draft_logits_ptr,
            draft_logits_stride_0,
            draft_logits_stride_1,
            draft_local_max_ptr,
            draft_local_max_stride,
            draft_local_sumexp_ptr,
            draft_local_sumexp_stride,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
            HAS_DRAFT_LOGITS,
        )
        log_p = tl.minimum(log_p + (target_logprob - draft_logprob), 0.0)
        tl.store(cumulative_log_p_ptr + logit_idx, log_p)


@triton.jit
def _compute_local_residual_mass_kernel(
    # [num_logits, num_blocks]
    local_residual_mass_ptr,
    local_residual_mass_stride,
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_speculative_steps,
    vocab_num_blocks,
    BLOCK_SIZE: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    if draft_step_idx == 0 or draft_step_idx >= num_speculative_steps:
        # The acceptance threshold, h, looks one position ahead and sums
        # over: max(p_i * M_b(x|x_{<i}) - M_s(x|x_{<i}), 0). Tokens at the
        # first and last (bonus) positions aren't needed for this computation.
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp == 0.0:
        return

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size
    target_log_probs, draft_log_probs, _, _ = _compute_global_logprobs_and_logsumexp(
        block_offsets,
        mask,
        logit_idx,
        req_state_idx,
        draft_step_idx,
        target_logits_ptr,
        target_logits_stride,
        target_local_max_ptr,
        target_local_max_stride,
        target_local_sumexp_ptr,
        target_local_sumexp_stride,
        draft_logits_ptr,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_local_max_ptr,
        draft_local_max_stride,
        draft_local_sumexp_ptr,
        draft_local_sumexp_stride,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
        True,  # HAS_DRAFT_LOGITS
    )

    # Compute the residual mass: max(p_i * M_b(x|x_{<i}) - M_s(x|x_{<i}), 0)
    p = tl.exp(tl.load(cumulative_log_p_ptr + logit_idx - 1).to(tl.float32))
    m_b = tl.exp(target_log_probs)
    m_s = tl.exp(draft_log_probs)
    partial = tl.sum(tl.maximum(p * m_b - m_s, 0.0), axis=0)
    tl.store(
        local_residual_mass_ptr + logit_idx * local_residual_mass_stride + block_idx,
        partial,
    )


@triton.jit
def _rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [num_speculative_steps]
    synthetic_conditional_rates_ptr,
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, num_blocks]
    local_residual_mass_ptr,
    local_residual_mass_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0

    accepted_length = tl.zeros((), tl.int64)
    target_lse = 0.0
    draft_lse = 0.0
    accepted = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        pos = tl.load(pos_ptr + logit_idx)
        u = tl_rand32(seed, pos, includes_zero=False)
        if USE_BLOCK_VERIFICATION and not is_greedy:
            # Block verification (Sun et al., 2024): https://arxiv.org/abs/2403.10444
            prefix_joint_ratio = tl.exp(
                tl.load(cumulative_log_p_ptr + logit_idx).to(tl.float32)
            )
            if i < num_draft_tokens - 1:
                residual_mass = _compute_global_residual_mass(
                    local_residual_mass_ptr,
                    local_residual_mass_stride,
                    prefix_joint_ratio,
                    target_logits_ptr,
                    target_logits_stride,
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    draft_sampled_ptr,
                    logit_idx + 1,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                    HAS_DRAFT_LOGITS,
                )
                denom = residual_mass + 1.0 - prefix_joint_ratio
                h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
            else:
                h = prefix_joint_ratio
            accepted_length = tl.where(u <= h, i + 1, accepted_length)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
        elif accepted:
            if is_greedy:
                # Greedy sampling. Accept IFF draft matches target argmax.
                # NOTE: Target argmax is stored directly so that resampling
                # can be skipped upon rejection.
                target_argmax = _compute_global_target_argmax(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_argmax_ptr,
                    target_local_argmax_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    # -1 is used for padded draft token ids that should be rejected.
                    accepted &= (u < rate) & (draft_sampled >= 0)
                else:
                    accepted &= target_argmax == draft_sampled
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    draft_sampled if accepted else target_argmax,
                )
            else:
                # Speculative decoding (Leviathan et al., 2023): https://arxiv.org/abs/2211.17192
                # -1 is used for padded draft token ids that should be rejected.
                is_valid_draft = draft_sampled >= 0
                # Avoid possible OOB ptr access.
                draft_sampled = tl.maximum(0, draft_sampled)
                target_logprob, draft_logprob, target_lse, draft_lse = (
                    _compute_global_logprobs_and_logsumexp(
                        draft_sampled,
                        True,  # mask
                        logit_idx,
                        req_state_idx,
                        i,
                        target_logits_ptr,
                        target_logits_stride,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        draft_logits_ptr,
                        draft_logits_stride_0,
                        draft_logits_stride_1,
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sumexp_ptr,
                        draft_local_sumexp_stride,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        HAS_DRAFT_LOGITS,
                    )
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted &= u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted &= target_logprob > tl.log(u) + draft_logprob
                accepted &= is_valid_draft
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            accepted_length += accepted
    tl.store(rejected_steps_ptr + req_idx, accepted_length)
    if USE_BLOCK_VERIFICATION and not is_greedy and accepted_length < num_draft_tokens:
        # Compute the target and draft log exponential sums for the
        # rejected token.
        rejected_idx = start_idx + accepted_length
        target_lse = _compute_global_logsumexp(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            rejected_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        if HAS_DRAFT_LOGITS:
            draft_lse = _compute_global_logsumexp(
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sumexp_ptr,
                draft_local_sumexp_stride,
                rejected_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


@triton.jit
def _resample_kernel(
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_reqs]
    rejected_step_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [num_logits]
    cumulative_log_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx).to(tl.int64)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. No resampling needed because
        # the target argmax is already in the sampled tensor.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    # Compute the residual logits to resample the rejected token from.
    if is_bonus:
        # Bonus token (no rejections). Directly use the target logits.
        residual_logits = target_logits
    elif HAS_DRAFT_LOGITS:
        draft_logits = tl.load(
            draft_logits_ptr
            + req_state_idx * draft_logits_stride_0
            + resample_idx * draft_logits_stride_1
            + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        target_lse = tl.load(target_rejected_logsumexp_ptr + req_idx)
        draft_lse = tl.load(draft_rejected_logsumexp_ptr + req_idx)
        target_log_probs = target_logits - target_lse
        if USE_BLOCK_VERIFICATION:
            # Block residual is:
            #   max(p_tau * M_b(x) - M_s(x), 0) / Z.
            # Scale the target logprobs by log(p_tau). p_0 = 1, so skip
            # shifting when nothing was accepted (tau == 0).
            log_p_tau = 0.0
            if resample_idx > 0:
                log_p_tau = tl.load(cumulative_log_p_ptr + resample_token_idx - 1).to(
                    tl.float32
                )
            target_log_probs += log_p_tau
        draft_log_probs = draft_logits - draft_lse
        # Compute the residual:
        #   r(x) = max(p(x) - q(x), 0)
        # Gumbel sampling needs logits, so we compute it in log space:
        #   log(r(x)) = log(max(exp(log_p(x)) - exp(log_q(x)), 0))
        # The more numerically stable form is:
        #   log(max(exp(a) - exp(b), 0)) = a + log(max(1 - exp(b - a), 0))
        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            ratio < 1.0,
            target_log_probs + tldevice.log1p(-ratio),
            float("-inf"),
        ).to(tl.float32)
    else:
        # One-hot draft. The residual is just the target distribution with
        # the rejected draft token probability zeroed out.
        # NOTE: During block verification, the residual becomes:
        #   0                   if x == rejected_draft_token
        #   p_tau * M_b(x) / Z  otherwise
        # Therefore p_tau is a constant that cancels under normalization,
        # and does not need to be applied.
        rejected_draft_token = tl.load(draft_sampled_ptr + resample_token_idx + 1)
        residual_logits = tl.where(
            block != rejected_draft_token,
            target_logits,
            float("-inf"),
        ).to(tl.float32)

    # Resample the rejected/bonus token.
    value, idx = gumbel_block_argmax(
        residual_logits,
        block,
        mask,
        resample_token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        None,  # processed_logits_ptr
        0,  # processed_logits_stride
        None,  # processed_logits_col_ptr
        vocab_size,
        APPLY_TEMPERATURE=False,
        USE_FP64=USE_FP64,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + block_idx,
        token_id,
    )
    tl.store(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_idx,
        value,
    )


@triton.jit
def _insert_resampled_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    resample_num_blocks,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + num_sampled
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    # Increment the number of sampled tokens.
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. The target argmax is already
        # in the sampled tensor.
        return

    # Insert the resampled token.
    block = tl.arange(0, PADDED_RESAMPLE_NUM_BLOCKS)
    mask = block < resample_num_blocks
    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)
    resampled = tl.load(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + resampled_max_block_idx,
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + num_sampled,
        resampled,
    )


def rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64: bool = False,
    use_block_verification: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert target_logits.ndim == 2 and target_logits.stride(-1) == 1
    assert draft_logits is None or (
        draft_logits.ndim == 3 and draft_logits.stride(-1) == 1
    )
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    draft_logits_stride_0 = 0
    draft_logits_stride_1 = 0
    if has_draft_logits := draft_logits is not None:
        draft_logits_stride_0 = draft_logits.stride(0)
        draft_logits_stride_1 = draft_logits.stride(1)
        # In some cases (e.g. MiMo v2.5 Pro + DFlash) the target model's
        # vocab size is larger than the draft's due to padding.
        vocab_size = min(vocab_size, draft_logits.size(-1))

    # Compute the per-vocab-block logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    target_local_argmax = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.int64
    )
    target_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    target_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    _compute_local_logits_stats_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
    )

    # Precompute the running joint ratio and residual mass for block
    # verification.
    if use_block_verification:
        assert synthetic_conditional_rates is None, (
            "Block verification is incompatible with synthetic acceptance rates."
        )

        # Compute the log of the running joint ratio, p_i.
        # cumulative_log_p[start + i] = log(p_{i+1}), the cumulative ratio after
        # the (i+1)-th draft token.
        cumulative_log_p = target_logits.new_empty(num_logits, dtype=torch.float32)
        _compute_cumulative_log_p_kernel[(num_reqs,)](
            cumulative_log_p,
            target_logits,
            target_logits.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits_stride_0,
            draft_logits_stride_1,
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=has_draft_logits,
            num_warps=1,
        )

        # Compute the per-vocab-block partials of the residual mass, later reduced
        # to the total by _compute_global_residual_mass. Only launched for full
        # draft logits distributions. One-hot drafts used a closed-form residual
        # mass instead.
        if has_draft_logits:
            local_residual_mass = target_logits.new_empty(
                num_logits, vocab_num_blocks, dtype=torch.float32
            )
            _compute_local_residual_mass_kernel[(num_logits, vocab_num_blocks)](
                local_residual_mass,
                local_residual_mass.stride(0),
                cumulative_log_p,
                target_logits,
                target_logits.stride(0),
                target_local_max,
                target_local_max.stride(0),
                target_local_sumexp,
                target_local_sumexp.stride(0),
                draft_logits,
                draft_logits_stride_0,
                draft_logits_stride_1,
                draft_local_max,
                draft_local_max.stride(0),
                draft_local_sumexp,
                draft_local_sumexp.stride(0),
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                vocab_size,
                num_speculative_steps,
                vocab_num_blocks,
                BLOCK_SIZE=VOCAB_BLOCK_SIZE,
                PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            )
        else:
            local_residual_mass = None
    else:
        cumulative_log_p = None
        local_residual_mass = None

    # Sample up until the first rejected/bonus token, and store
    # the step.
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    _rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        synthetic_conditional_rates,
        cumulative_log_p,
        local_residual_mass,
        local_residual_mass.stride(0) if local_residual_mass is not None else 0,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        USE_BLOCK_VERIFICATION=use_block_verification,
        num_warps=1,
    )

    # Resample the rejected/bonus tokens.
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.int64
    )
    resampled_local_max = target_logits.new_empty(
        num_reqs,
        resample_num_blocks,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    _resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_rejected_logsumexp,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        cumulative_log_p,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
    )

    # K3BON: snapshot num_sampled BEFORE _insert_resampled_kernel
    # increments it in place (line ~835: num_sampled + 1). Every extra
    # candidate is fed a clone of THIS tensor.
    _k3bon_num_sampled_pre = num_sampled.clone()
    # Insert the resampled tokens into the output sampled.
    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
    )
    # K3BON: one entropy-gated branch per trajectory. Inert unless K3BON_C > 1
    # or K3BON_DUMP_ALL is set. Overwriting `sampled` here is upstream of
    # sampler_output.sampled_token_ids, postprocess_sampled and propose, so the
    # winner reaches the KV, the detokenizer and the next draft anchor with no
    # other edit.
    if _k3bon_cfg().active:
        _k3bon_branch(
            sampled,
            _k3bon_num_sampled_pre,
            (
                resampled_local_argmax,
                resampled_local_max,
                {
                    "num_blocks": resample_num_blocks,
                    "head": (
                        target_logits, target_logits.stride(0),
                        target_rejected_logsumexp,
                        draft_logits, draft_logits_stride_0,
                        draft_logits_stride_1, draft_rejected_logsumexp,
                        _k3bon_num_sampled_pre, cu_num_logits,
                        expanded_idx_mapping, draft_sampled, temperature,
                    ),
                    "tail": (pos, cumulative_log_p, vocab_size),
                    "kw": dict(
                        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
                        HAS_DRAFT_LOGITS=has_draft_logits,
                        USE_FP64=use_fp64,
                        USE_BLOCK_VERIFICATION=use_block_verification,
                    ),
                },
            ),
            {
                "args": (
                    resample_num_blocks, cu_num_logits,
                    expanded_idx_mapping, temperature,
                ),
                "kw": dict(
                    PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks
                ),
            },
            target_logits,
            cu_num_logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            num_reqs,
        )
    # K3BON_PATCH_APPLIED
    return sampled, num_sampled


# ---------------------------------------------------------------- K3BON begin
# Diverge-token best-of-N: one entropy-gated branch per trajectory.
# See memo/benchmark-reports/diverge-bestof/patch_k3_bestof.py

class _K3BonConfig:
    """Read once. An unset K3BON_C leaves every code path below inert."""

    def __init__(self):
        self.n_cand = int(os.environ.get("K3BON_C", "1"))
        # Gate quantity as specified: entropy of the renormalised nucleus, nats.
        self.h_thresh = float(os.environ.get("K3BON_H", "0.2472"))
        self.top_p = float(os.environ.get("K3BON_TOP_P", "0.95"))
        # engine | forced-0 .. forced-(C-1) | random
        self.policy = os.environ.get("K3BON_POLICY", "engine")
        self.dump = os.environ.get("K3BON_DUMP") or None
        # Also record the gate at EVERY resample position, not just the branch.
        # Measurement-only: needed for the gate-rate sanity check, and lets the
        # threshold be re-cut offline from a single run.
        self.dump_all = os.environ.get("K3BON_DUMP_ALL") or None
        self.seed_stride = int(os.environ.get("K3BON_SEED_STRIDE", "1000003"))
        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
            )

            self.tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            self.tp_rank = 0
        if self.tp_rank != 0:
            # One writer only. 8 TP ranks appending to one path is the defect
            # this project has already paid for twice.
            self.dump = None
            self.dump_all = None
        self.forced_k = -1
        if self.policy.startswith("forced-"):
            self.forced_k = int(self.policy.split("-", 1)[1])
            if not 0 <= self.forced_k < self.n_cand:
                raise ValueError(
                    f"K3BON_POLICY={self.policy} needs 0 <= k < C={self.n_cand}"
                )

    @property
    def active(self):
        return self.n_cand > 1 or self.dump_all is not None


_K3BON_CFG = None
_K3BON_STATE = None


def _k3bon_cfg():
    global _K3BON_CFG
    if _K3BON_CFG is None:
        _K3BON_CFG = _K3BonConfig()
        logger.info(
            "K3BON C=%d policy=%s H*=%.4f top_p=%.3f dump=%s dump_all=%s "
            "(active=%s)",
            _K3BON_CFG.n_cand, _K3BON_CFG.policy, _K3BON_CFG.h_thresh,
            _K3BON_CFG.top_p, _K3BON_CFG.dump, _K3BON_CFG.dump_all,
            _K3BON_CFG.active,
        )
    return _K3BON_CFG


class _K3BonState:
    """Per-slot latch, plus the record buffers.

    The latch is a GPU bool over max_num_reqs so selection needs no CPU sync.
    It is sized lazily from the first temperature tensor seen, which is
    [max_num_reqs], and it is cleared per slot by RequestState.add_request --
    see k3bon_reset_slot. `req_state_idx` is a RECYCLED slot, not a request id:
    at concurrency 1 consecutive requests reuse the same slot, so without that
    reset only the first request of each slot would ever branch.
    """

    N_FIELDS = 12

    def __init__(self):
        self.latched = None
        self.calls = 0
        self.buf = []
        self.rows = 0
        self.written = 0
        self.buf_all = []
        self.rows_all = 0
        self.written_all = 0

    def ensure(self, max_num_reqs, device):
        if self.latched is None or self.latched.numel() < max_num_reqs:
            self.latched = torch.zeros(
                max_num_reqs, dtype=torch.bool, device=device
            )
        return self.latched

    def reset_slot(self, idx):
        if self.latched is not None and 0 <= idx < self.latched.numel():
            self.latched[idx] = False

    def flush(self):
        cfg = _k3bon_cfg()
        if cfg.dump and self.buf:
            with open(cfg.dump, "ab") as f:
                for row in self.buf:
                    f.write(row)
            self.written += len(self.buf)
            self.buf = []
        if cfg.dump_all and self.buf_all:
            with open(cfg.dump_all, "ab") as f:
                for row in self.buf_all:
                    f.write(row)
            self.written_all += len(self.buf_all)
            self.buf_all = []


def k3bon_state():
    global _K3BON_STATE
    if _K3BON_STATE is None:
        _K3BON_STATE = _K3BonState()
        atexit.register(_K3BON_STATE.flush)
    return _K3BON_STATE


def k3bon_reset_slot(idx):
    """Called from RequestState.add_request when a slot is claimed."""
    if _K3BON_STATE is not None:
        _K3BON_STATE.reset_slot(idx)


def _k3bon_nucleus_stats(logits_row, temperature_row, top_p):
    """Nucleus stats of the TARGET distribution at the branch candidates' row.

    Returns (H, p_top1, collision, k_nucleus), each [num_reqs], float32.
    `H` is the entropy of the nucleus AFTER renormalisation, in nats: the
    server samples at top_p, so full-vocab entropy answers a question the
    sampler never asks. Greedy rows (temperature 0) get H=0 so they can never
    open a branch -- consistent with _resample_kernel, which returns early at
    temp == 0.0 for non-bonus tokens.
    """
    t = temperature_row.to(torch.float32).clamp_min(1e-6).unsqueeze(-1)
    probs = torch.softmax(logits_row.to(torch.float32) / t, dim=-1)
    srt, _ = torch.sort(probs, dim=-1, descending=True)
    csum = torch.cumsum(srt, dim=-1)
    # Keep the smallest prefix whose mass >= top_p (the token that crosses the
    # threshold is inside the nucleus, matching the usual convention).
    inside = (csum - srt) < top_p
    k_nucleus = inside.sum(dim=-1).to(torch.float32)
    kept = torch.where(inside, srt, torch.zeros_like(srt))
    mass = kept.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    q = kept / mass
    h = -(q * torch.log(q.clamp_min(1e-20))).sum(dim=-1)
    p_top1 = q[:, 0]
    collision = 1.0 - (q * q).sum(dim=-1)
    greedy = temperature_row.to(torch.float32) <= 0.0
    h = torch.where(greedy, torch.zeros_like(h), h)
    return h, p_top1, collision, k_nucleus


def _k3bon_branch(
    sampled,
    num_sampled_pre,
    resample_args,
    insert_args,
    target_logits,
    cu_num_logits,
    expanded_idx_mapping,
    temperature,
    seed,
    pos,
    num_reqs,
):
    """Draw C candidates for the resample token, gate, select, record.

    `sampled` is modified in place at the resample position of the requests
    whose gate fires and whose latch is clear. Returns nothing.

    `num_sampled_pre` MUST be the value _rejection_kernel wrote, i.e. before
    _insert_resampled_kernel incremented it.
    """
    cfg = _k3bon_cfg()
    st = k3bon_state()
    st.calls += 1

    dev = sampled.device
    latched = st.ensure(temperature.shape[0], dev)

    # Row of target_logits the resample reads: start + rejected_step.
    # Both _resample_kernel (resample_token_idx = start_idx + resample_idx) and
    # _insert_resampled_kernel (start_idx + num_sampled) index it this way.
    start = cu_num_logits[:num_reqs].to(torch.int64)
    resample_row = start + num_sampled_pre[:num_reqs].to(torch.int64)
    req_state_idx = expanded_idx_mapping[resample_row].to(torch.int64)

    temp_row = temperature[req_state_idx]
    h, p_top1, collision, k_nuc = _k3bon_nucleus_stats(
        target_logits[resample_row], temp_row, cfg.top_p
    )

    gate = h >= cfg.h_thresh
    fires = gate & (~latched[req_state_idx])

    branch_pos = pos[resample_row].to(torch.float32)
    cand0 = sampled[torch.arange(num_reqs, device=dev), num_sampled_pre[:num_reqs].to(torch.int64)]

    if cfg.dump_all:
        st.buf_all.append(
            torch.stack([
                torch.full_like(h, float(st.calls)),
                req_state_idx.to(torch.float32),
                branch_pos,
                h, p_top1, collision, k_nuc,
                gate.to(torch.float32),
                latched[req_state_idx].to(torch.float32),
            ], dim=-1).to(torch.float32).cpu().numpy().tobytes()
        )
        st.rows_all += num_reqs

    if cfg.n_cand <= 1 or not bool(fires.any()):
        # Nothing to branch. The `.any()` sync is the price of not drawing
        # candidates for a batch that cannot use them; with C=1 it is skipped
        # entirely so the identity arm adds no sync at all.
        if cfg.dump_all:
            st.flush()
        return

    # --- draw the extra candidates -------------------------------------------
    (res_argmax, res_max, r_rest) = resample_args
    ins_rest = insert_args
    cands = [cand0]
    for c in range(1, cfg.n_cand):
        # A fresh seed tensor is the whole intervention: gumbel_block_argmax
        # keys on tl.randint(seed, pos), so a different seed is an i.i.d. draw
        # from the SAME residual distribution.
        seed_c = seed + c * cfg.seed_stride
        am_c = torch.empty_like(res_argmax)
        mx_c = torch.empty_like(res_max)
        _resample_kernel[(num_reqs, r_rest["num_blocks"])](
            am_c, am_c.stride(0), mx_c, mx_c.stride(0),
            *r_rest["head"], seed_c, *r_rest["tail"],
            **r_rest["kw"],
        )
        # Cloned num_sampled: _insert_resampled_kernel increments it in place.
        ns_c = num_sampled_pre.clone()
        s_c = sampled.clone()
        _insert_resampled_kernel[(num_reqs,)](
            s_c, s_c.stride(0), ns_c,
            am_c, am_c.stride(0), mx_c, mx_c.stride(0),
            *ins_rest["args"], **ins_rest["kw"],
        )
        cands.append(
            s_c[torch.arange(num_reqs, device=dev),
                num_sampled_pre[:num_reqs].to(torch.int64)]
        )

    stack = torch.stack(cands, dim=0)                       # [C, num_reqs]
    # Distinct candidate count per request: candidate i counts iff no earlier j
    # drew the same token. This is the empirical answer to "did a branch exist
    # here" -- the gate reads the TARGET nucleus, but the candidates come from
    # the residual, so a firing gate does not guarantee two distinct draws.
    eq = stack.unsqueeze(1) == stack.unsqueeze(0)            # [C, C, num_reqs]
    earlier = torch.tril(
        torch.ones(cfg.n_cand, cfg.n_cand, dtype=torch.bool, device=dev),
        diagonal=-1,
    ).unsqueeze(-1)
    is_first = ~(eq & earlier).any(dim=1)                    # [C, num_reqs]
    n_distinct = is_first.sum(dim=0).to(torch.float32)

    if cfg.forced_k >= 0:
        k = torch.full((num_reqs,), cfg.forced_k, dtype=torch.int64, device=dev)
    elif cfg.policy == "random":
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed[req_state_idx][0].item()) + st.calls)
        k = torch.randint(0, cfg.n_cand, (num_reqs,), generator=g).to(dev)
    else:
        k = torch.zeros(num_reqs, dtype=torch.int64, device=dev)

    k = torch.where(fires, k, torch.zeros_like(k))
    chosen = stack.gather(0, k.unsqueeze(0)).squeeze(0)

    rows = torch.arange(num_reqs, device=dev)
    cols = num_sampled_pre[:num_reqs].to(torch.int64)
    sampled[rows, cols] = torch.where(fires, chosen, cand0)

    # Latch the requests that just branched.
    latched[req_state_idx] = latched[req_state_idx] | fires

    if cfg.dump:
        st.buf.append(
            torch.stack([
                torch.full_like(h, float(st.calls)),
                req_state_idx.to(torch.float32),
                branch_pos,
                h, p_top1, collision, k_nuc,
                torch.full_like(h, float(cfg.n_cand)),
                n_distinct,
                k.to(torch.float32),
                cand0.to(torch.float32),
                chosen.to(torch.float32),
            ], dim=-1)[fires].to(torch.float32).cpu().numpy().tobytes()
        )
        st.rows += int(fires.sum().item())
    # Flush every call. A threshold of 256 wrote 0 B in this project once
    # already: a short sweep makes ~100 verify calls and SIGTERM skips atexit.
    st.flush()
# ------------------------------------------------------------------ K3BON end
