# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import torch
import random

from vllm.v1.sample.logits_processor.interface import (BatchUpdate,
                                                       LogitsProcessor,
                                                       MoveDirectionality)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MinPLogitsProcessor(LogitsProcessor):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=is_pin_memory)
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            # Pre-allocated device tensor
            self.min_p_device: torch.Tensor = torch.empty((max_num_reqs, ),
                                                          dtype=torch.float32,
                                                          device=device)
        else:
            self.min_p_device = self.min_p_cpu_tensor
        # Current slice of the device tensor
        self.min_p: torch.Tensor = self.min_p_device[:0]

    def is_argmax_invariant(self) -> bool:
        """Min-p never impacts greedy sampling"""
        return True

    def get_min_p_by_index(self, index: int) -> float:
        return float(self.min_p_cpu[index])

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            min_p = params.min_p
            if self.min_p_cpu[index] != min_p:
                needs_update = True
                self.min_p_cpu[index] = min_p
            if min_p:
                self.min_p_count += 1

        if self.min_p_count:
            # Process removed requests.
            needs_update |= bool(batch_update.removed)
            for index in batch_update.removed:
                if self.min_p_cpu[index]:
                    self.min_p_count -= 1

            # Process moved requests, unidirectional (a->b) and swap (a<->b)
            for adx, bdx, direct in batch_update.moved:
                change = (min_p_a :=
                          self.min_p_cpu[adx]) != (min_p_b :=
                                                   self.min_p_cpu[bdx])
                needs_update |= change
                if change:
                    self.min_p_cpu[bdx] = min_p_a
                    if direct == MoveDirectionality.SWAP:
                        self.min_p_cpu[adx] = min_p_b

        # Update tensors if needed.
        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
            self.min_p = self.min_p_device[:size]
            if self.use_double_tensor:
                self.min_p.copy_(self.min_p_cpu_tensor[:size],
                                 non_blocking=True)
            self.min_p.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values,
                                       dim=-1,
                                       keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits[invalid_token_mask] = -float('inf')
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):

    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (self._device_tensor([], torch.int32),
                             self._device_tensor([], torch.int32))

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        needs_update: bool = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            if lb := params.logit_bias:
                self.biases[index] = lb
                needs_update = True
            else:
                # Drop biases metadata at batch index
                if self.biases.pop(index, None) is not None:
                    # If a new request replaces an old request which
                    # specified biases, we should update processor tensors
                    needs_update = True

        if self.biases:
            # Process removed requests.
            for index in batch_update.removed:
                if self.biases.pop(index, None):
                    needs_update = True

            # Process moved requests, unidirectional (a->b) and swap (a<->b)
            for a_index, b_index, direct in batch_update.moved:
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if (a_entry := self.biases.pop(a_index, None)) is None:
                        if self.biases.pop(b_index, None) is not None:
                            needs_update = True
                    else:
                        self.biases[b_index] = a_entry
                        needs_update = True
                else:
                    a_entry = self.biases.pop(a_index, None)
                    if (b_entry := self.biases.pop(b_index, None)) is not None:
                        self.biases[a_index] = b_entry
                        needs_update = True
                    if a_entry is not None:
                        self.biases[b_index] = a_entry
                        needs_update = True

        # Update tensors if needed.
        if needs_update:
            reqs, tok_ids, biases = [], [], []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (self._device_tensor(reqs, torch.int32),
                                 self._device_tensor(tok_ids, torch.int32))

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        # index -> (min_toks, output_token_ids, stop_token_ids)
        self.device = device
        self.pin_memory = is_pin_memory
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}

        # (req_idx_tensor,eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor,
                                 torch.Tensor] = (self._device_tensor(
                                     [], torch.int32),
                                                  self._device_tensor(
                                                      [], torch.int32))

    def is_argmax_invariant(self) -> bool:
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        needs_update = False

        if batch_update:
            # Process added requests.
            for index, params, _, output_tok_ids in batch_update.added:
                if ((min_tokens := params.min_tokens)
                        and len(output_tok_ids) < min_tokens):
                    # Replace request metadata at batch index
                    self.min_toks[index] = (min_tokens, output_tok_ids,
                                            params.all_stop_token_ids)
                    needs_update = True
                else:
                    # Drop min_toks metadata at batch index
                    if self.min_toks.pop(index, None) is not None:
                        # If a new request replaces an old request which
                        # specified min_toks, we should update processor tensors
                        needs_update = True

            if self.min_toks:
                # Process removed requests.
                for index in batch_update.removed:
                    if self.min_toks.pop(index, None):
                        needs_update = True

                # Process moved requests, unidirectional (a->b) and
                # swapped (a<->b)
                for a_index, b_index, direct in batch_update.moved:
                    if direct == MoveDirectionality.UNIDIRECTIONAL:
                        if (a_entry := self.min_toks.pop(a_index,
                                                         None)) is None:
                            if self.min_toks.pop(b_index, None) is not None:
                                needs_update = True
                        else:
                            self.min_toks[b_index] = a_entry
                            needs_update = True
                    else:
                        a_entry = self.min_toks.pop(a_index, None)
                        if (b_entry := self.min_toks.pop(b_index,
                                                         None)) is not None:
                            self.min_toks[a_index] = b_entry
                            needs_update = True
                        if a_entry is not None:
                            self.min_toks[b_index] = a_entry
                            needs_update = True

        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            to_remove = tuple(index for index, (min_toks, out_tok_ids,
                                                _) in self.min_toks.items()
                              if len(out_tok_ids) >= min_toks)
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            for req, (_, _, stop_tok_ids) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)

            self.logits_slice = (self._device_tensor(reqs, torch.int32),
                                 self._device_tensor(tok_ids, torch.int32))

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits[self.logits_slice] = -float("inf")
        return logits


class XTCLogitsProcessor(LogitsProcessor):

    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
        # 存储每个 request 的参数 (threshold, probability, filter_value)
        self.params: dict[int, tuple[float, float, float]] = {}
        # 用于快速访问的 tensor
        self.reqs_tensor: torch.Tensor = torch.tensor(())
        self.thresholds: torch.Tensor = torch.tensor(())
        self.probabilities: torch.Tensor = torch.tensor(())
        self.filter_values: torch.Tensor = torch.tensor(())

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        needs_update: bool = False

        # 新增请求
        for index, params, _, _ in batch_update.added:
            if params.xtc_threshold is not None and params.xtc_probability is not None:
                thr = params.xtc_threshold
                prob = params.xtc_probability
                fv = -float("Inf")  # 可以固定，或者之后扩展 SamplingParams 加上 xtc_filter_value
                self.params[index] = (thr, prob, fv)
                needs_update = True
            else:
                # 如果新 request 没有 XTC 参数，移除旧的
                if self.params.pop(index, None) is not None:
                    needs_update = True

        if self.params:
            # 删除请求
            for index in batch_update.removed:
                if self.params.pop(index, None):
                    needs_update = True

            # 处理请求移动
            for a_index, b_index, direct in batch_update.moved:
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if (a_entry := self.params.pop(a_index, None)) is None:
                        if self.params.pop(b_index, None) is not None:
                            needs_update = True
                    else:
                        self.params[b_index] = a_entry
                        needs_update = True
                else:
                    a_entry = self.params.pop(a_index, None)
                    if (b_entry := self.params.pop(b_index, None)) is not None:
                        self.params[a_index] = b_entry
                        needs_update = True
                    if a_entry is not None:
                        self.params[b_index] = a_entry
                        needs_update = True

        if needs_update:
            reqs, thr, prob, fv = [], [], [], []
            for req, (t, p, f) in self.params.items():
                reqs.append(req)
                thr.append(t)
                prob.append(p)
                fv.append(f)

            self.reqs_tensor = self._device_tensor(reqs, torch.int32)
            self.thresholds = self._device_tensor(thr, torch.float32)
            self.probabilities = self._device_tensor(prob, torch.float32)
            self.filter_values = self._device_tensor(fv, torch.float32)


    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.params:
            return logits

        # 按 request 处理
        for i, req in enumerate(self.reqs_tensor):
            thr = self.thresholds[i].item()
            prob = self.probabilities[i].item()
            fv = self.filter_values[i].item()

            if random.random() >= prob:
                continue

            scores = logits[req]
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            probs = sorted_logits.softmax(dim=-1)

            sorted_indices_to_remove = torch.full_like(probs, False, dtype=torch.bool)
            sorted_indices_to_remove[..., :-1] = probs[..., 1:] >= thr

            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[req] = scores.masked_fill(indices_to_remove, fv)

        return logits
