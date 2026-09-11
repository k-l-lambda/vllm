#!/usr/bin/env python3
"""Diverge-token best-of-N for K3 DSpark: one entropy-gated branch per trajectory.

Implements PLAN.md revision 2, first deliverable: the branch mechanism and the
FORCED-k policies, which are what the B0/B1 paired-trajectory arms need. The
confidence-scoring policies (best-conf / worst-conf) are NOT here -- see
"Deliberately not implemented" below.

Two edit sites, not the plan's five:

  site 1  rejection_sampler_utils.py :: rejection_sample()
          gate + draw C candidates + select + record, all inside the function
          that already owns the resample.
  site 5  states.py :: RequestState
          the per-slot latch and its reset.

Why sites 2, 3 and 4 are not needed for this deliverable
--------------------------------------------------------
The plan routed alternatives OUT of the sampler (site 2), scored them in the
speculator (site 3), and selected in model_runner (site 4). That routing exists
only to reach the draft confidence head. A forced choice needs no score, and
`rejection_sample`'s return value `sampled` is upstream of
`sampler_output.sampled_token_ids`, `postprocess_sampled` and `propose`, so
overwriting it here propagates to the KV, the detokenizer and the next draft
anchor with no further edit. Three fewer files to patch, and site 4's 4-line
vendor/upstream delta stops mattering.

Why no Triton kernel is modified
--------------------------------
`_resample_kernel`'s draw is keyed by `(seed, pos)` through
`gumbel_block_argmax(..., seed_ptr, pos_ptr, ...)`, and `seed` is a
`[max_num_reqs]` tensor argument. An i.i.d. candidate is therefore just another
launch with a different seed tensor. `_rejection_kernel` runs ONCE, so
`num_sampled`, `cu_num_logits`, `target_rejected_logsumexp`,
`draft_rejected_logsumexp` and `cumulative_log_p` are all frozen and shared --
which is what makes the intervention exactly one token wide.

The mutation that would have made this silently wrong
-----------------------------------------------------
`_insert_resampled_kernel` increments its input in place:
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)      # line 835
so a second call sees an already-incremented `num_sampled` and writes to
`start_idx + num_sampled + 1` -- one slot PAST the resample position. Every
extra candidate is therefore given a fresh CLONE of the pre-increment
`num_sampled`, and the clone's increment is discarded. GATE C below is what
catches this if it is ever reintroduced.

Selection is fully vectorised on GPU (mask + torch.where) so no `.item()` /
CPU sync is added to the decode loop. Recording costs one small D2H copy per
call and is off unless K3BON_DUMP is set.

Deliberately not implemented
----------------------------
* best-conf / worst-conf selection. Needs the draft confidence head, i.e. the
  plan's site 3 plus the site 2 routing. Run B0/B1 first: if first-block
  confidence does not rank two realised branches, those arms are not worth GPU
  time, and that verdict costs two forced runs.
* Residual-side gate quantities. The candidates are drawn from the residual but
  the gate reads the TARGET nucleus, as specified. A target-side gate can fire
  where the residual is degenerate, so the number of DISTINCT candidates drawn
  is recorded per branch point -- the empirical ground truth for "did a branch
  exist here", free because the candidates are drawn anyway.

Idempotent, anchor-guarded, rank-0 only for recording.
"""

import re
import sys

MARK = "K3BON_PATCH_APPLIED"


def _anchor(src, anchor, what, path):
    """Locate a unique FULL-LINE anchor. Refuses a substring or a repeat."""
    lines = src.split("\n")
    hits = [i for i, ln in enumerate(lines) if ln == anchor]
    if len(hits) != 1:
        raise SystemExit(
            f"{path}: anchor for {what} matched {len(hits)} full lines, need "
            f"exactly 1.\n  anchor: {anchor!r}"
        )
    return hits[0]


HELPERS = '''

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
'''


def patch_sampler_utils(path: str) -> str:
    src = open(path).read()
    if MARK in src:
        return "rejection_sampler_utils: already patched"

    lines = src.split("\n")

    # --- imports -------------------------------------------------------------
    if "\nimport atexit\n" not in src:
        src = src.replace("\nimport torch\n",
                          "\nimport atexit\nimport os\n\nimport torch\n", 1)
    elif "\nimport os\n" not in src:
        src = src.replace("\nimport torch\n", "\nimport os\n\nimport torch\n", 1)
    if "init_logger" not in src:
        src = src.replace("\nimport torch\n",
                          "\nimport torch\n\nfrom vllm.logger import init_logger\n", 1)
    if "logger = init_logger(__name__)" not in src:
        src = src.replace("\n@triton.jit\n",
                          "\nlogger = init_logger(__name__)\n\n\n@triton.jit\n", 1)

    # --- helpers, appended after the module body ----------------------------
    src = src.rstrip("\n") + "\n" + HELPERS

    # --- site 1: capture the pre-increment num_sampled and branch -----------
    lines = src.split("\n")
    ins_anchor = "    # Insert the resampled tokens into the output sampled."
    i = _anchor(src, ins_anchor, "insert-resampled comment", path)

    pre = (
        "    # K3BON: snapshot num_sampled BEFORE _insert_resampled_kernel\n"
        "    # increments it in place (line ~835: num_sampled + 1). Every extra\n"
        "    # candidate is fed a clone of THIS tensor.\n"
        "    _k3bon_num_sampled_pre = num_sampled.clone()\n"
    )
    lines.insert(i, pre.rstrip("\n"))
    src = "\n".join(lines)

    # --- site 1b: branch just before the return -----------------------------
    ret_anchor = "    return sampled, num_sampled"
    i = _anchor(src, ret_anchor, "rejection_sample return", path)
    lines = src.split("\n")
    branch = '''    # K3BON: one entropy-gated branch per trajectory. Inert unless K3BON_C > 1
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
    # ''' + MARK + "\n"
    lines.insert(i, branch.rstrip("\n"))
    src = "\n".join(lines)

    open(path, "w").write(src)
    return "rejection_sampler_utils: patched (site 1)"


def patch_states(path: str) -> str:
    src = open(path).read()
    if MARK in src:
        return "states: already patched"

    anchor = "        self.draft_tokens[req_idx].zero_()"
    i = _anchor(src, anchor, "add_request slot init", path)
    lines = src.split("\n")
    add = (
        "\n        # " + MARK + ": clear the diverge-token branch latch for this\n"
        "        # slot. free_indices recycles slots, so at concurrency 1\n"
        "        # consecutive requests share one -- without this only the first\n"
        "        # request of each slot would ever branch, and every later row\n"
        "        # would read as 'no diverge token found'.\n"
        "        try:\n"
        "            from vllm.v1.worker.gpu.spec_decode."
        "rejection_sampler_utils import (\n"
        "                k3bon_reset_slot,\n"
        "            )\n"
        "\n"
        "            k3bon_reset_slot(req_idx)\n"
        "        except Exception:\n"
        "            pass"
    )
    lines.insert(i + 1, add)
    src = "\n".join(lines)
    open(path, "w").write(src)
    return "states: patched (site 5)"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: patch_k3_bestof.py <rejection_sampler_utils.py> <states.py>"
        )
    print(patch_sampler_utils(sys.argv[1]))
    print(patch_states(sys.argv[2]))
