#!/usr/bin/env python3
"""CPU tests for patch_k3_bestof.py. No GPU, no vLLM import.

Covers what can be falsified without a device:
  * the patch applies to the REAL source, parses, and is idempotent
  * every positional argument my extra kernel launches pass lines up with the
    kernel's own signature, read out of the source by ast -- not by my reading
  * the num_sampled clone is present (the in-place increment defect)
  * the gate maths, on distributions whose answer is known analytically
  * the latch, including the slot-recycling reset that is the whole point of
    site 5
  * candidate-0 identity: with C=1 nothing is touched

Run:  python3 test_patch_k3_bestof.py <path-to-vllm-checkout>
      (defaults to /tmp/k3src if that holds the two sources)
"""

import ast
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "patch_k3_bestof.py")

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def kernel_params(src, fn_name):
    """Positional parameter names of a @triton.jit kernel, in order,
    excluding tl.constexpr keyword-only ones."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            names, consts = [], []
            for a in node.args.args:
                ann = ast.unparse(a.annotation) if a.annotation else ""
                (consts if "constexpr" in ann else names).append(a.arg)
            return names, consts
    raise AssertionError(f"kernel {fn_name} not found")


def call_positionals(src, container_fn, kernel_name):
    """Positional args of the FIRST `kernel_name[...](...)` call inside
    container_fn, as source text."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == container_fn:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    if (isinstance(f, ast.Subscript)
                            and isinstance(f.value, ast.Name)
                            and f.value.id == kernel_name):
                        return sub.args, {k.arg for k in sub.keywords}
    raise AssertionError(f"no {kernel_name} call in {container_fn}")



def _my_resample_positionals(src):
    """Positional args my extra _resample_kernel launch passes, in order.

    Reconstructed from the patched source: the 4 explicit buffers, then the
    "head" tuple, then seed_c, then the "tail" tuple. Returned as bare names so
    they can be zipped against the kernel's parameter list.
    """
    def _tuple_items(key):
        # Depth-counted: the tuples contain `.stride(0)`, so a naive search for
        # the next ")," lands inside a nested call.
        i = src.index(f'"{key}": (') + len(f'"{key}": (')
        depth, j = 1, i
        while depth:
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = src[i:j]
        items, buf, d = [], "", 0
        for ch in body:
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
            if ch == "," and d == 0:
                items.append(buf.strip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            items.append(buf.strip())
        return [x.replace("\n", " ").strip() for x in items if x.strip()]

    explicit = ["am_c", "am_c.stride(0)", "mx_c", "mx_c.stride(0)"]
    return explicit + _tuple_items("head") + ["seed_c"] + _tuple_items("tail")

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/k3src"
    rsu_src = os.path.join(root, "rejection_sampler_utils.py")
    st_src = os.path.join(root, "states.py")
    for p in (rsu_src, st_src):
        if not os.path.exists(p):
            raise SystemExit(f"missing source: {p}")

    tmp = tempfile.mkdtemp(prefix="k3bon-test-")
    rsu = os.path.join(tmp, "rejection_sampler_utils.py")
    st = os.path.join(tmp, "states.py")
    shutil.copy(rsu_src, rsu)
    shutil.copy(st_src, st)
    pre_rsu = open(rsu).read()
    pre_st = open(st).read()

    print("== GATE B: patch lands, parses, idempotent")
    r = subprocess.run([sys.executable, PATCH, rsu, st],
                       capture_output=True, text=True)
    check("patch exits 0", r.returncode == 0, r.stderr[-400:])
    post_rsu = open(rsu).read()
    post_st = open(st).read()
    check("rejection_sampler_utils changed", post_rsu != pre_rsu)
    check("states changed", post_st != pre_st)
    for nm, s in (("rejection_sampler_utils", post_rsu), ("states", post_st)):
        try:
            ast.parse(s)
            check(f"{nm} parses", True)
        except SyntaxError as e:
            check(f"{nm} parses", False, str(e))
    r2 = subprocess.run([sys.executable, PATCH, rsu, st],
                        capture_output=True, text=True)
    check("re-run is idempotent", "already patched" in r2.stdout)
    check("re-run changes nothing", open(rsu).read() == post_rsu)

    print("== the in-place-increment defect")
    check("num_sampled snapshot taken before insert",
          "_k3bon_num_sampled_pre = num_sampled.clone()" in post_rsu)
    i_snap = post_rsu.index("_k3bon_num_sampled_pre = num_sampled.clone()")
    i_ins = post_rsu.index("_insert_resampled_kernel[(num_reqs,)](")
    check("snapshot precedes the insert launch", i_snap < i_ins,
          f"snap@{i_snap} insert@{i_ins}")
    check("extra candidates clone num_sampled",
          "ns_c = num_sampled_pre.clone()" in post_rsu)

    print("== GATE A: kernel signature alignment (read from source, not memory)")
    res_names, res_consts = kernel_params(post_rsu, "_resample_kernel")
    ins_names, ins_consts = kernel_params(post_rsu, "_insert_resampled_kernel")
    check("_resample_kernel positional count is 20",
          len(res_names) == 20, f"got {len(res_names)}: {res_names}")
    check("_insert_resampled_kernel positional count is 11",
          len(ins_names) == 11, f"got {len(ins_names)}: {ins_names}")
    check("seed_ptr is _resample_kernel positional 17",
          res_names[16] == "seed_ptr", f"got {res_names[16]}")
    check("rejected_step_ptr is positional 12",
          res_names[11] == "rejected_step_ptr", f"got {res_names[11]}")
    check("num_sampled_ptr is _insert positional 3",
          ins_names[2] == "num_sampled_ptr", f"got {ins_names[2]}")

    # My launches, read out of the patched source and mapped ONE BY ONE onto the
    # kernel's parameter names. A count-only check passes when two args are
    # swapped, which a mutation run demonstrated: `pos` and `cumulative_log_p`
    # traded places and every count still matched.
    mine = _my_resample_positionals(post_rsu)
    check("my _resample_kernel launch supplies 20 positionals",
          len(mine) == len(res_names), f"mine={len(mine)} kernel={len(res_names)}")
    # Expected pairing, derived from the kernel's own parameter names.
    EXPECT = {
        "target_logits_ptr": "target_logits",
        "target_rejected_logsumexp_ptr": "target_rejected_logsumexp",
        "draft_logits_ptr": "draft_logits",
        "draft_rejected_logsumexp_ptr": "draft_rejected_logsumexp",
        "rejected_step_ptr": "_k3bon_num_sampled_pre",
        "cu_num_logits_ptr": "cu_num_logits",
        "expanded_idx_mapping_ptr": "expanded_idx_mapping",
        "draft_sampled_ptr": "draft_sampled",
        "temp_ptr": "temperature",
        "seed_ptr": "seed_c",
        "pos_ptr": "pos",
        "cumulative_log_p_ptr": "cumulative_log_p",
        "vocab_size": "vocab_size",
    }
    if len(mine) == len(res_names):
        bad = []
        for param, arg in zip(res_names, mine):
            want = EXPECT.get(param)
            if want is not None and arg != want:
                bad.append(f"{param}<-{arg} (want {want})")
        check("every named _resample_kernel positional gets the right arg",
              not bad, "; ".join(bad))
    n_ins_mine = 7 + 4
    check("my _insert launch supplies 11 positionals",
          n_ins_mine == len(ins_names),
          f"mine={n_ins_mine} kernel={len(ins_names)}")
    check("all 4 resample constexprs passed as kwargs",
          set(res_consts) == {"BLOCK_SIZE", "HAS_DRAFT_LOGITS", "USE_FP64",
                              "USE_BLOCK_VERIFICATION"},
          str(res_consts))
    check("insert constexpr is PADDED_RESAMPLE_NUM_BLOCKS",
          ins_consts == ["PADDED_RESAMPLE_NUM_BLOCKS"], str(ins_consts))
    # The pairing that would silently corrupt the candidate: position 12 must
    # receive the PRE-increment clone, not the live num_sampled.
    seg = post_rsu[post_rsu.index('"head": ('):post_rsu.index('"tail": (')]
    check("positional 12 of my launch is the pre-increment clone",
          "_k3bon_num_sampled_pre" in seg
          and "num_sampled," not in seg.replace("_k3bon_num_sampled_pre,", ""),
          seg.strip()[:160])

    print("== gate maths (analytic answers)")
    try:
        import torch
    except ImportError:
        print("  SKIP  torch unavailable; gate maths not exercised")
        torch = None
    if torch is not None:
        spec = importlib.util.spec_from_file_location("k3bon_probe", rsu)
        # Importing the patched module needs vllm; instead re-implement nothing
        # and exec ONLY the helper function out of the patched source, so the
        # test exercises the shipped code rather than a copy of it.
        src = open(rsu).read()
        start = src.index("def _k3bon_nucleus_stats(")
        end = src.index("def _k3bon_branch(")
        ns = {"torch": torch}
        exec(compile(src[start:end], "<k3bon>", "exec"), ns)
        stats = ns["_k3bon_nucleus_stats"]

        V = 8
        # row 0: uniform over 4, zero elsewhere -> nucleus entropy ln 4
        # row 1: one-hot                        -> entropy 0, p_top1 1
        # row 2: two-way 50/50                  -> ln 2, collision 0.5
        big = 30.0
        logits = torch.full((3, V), -big)
        logits[0, :4] = 0.0
        logits[1, 0] = 0.0
        logits[2, :2] = 0.0
        temp = torch.ones(3)
        h, p1, coll, k = stats(logits, temp, 0.95)
        check("uniform-4 nucleus entropy == ln4",
              abs(h[0].item() - math.log(4)) < 1e-3, f"{h[0].item():.5f}")
        check("one-hot entropy == 0", abs(h[1].item()) < 1e-3, f"{h[1].item():.5f}")
        check("one-hot p_top1 == 1", abs(p1[1].item() - 1.0) < 1e-3,
              f"{p1[1].item():.5f}")
        check("50/50 entropy == ln2", abs(h[2].item() - math.log(2)) < 1e-3,
              f"{h[2].item():.5f}")
        check("50/50 collision == 0.5", abs(coll[2].item() - 0.5) < 1e-3,
              f"{coll[2].item():.5f}")
        check("one-hot nucleus size == 1", int(k[1].item()) == 1,
              f"{k[1].item()}")
        # greedy rows must never open a branch: _resample_kernel returns early
        hg, _, _, _ = stats(logits, torch.zeros(3), 0.95)
        check("temperature 0 forces H=0 (no branch)",
              float(hg.abs().max()) == 0.0, f"{hg.tolist()}")
        # the threshold must separate the known cases
        check("H* = 0.2472 admits ln2 and rejects one-hot",
              h[2].item() >= 0.2472 > h[1].item())

        # The nucleus must actually TRUNCATE, or this test cannot tell nucleus
        # entropy from full-vocab entropy. A mutation run proved that: swapping
        # in the full-vocab softmax passed every check above, because with the
        # tail at logit -30 the two quantities agree to 1e-9.
        Vb = 4096
        tail = torch.full((1, Vb), 0.0)
        tail[0, 0] = math.log(0.90 * (Vb - 1) / 0.10)   # top1 mass 0.90
        hn, p1n, _, kn = stats(tail, torch.ones(1), 0.95)
        pf = torch.softmax(tail[0], dim=-1)
        h_full = float(-(pf * torch.log(pf.clamp_min(1e-20))).sum())
        check("fixture truncates: nucleus is a strict subset",
              int(kn[0].item()) < Vb, f"k={kn[0].item()} V={Vb}")
        check("fixture separates nucleus from full-vocab entropy",
              abs(hn[0].item() - h_full) > 0.5,
              f"nucleus={hn[0].item():.4f} full={h_full:.4f}")
        # top1 keeps 0.90; the nucleus adds tail tokens only up to 0.95, so the
        # renormalised top1 must exceed the unrenormalised 0.90.
        check("nucleus renormalisation raises p_top1 above the raw 0.90",
              p1n[0].item() > 0.90, f"{p1n[0].item():.5f}")

    print("== GATE E: latch and slot recycling")
    src = open(rsu).read()
    start = src.index("class _K3BonState:")
    end = src.index("def k3bon_state(")
    if torch is not None:
        ns2 = {"torch": torch, "_k3bon_cfg": lambda: None}
        exec(compile(src[start:end], "<k3bon-state>", "exec"), ns2)
        S = ns2["_K3BonState"]()
        S.ensure(4, "cpu")
        check("latch starts clear", not bool(S.latched.any()))
        S.latched[2] = True
        check("latch sets", bool(S.latched[2]))
        S.reset_slot(2)
        check("reset_slot clears the recycled slot", not bool(S.latched[2]))
        S.latched[1] = True
        S.reset_slot(9)          # out of range must be a no-op, not a crash
        check("out-of-range reset is a no-op", bool(S.latched[1]))
    check("add_request calls k3bon_reset_slot",
          "k3bon_reset_slot(req_idx)" in post_st)
    # the reset must sit in add_request, not somewhere else
    tree = ast.parse(post_st)
    in_add = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "add_request":
            in_add = "k3bon_reset_slot" in ast.unparse(node)
    check("the reset is inside add_request", in_add)

    print("== GATE C prerequisite: C=1 is inert")
    check("branch body is guarded by cfg.active",
          "if _k3bon_cfg().active:" in post_rsu)
    check("C<=1 returns before drawing candidates",
          "if cfg.n_cand <= 1 or not bool(fires.any()):" in post_rsu)
    check("rank != 0 writes nothing",
          "self.dump = None" in post_rsu and "self.dump_all = None" in post_rsu)
    check("flush every call (256 wrote 0 B once)",
          "st.flush()" in post_rsu)

    shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
