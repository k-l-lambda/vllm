#!/usr/bin/env python3
"""Mutation harness for test_patch_k3_bestof.py.

Breaks patch_k3_bestof.py in known ways and requires the suite to catch each
one. The `intact` control MUST score PASS: without it a harness that fails for
an unrelated reason would report every mutation as CAUGHT and look perfect.

This harness earned its place. On its first run the suite scored 38/38 green
while MISSING two real defects:
  * `pos` and `cumulative_log_p` swapped in the kernel launch -- the alignment
    check counted positionals but never mapped them, so any permutation passed.
  * full-vocab entropy substituted for nucleus entropy -- every gate fixture
    put the whole support inside the nucleus, so the two quantities agreed to
    1e-9 and the test could not tell them apart.
Both are now caught. A green suite is not evidence until it has been shown able
to fail.

Usage:  python3 mutate_patch_k3_bestof.py [<source-dir>]     (default /tmp/k3src)
Restores patch_k3_bestof.py on exit, including on failure.
"""

import os
import subprocess
import sys

D = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(D, "patch_k3_bestof.py")
TEST = os.path.join(D, "test_patch_k3_bestof.py")

HEAD_LINE = ("                        expanded_idx_mapping, draft_sampled, "
             "temperature,")

MUTATIONS = [
    ("intact (control)", None, None),
    # the in-place increment of num_sampled
    ("drop the num_sampled clone",
     "        ns_c = num_sampled_pre.clone()",
     "        ns_c = num_sampled_pre"),
    ("delete the pre-increment snapshot",
     '        "    _k3bon_num_sampled_pre = num_sampled.clone()\\n"',
     '        ""'),
    # kernel argument order
    ("swap pos <-> cumulative_log_p in the tail",
     '                    "tail": (pos, cumulative_log_p, vocab_size),',
     '                    "tail": (cumulative_log_p, pos, vocab_size),'),
    ("drop one head arg (19 positionals)",
     HEAD_LINE,
     "                        expanded_idx_mapping, draft_sampled,"),
    ("swap draft_sampled <-> temperature",
     HEAD_LINE,
     "                        expanded_idx_mapping, temperature, draft_sampled,"),
    # gate maths
    ("full-vocab entropy instead of nucleus",
     "    h = -(q * torch.log(q.clamp_min(1e-20))).sum(dim=-1)",
     "    h = -(probs * torch.log(probs.clamp_min(1e-20))).sum(dim=-1)"),
    ("greedy rows not forced to H=0",
     "    h = torch.where(greedy, torch.zeros_like(h), h)",
     "    h = h"),
    # the latch
    ("remove the reset from add_request",
     "            k3bon_reset_slot(req_idx)",
     "            pass  # reset removed"),
    ("reset_slot ignores its index",
     "        if self.latched is not None and 0 <= idx < self.latched.numel():\n"
     "            self.latched[idx] = False",
     "        if self.latched is not None and 0 <= idx < self.latched.numel():\n"
     "            pass"),
    # inertness at C=1
    ("remove the C<=1 early return",
     "    if cfg.n_cand <= 1 or not bool(fires.any()):",
     "    if False:"),
]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/k3src"
    orig = open(PATCH).read()
    missed, skipped, broken = [], [], []
    print(f"{'mutation':<46} {'verdict':<8} first failing check")
    print("-" * 104)
    try:
        for name, old, new in MUTATIONS:
            src = orig
            if old is not None:
                n = src.count(old)
                if n != 1:
                    skipped.append(f"{name} (anchor count={n})")
                    print(f"{name:<46} {'SKIP':<8} anchor count={n}")
                    continue
                src = src.replace(old, new)
            open(PATCH, "w").write(src)
            r = subprocess.run([sys.executable, TEST, root],
                               capture_output=True, text=True)
            fails = [l.strip() for l in r.stdout.split("\n")
                     if l.strip().startswith("FAIL")]
            if old is None:
                verdict = "PASS" if r.returncode == 0 else "BROKEN"
                if verdict == "BROKEN":
                    broken.append(name)
            else:
                verdict = "CAUGHT" if r.returncode != 0 else "MISSED"
                if verdict == "MISSED":
                    missed.append(name)
            if fails:
                first = fails[0][:60]
            elif r.returncode == 0:
                first = "(suite clean)"
            else:
                first = r.stderr.strip().split("\n")[-1][:60]
            print(f"{name:<46} {verdict:<8} {first}")
    finally:
        open(PATCH, "w").write(orig)
    print("-" * 104)
    print(f"patch restored ({len(orig)} bytes)")
    if broken:
        print("HARNESS INVALID: the intact control did not pass; every other "
              "verdict here is meaningless")
        return 2
    if skipped:
        print(f"SKIPPED {len(skipped)} (anchor drift, mutation not exercised):")
        for s in skipped:
            print(f"  {s}")
    if missed:
        print(f"MISSED {len(missed)}:")
        for m in missed:
            print(f"  {m}")
        return 1
    print(f"all {len(MUTATIONS) - 1} mutations caught; intact control passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
