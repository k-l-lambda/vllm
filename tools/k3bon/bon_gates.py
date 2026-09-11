#!/usr/bin/env python3
"""Cross-arm gates C and D, plus the F cross-check that needs two arms.

Run after every arm has written arm-<name>.jsonl. Each gate here compares two
arms, so it cannot live in the single-arm driver.

  GATE C  a0p (patched, C=1) vs a0u (unpatched): ids identical.
  GATE D  b0  (patched, C=2 forced-0) vs a0u: ids identical.  <- the KV gate
  GATE F' b1  (patched, C=2 forced-1) vs b0: ids must DIFFER somewhere.

C and D are same-signed and F' is opposite-signed on purpose. A patch whose
branch never executes passes C and D and fails F'. A patch that corrupts state
fails D. Neither failure is visible to the other gate, which is why a single
"is it identical" check is not enough.

Exit codes: 0 all pass, 3 a gate failed, 4 a gate could not be evaluated.
"""
import json, os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bon_replay import cmp_ids


def load(out_dir, arm):
    p = os.path.join(out_dir, f"arm-{arm}.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = {n: load(args.out, n) for n in ("a0u", "a0p", "b0", "b1")}
    for n, v in arms.items():
        print(f"[gates] arm {n}: {'absent' if v is None else str(len(v)) + ' rows'}")

    report, rc = {}, 0

    def identity_gate(name, a, b, label):
        nonlocal rc
        if arms[a] is None or arms[b] is None:
            report[name] = {"status": "SKIP -- an arm is absent"}
            print(f"[gates] {name}: SKIP (need {a} and {b})")
            return 4
        c = cmp_ids(arms[a], arms[b], a, b)
        report[name] = c
        if c["pairs"] == 0:
            print(f"[gates] {name}: CANNOT EVALUATE -- 0 shared task_ids")
            return 4
        if c["no_ids"]:
            # Comparing empty id lists reads as identical. Refuse instead.
            print(f"[gates] {name}: CANNOT EVALUATE -- {c['no_ids']} pairs "
                  f"missing token ids")
            return 4
        ok = len(c["diverged"]) == 0
        print(f"[gates] {name} ({label}): {c['identical']}/{c['pairs']} identical"
              f" -> {'PASS' if ok else 'FAIL'}")
        for d in c["diverged"][:5]:
            print(f"[gates]   diverged task={d['task_id']} at position "
                  f"{d['first_diff']}")
        return 0 if ok else 3

    r = identity_gate("GATE_C_c1_identity", "a0u", "a0p",
                      "selector compiled in, C=1, must change nothing")
    rc = max(rc, r)
    r = identity_gate("GATE_D_forced0_kv_identity", "a0u", "b0",
                      "both candidate forwards run, forced-0, must not drift")
    rc = max(rc, r)
    d_passed = (r == 0)

    # GATE F': forced-1 must differ from forced-0 somewhere, or nothing branched.
    if arms["b0"] is None or arms["b1"] is None:
        report["GATE_F_forced1_differs"] = {"status": "SKIP -- need b0 and b1"}
        print("[gates] GATE_F_forced1_differs: SKIP (need b0 and b1)")
        rc = max(rc, 4)
    else:
        c = cmp_ids(arms["b0"], arms["b1"], "b0", "b1")
        report["GATE_F_forced1_differs"] = c
        if c["pairs"] == 0 or c["no_ids"]:
            print("[gates] GATE_F_forced1_differs: CANNOT EVALUATE")
            rc = max(rc, 4)
        else:
            ok = len(c["diverged"]) > 0
            print(f"[gates] GATE_F_forced1_differs: {len(c['diverged'])}"
                  f"/{c['pairs']} rows differ -> {'PASS' if ok else 'FAIL'}")
            if ok and not d_passed:
                # b1 differing from b0 proves nothing about the branch when b0
                # itself drifted off the unpatched reference: the difference could
                # be the same corruption GATE D just caught. F is only readable
                # once forced-0 is known to reproduce.
                report["GATE_F_forced1_differs"]["caveat"] = (
                    "NOT INTERPRETABLE -- GATE D failed, so b0 is not a valid "
                    "baseline for this contrast")
                print("[gates]   NOT INTERPRETABLE: GATE D failed, so the b0 "
                      "baseline is itself suspect. Fix D before reading F.")
            if not ok:
                print("[gates]   forcing candidate 1 changed NOTHING. Either the "
                      "latch never cleared, or the two draws always collided, or "
                      "the branch never executed. Downstream arms would measure "
                      "nothing.")
            rc = max(rc, 0 if ok else 3)

    # Paired accept_len, reported not gated -- B0 vs B1 is the oracle contrast.
    if arms["b0"] and arms["b1"]:
        by0 = {r["task_id"]: r for r in arms["b0"]}
        pairs = [(by0[r["task_id"]]["accept_len"], r["accept_len"])
                 for r in arms["b1"]
                 if r["task_id"] in by0
                 and by0[r["task_id"]]["accept_len"] is not None
                 and r["accept_len"] is not None]
        if pairs:
            n = len(pairs)
            wins = sum(1 for x, y in pairs if y > x)
            ties = sum(1 for x, y in pairs if y == x)
            m0 = sum(x for x, _ in pairs) / n
            m1 = sum(y for _, y in pairs) / n
            oracle = sum(max(x, y) for x, y in pairs) / n
            report["b0_vs_b1_accept_len"] = {
                "n": n, "mean_b0": round(m0, 4), "mean_b1": round(m1, 4),
                "b1_wins": wins, "ties": ties,
                "oracle_mean_max": round(oracle, 4),
                "oracle_gain_over_b0": round(oracle - m0, 4)}
            print(f"[gates] accept_len paired n={n} b0={m0:.4f} b1={m1:.4f} "
                  f"oracle(max)={oracle:.4f} b1_wins={wins} ties={ties}")
            print("[gates]   NOTE oracle picks per row with hindsight; it is a "
                  "CEILING, not an achievable arm.")

    with open(os.path.join(args.out, "gates.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"[gates] wrote gates.json rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
