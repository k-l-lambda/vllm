#!/usr/bin/env python3
"""Driver for the diverge-token best-of-N arms: run rows, capture ids, gate.

WHY THIS EXISTS SEPARATELY FROM ent_replay.py. That driver reconstructs per-slot
confidence outcomes and carries a large apparatus for it. This run needs one thing
that driver does not provide: the SERVER-RETURNED token ids, because gates C and D
are bit-identity claims. Re-tokenising the text locally cannot support them -- a
join of returned tokens is not the streamed text, and this project has already
published a wrong number from exactly that reconstruction.

`return_token_ids` exists in this vLLM (chat_completion/protocol.py:390) and the
streaming path attaches the ids to the stream CHOICE, not the delta
(chat_completion/serving.py:679-690). There is a trap: serving.py:641-645 sets
`hide_stream_metadata = (not request.include_reasoning and parser is not None)`
and that suppresses logprobs AND token_ids. K3 is a reasoning model with a parser,
so `include_reasoning` must be TRUE or the identity gates silently read nothing.
`--gate-min-ids` refuses that case instead of scoring it.

WHAT EACH GATE CAN REJECT, and why none of them is circular:

  C  patched C=1 vs UNPATCHED, same rows/seed: ids identical. The selector compiled
     in but never branching must change nothing.
  D  patched C=2 forced-0 vs UNPATCHED: ids identical. Both candidate forwards run,
     so a losing candidate that contaminated engine state drifts the tokens here.
     This is the KV gate and it is the reason C is not sufficient.
  E  latch/reset over sequential requests. Read from the dump_all stream, which is
     the ONLY one carrying gate-fired-but-latched: the main dump is filtered by
     [fires], so comparing it against itself is a tautology. Needs K3BON_DUMP_ALL.
  F  a branch must EXIST: n_distinct >= 2 at >= 1 branch point, and forced-1 must
     differ from forced-0 somewhere. A firing gate does not imply two distinct
     draws -- the gate reads the target nucleus, the candidates come from the
     residual. If they always collide, every downstream arm measures nothing.
  G  gate rate against r7's prior (9.3% of positions diverge, 81.4% degenerate).

Note D and F are deliberately opposite-signed: D demands forced-0 change nothing,
F demands forced-1 change something. A patch that silently no-ops passes D and
fails F, which is the failure a single identity gate cannot see.
"""
import argparse, json, os, struct, sys, time, urllib.request

# Main dump: 12 float32 fields (see _K3BonState.N_FIELDS).
BON_FIELDS = ["call", "slot", "pos", "h", "p_top1", "collision", "k_nuc",
              "n_cand", "n_distinct", "k", "cand0", "chosen"]
# dump_all: 9 float32 fields (N_FIELDS_ALL) -- a DIFFERENT width. Reading one with
# the other's stride silently yields garbage that still parses, so both widths are
# named here and checked against the file size before any row is interpreted.
ALL_FIELDS = ["call", "slot", "pos", "h", "p_top1", "collision", "k_nuc",
              "gate", "latched"]
BON_ROW = 4 * len(BON_FIELDS)
ALL_ROW = 4 * len(ALL_FIELDS)


def log(msg):
    print(f"[bon] {msg}", flush=True)


def read_dump(path, fields, row_bytes):
    """Read a float32 dump. Refuses a size that is not a whole number of rows."""
    if not path or not os.path.exists(path):
        return None, "absent"
    n = os.path.getsize(path)
    if n == 0:
        return [], "empty"
    if n % row_bytes:
        return None, f"size {n} not a multiple of {row_bytes}"
    out = []
    with open(path, "rb") as fh:
        blob = fh.read()
    for i in range(0, n, row_bytes):
        vals = struct.unpack("<%df" % len(fields), blob[i:i + row_bytes])
        out.append(dict(zip(fields, vals)))
    # A size check alone does not catch a stride mismatch: 4 dump_all rows (144 B)
    # divide evenly by the main 48 B stride and would read as 3 plausible rows.
    # These columns are booleans and n_cand is a small positive int, so a wrong
    # stride lands float garbage in them. Checked before any row is interpreted.
    for col in ("gate", "latched"):
        if col in fields:
            bad = [r[col] for r in out if r[col] not in (0.0, 1.0)]
            if bad:
                return None, (f"{col} carries non-boolean values "
                              f"(e.g. {bad[0]!r}) -- wrong row width for this file")
    if "n_cand" in fields:
        bad = [r["n_cand"] for r in out
               if not (1.0 <= r["n_cand"] <= 64.0) or r["n_cand"] != int(r["n_cand"])]
        if bad:
            return None, (f"n_cand carries implausible values (e.g. {bad[0]!r}) "
                          f"-- wrong row width for this file")
    return out, "ok"


def stream(base, model, messages, max_tokens, temp, top_p, seed, want_ids=True,
           timeout=3600):
    """One streaming request. Returns server-returned ids, not a re-tokenisation."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temp, "stream": True,
            "stream_options": {"include_usage": True}}
    if top_p is not None:
        body["top_p"] = top_p
    if seed is not None:
        body["seed"] = seed
    if want_ids:
        body["return_token_ids"] = True
        # MANDATORY with return_token_ids on a reasoning model: without it
        # hide_stream_metadata drops both logprobs and token_ids and the identity
        # gates would score zero ids as a pass.
        body["include_reasoning"] = True
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    ids, text, usage, finish = [], [], {}, None
    t0 = time.time()
    ttft = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            ch = (ev.get("choices") or [None])[0]
            if not ch:
                continue
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            # The ids live on the CHOICE, not the delta.
            tid = ch.get("token_ids")
            if tid:
                ids.extend(tid)
            d = ch.get("delta") or {}
            piece = (d.get("content") or "") or (d.get("reasoning_content")
                                                 or d.get("reasoning") or "")
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                text.append(piece)
    return {"ids": ids, "text": "".join(text), "finish": finish,
            "completion_tokens": usage.get("completion_tokens"),
            "wall": time.time() - t0, "ttft": ttft}


def run_arm(bench, base, model, rows, max_tokens, temp, top_p, seed, want_ids,
            arm, out_dir):
    """Run every row once. accept_len comes from the engine's own counters."""
    res = []
    for i, row in enumerate(rows):
        b = bench.snapshot(base) if bench else None
        t = stream(base, model, row["messages"], max_tokens, temp, top_p, seed,
                   want_ids=want_ids)
        a = bench.snapshot(base) if bench else None
        d = bench.delta(b, a) if bench else {}
        rec = {"arm": arm, "task_id": row.get("task_id"), "seed": seed,
               "ids": t["ids"], "n_ids": len(t["ids"]),
               "completion_tokens": t["completion_tokens"],
               "finish": t["finish"], "wall": round(t["wall"], 3),
               "ttft": round(t["ttft"], 3) if t["ttft"] else None,
               "accept_len": d.get("accept_len"),
               "accepted": d.get("accepted"), "drafts": d.get("drafts"),
               "text_sha": __import__("hashlib").sha256(
                   t["text"].encode()).hexdigest()[:16]}
        res.append(rec)
        log(f"{arm} row {i+1}/{len(rows)} task={rec['task_id']} "
            f"ids={rec['n_ids']} ct={rec['completion_tokens']} "
            f"al={rec['accept_len']} wall={rec['wall']}s")
    with open(os.path.join(out_dir, f"arm-{arm}.jsonl"), "w") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    return res


def cmp_ids(a_rows, b_rows, label_a, label_b):
    """Positionwise id comparison. Reports the FIRST divergence, not just a count."""
    out = {"pairs": 0, "identical": 0, "diverged": [], "no_ids": 0}
    by_a = {r["task_id"]: r for r in a_rows}
    for rb in b_rows:
        ra = by_a.get(rb["task_id"])
        if ra is None:
            continue
        out["pairs"] += 1
        if not ra["ids"] or not rb["ids"]:
            out["no_ids"] += 1
            continue
        if ra["ids"] == rb["ids"]:
            out["identical"] += 1
        else:
            n = min(len(ra["ids"]), len(rb["ids"]))
            first = next((i for i in range(n) if ra["ids"][i] != rb["ids"][i]), n)
            out["diverged"].append({
                "task_id": rb["task_id"], "first_diff": first,
                f"n_{label_a}": len(ra["ids"]), f"n_{label_b}": len(rb["ids"]),
                f"{label_a}_at": ra["ids"][first] if first < len(ra["ids"]) else None,
                f"{label_b}_at": rb["ids"][first] if first < len(rb["ids"]) else None})
    return out


def gate_e(all_rows):
    """Latch/reset: per slot, requests that branched vs whose gate fired.

    Read from dump_all because it is the only stream with gate-fired-but-latched.
    A stuck latch reads as "gate fired 20 times, branched once" on one slot, which
    is the DEFAULT failure at concurrency 1 where every request reuses one slot.
    """
    if all_rows is None:
        return {"status": "no dump_all -- GATE E cannot be evaluated"}
    fired = sum(1 for r in all_rows if r["gate"] >= 0.5)
    fired_free = sum(1 for r in all_rows if r["gate"] >= 0.5 and r["latched"] < 0.5)
    fired_latched = fired - fired_free
    slots = {}
    for r in all_rows:
        s = int(r["slot"])
        d = slots.setdefault(s, {"rows": 0, "fired": 0, "fired_latched": 0})
        d["rows"] += 1
        if r["gate"] >= 0.5:
            d["fired"] += 1
            if r["latched"] >= 0.5:
                d["fired_latched"] += 1
    return {"status": "ok", "rows": len(all_rows), "gate_fired": fired,
            "fired_while_unlatched": fired_free,
            "fired_while_latched": fired_latched,
            "distinct_slots": len(slots),
            "per_slot": {str(k): v for k, v in sorted(slots.items())}}


def gate_f(bon_rows):
    """A branch must exist: >=1 branch point with two distinct candidates."""
    if not bon_rows:
        return {"status": "no branch rows recorded"}
    n = len(bon_rows)
    dist2 = sum(1 for r in bon_rows if r["n_distinct"] >= 2)
    chose_alt = sum(1 for r in bon_rows if r["chosen"] != r["cand0"])
    return {"status": "ok", "branch_rows": n,
            "with_2plus_distinct": dist2,
            "frac_2plus_distinct": round(dist2 / n, 4),
            "chosen_differs_from_cand0": chose_alt,
            "mean_n_distinct": round(sum(r["n_distinct"] for r in bon_rows) / n, 4)}


def gate_g(all_rows, bon_rows):
    """Gate rate against r7's prior: 9.3% of positions diverge, 81.4% degenerate."""
    src = all_rows if all_rows else bon_rows
    if not src:
        return {"status": "no rows"}
    n = len(src)
    out = {"status": "ok", "rows": n}
    if all_rows:
        fired = sum(1 for r in src if r["gate"] >= 0.5)
        out["gate_rate"] = round(fired / n, 4)
        out["prior_r7_diverge_rate"] = 0.093
    hs = sorted(r["h"] for r in src)
    ps = sorted(r["p_top1"] for r in src)
    out["h_median"] = round(hs[n // 2], 4)
    out["p_top1_median"] = round(ps[n // 2], 4)
    out["degenerate_frac_p_top1_gt_0.9964"] = round(
        sum(1 for r in src if r["p_top1"] > 0.9964) / n, 4)
    pos = sorted(int(r["pos"]) for r in src)
    out["branch_pos_median"] = pos[n // 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="kimi-k3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True,
                    help="a0u | a0p | b0 | b1 (names the output file)")
    ap.add_argument("--bench", default=None, help="path to bench_k3h_block5.py")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-ids", action="store_true")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--dump-all", default=None)
    ap.add_argument("--gate-min-ids", type=int, default=1,
                    help="refuse if fewer than this many rows returned ids")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    log(f"arm={a.arm} rows={len(rows)} max_tokens={a.max_tokens} seed={a.seed}")

    bench = None
    if a.bench:
        import importlib.util
        spec = importlib.util.spec_from_file_location("bench", a.bench)
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)

    res = run_arm(bench, a.base, a.model, rows, a.max_tokens, a.temperature,
                  a.top_p, a.seed, not a.no_ids, a.arm, a.out)

    with_ids = sum(1 for r in res if r["ids"])
    log(f"arm={a.arm} rows_with_ids={with_ids}/{len(res)}")
    if not a.no_ids and with_ids < a.gate_min_ids:
        # An identity gate scored on zero ids is the failure mode this refuses:
        # hide_stream_metadata drops token_ids silently and every comparison
        # would read "identical" over empty lists.
        log(f"FATAL only {with_ids} rows returned token ids; identity gates "
            f"would be vacuous. Check include_reasoning/return_token_ids.")
        return 4

    summary = {"arm": a.arm, "rows": len(res), "rows_with_ids": with_ids,
               "seed": a.seed, "max_tokens": a.max_tokens}
    bon, st_bon = read_dump(a.dump, BON_FIELDS, BON_ROW)
    allr, st_all = read_dump(a.dump_all, ALL_FIELDS, ALL_ROW)
    log(f"dump={st_bon} rows={len(bon) if bon is not None else 'NA'}  "
        f"dump_all={st_all} rows={len(allr) if allr is not None else 'NA'}")
    if bon is None and a.dump:
        log(f"WARNING main dump unreadable: {st_bon}")
    if allr is None and a.dump_all:
        log(f"WARNING dump_all unreadable: {st_all}")
    summary["dump_status"] = st_bon
    summary["dump_all_status"] = st_all
    summary["gate_E_latch"] = gate_e(allr)
    summary["gate_F_branch_exists"] = gate_f(bon or [])
    summary["gate_G_rate"] = gate_g(allr, bon or [])
    with open(os.path.join(a.out, f"summary-{a.arm}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    log(f"wrote summary-{a.arm}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
