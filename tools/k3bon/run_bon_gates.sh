#!/bin/bash
# exp65 -- diverge-token best-of-N: the GATES, before any pilot.
#
# WHAT THIS RUN IS FOR. It does NOT measure the conjecture. It answers whether the
# patch is safe to measure with, and it can reject in five distinct ways. The 100-row
# B0/B1 pilot costs ~15 h of eager GPU time; running it before these gates would risk
# spending all of it on a patch that silently no-ops or corrupts state.
#
# FOUR ARMS, FOUR SERVERS. _k3bon_cfg() caches at first call, so K3BON_C and
# K3BON_POLICY cannot be flipped inside a live process. Each arm is its own launch:
#   a0u  UNPATCHED           the reference. Run FIRST, before the patch touches anything.
#   a0p  patched, C=1        GATE C: selector compiled in, never branches, changes nothing.
#   b0   patched, C=2 k=0    GATE D: both candidate forwards run, forced-0, must not drift.
#   b1   patched, C=2 k=1    GATE F: must differ from b0 somewhere, or nothing branched.
#
# GATES C/D AND F ARE OPPOSITE-SIGNED ON PURPOSE. C and D demand "changes nothing";
# F demands "changes something". A patch whose branch never executes passes C and D
# and fails F. A single identity check cannot see that failure.
#
# --enforce-eager IS MANDATORY. Under full CUDA graphs the draft step is replayed
# (dflash/speculator.py:453 -> run_fullgraph) so patched Python executes only at
# capture. Every arm here is eager, including a0u, so a0u is the only legal TPOT
# reference in this family -- b5base/seed5/twoseed were captured and are NOT comparable.
#
# TEMP MUST BE > 0. _resample_kernel returns early at temp == 0.0 for non-bonus
# tokens, so a greedy arm has no branch point at all and every gate would be vacuous.
set -u
OUT=${OUT_DIR:-/data/output/results-bon-gates}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1
echo "[bon] node=$(hostname) start=$(date -u +%FT%TZ) generation=${RUN_GENERATION:-unset}"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

VERIFIER=${VERIFIER:-/data/models/moonshotai/Kimi-K3}
D_B5=${D_B5:-/data/models/Inferact/Kimi-K3-DSpark-Block5}
B5_BYTES_EXPECT=5707153178
B5_MD5_EXPECT=e09c570535c14504e6cd0ecf3247dd81

# ---- preflight: the draft must be the one the 0.8340 AUC was measured on ------
# The published (1, 7424) head's calibration does NOT transfer to this (1, 7680)
# head -- different markov_rank, different norm split, and the 256K case reversed
# sign in that family. Byte size and md5 are both checked: size alone would accept
# a same-length different checkpoint.
[ -s "$D_B5/config.json" ] || { echo "[bon] FATAL missing $D_B5/config.json"; exit 1; }
BB=$(stat -c %s "$D_B5/model.safetensors" 2>/dev/null || echo 0)
[ "$BB" -eq "$B5_BYTES_EXPECT" ] || { echo "[bon] FATAL Block5 draft $BB != $B5_BYTES_EXPECT"; exit 1; }
echo "[bon] Block5 draft size verified: $BB B"
if [ "${SKIP_MD5:-0}" != "1" ]; then
  M=$(md5sum "$D_B5/model.safetensors" | cut -d' ' -f1)
  [ "$M" = "$B5_MD5_EXPECT" ] || { echo "[bon] FATAL Block5 md5 $M != $B5_MD5_EXPECT"; exit 1; }
  echo "[bon] Block5 md5 verified: $M"
fi
CFG_W=$(python3 -c "import json;print(json.load(open('$D_B5/config.json')).get('block_size'))")
[ "$CFG_W" = "${N_B5:-5}" ] || { echo "[bon] FATAL block_size=$CFG_W but N=${N_B5:-5}"; exit 1; }
echo "[bon] width verified: block_size=$CFG_W == N=${N_B5:-5}"
case "${TEMP:-1.0}" in
  0|0.0|0.00) echo "[bon] FATAL TEMP=${TEMP} -- _resample_kernel returns early at temp==0"
              echo "[bon]   for non-bonus tokens, so there is no branch point to gate."
              exit 1;;
esac
echo "[bon] temperature ${TEMP:-1.0} > 0, branch points exist"

# ---- GATE 1: the confidence head must be TRAINED and the SHAPE we think ------
# Verbatim from run_ent_replay.sh. r7 measured 0.8340 on THIS head; a random-init
# projection still emits numbers in [0,1] that look like probabilities.
echo "[bon] GATE 1: confidence head identity in $D_B5"
python3 - "$D_B5" <<'PYG1'
import json, struct, sys, math
d = sys.argv[1]
p = f"{d}/model.safetensors"
with open(p, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
base = 8 + n
hdr.pop("__metadata__", None)
cf = [k for k in hdr if "confidence" in k]
if not cf:
    print("[gate1] FATAL no confidence_head tensor in the checkpoint"); sys.exit(1)
cfg = json.load(open(f"{d}/config.json"))
want_in = cfg["hidden_size"] + (cfg.get("markov_rank", 0) if cfg.get("confidence_head_with_markov") else 0)
W = hdr["confidence_head.proj.weight"]
if list(W["shape"]) != [1, want_in]:
    print(f"[gate1] FATAL proj.weight {W['shape']} != [1, {want_in}] from config"); sys.exit(1)
def bf16(buf):
    return [struct.unpack("<f", struct.pack("<I", struct.unpack("<H", buf[i:i+2])[0] << 16))[0]
            for i in range(0, len(buf), 2)]
def rd(t):
    a, b = base + t["data_offsets"][0], base + t["data_offsets"][1]
    with open(p, "rb") as fh:
        fh.seek(a); return bf16(fh.read(b - a))
w = rd(W); bias = rd(hdr["confidence_head.proj.bias"])[0]
nz = sum(1 for x in w if x != 0.0)
mean = sum(w) / len(w)
std = math.sqrt(sum((x - mean) ** 2 for x in w) / len(w))
hid = cfg["hidden_size"]
sh = sum(x * x for x in w[:hid]); sm = sum(x * x for x in w[hid:])
print(f"[gate1] in_features {len(w)} (hidden {hid} + markov {len(w)-hid})")
print(f"[gate1] std {std:.6f}  bias {bias:+.5f}  nonzero {nz}/{len(w)}")
print(f"[gate1] squared-norm split: hidden {sh/(sh+sm):.1%}  markov {sm/(sh+sm):.1%}")
if nz != len(w):
    print(f"[gate1] FATAL {len(w)-nz} zero entries -- head is not fully trained"); sys.exit(1)
if not (0.001 < std < 0.05):
    print(f"[gate1] FATAL std {std} outside the trained range seen for this head family"); sys.exit(1)
print("[gate1] passed: head present, correct width, trained, dense")
PYG1
[ $? -eq 0 ] || { echo "[bon] FATAL GATE 1 failed"; exit 1; }

# ---- rows: the same corpus as twoseed, so a0u has a known noise floor ---------
PROMPTS_SRC=${PROMPTS:-/data/datasets/k3-h-v6-apivalid-655-c1sweep}
ROWS_JSONL="$PROMPTS_SRC/random.jsonl"
[ -s "$ROWS_JSONL" ] || { echo "[bon] FATAL missing $ROWS_JSONL"; exit 1; }
if [ -n "${EXPECT_SHA:-}" ]; then
  S=$(sha256sum "$ROWS_JSONL" | cut -d' ' -f1)
  [ "$S" = "$EXPECT_SHA" ] || { echo "[bon] FATAL corpus sha $S != $EXPECT_SHA"; exit 1; }
  echo "[bon] corpus sha verified: $S"
fi
GATE_ROWS=${GATE_ROWS:-8}
python3 - "$ROWS_JSONL" "$GATE_ROWS" "$OUT/gate-rows.jsonl" <<'PYROWS'
import json, sys
src, n, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
# Take the FIRST n rows in file order. Deterministic and identical across arms --
# a per-arm selection would make the identity gates compare different prompts.
out = []
with open(src) as fh:
    for line in fh:
        if not line.strip():
            continue
        r = json.loads(line)
        msgs = r.get("messages")
        if not msgs:
            p = r.get("prompt")
            if p is None:
                continue
            msgs = [{"role": "user", "content": p}]
        out.append({"task_id": r.get("task_id") or r.get("id") or f"row{len(out)}",
                    "messages": msgs})
        if len(out) >= n:
            break
with open(dst, "w") as fh:
    for r in out:
        fh.write(json.dumps(r) + "\n")
print(f"[rows] wrote {len(out)} rows to {dst}")
PYROWS
[ -s "$OUT/gate-rows.jsonl" ] || { echo "[bon] FATAL no gate rows extracted"; exit 1; }
NROWS=$(wc -l < "$OUT/gate-rows.jsonl")
echo "[bon] gate rows: $NROWS"

VROOT=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
SU_PY="$VROOT/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
ST_PY="$VROOT/v1/worker/gpu/states.py"
echo "[bon] vllm root: $VROOT"

PORT=${VLLM_PORT:-19538}
BASE="http://127.0.0.1:$PORT"

# ---- one server per arm ------------------------------------------------------
# Returns with SRV set, or exits. Readiness is polled on /v1/models; the deadline
# is generous because a TP8 load of the 1.5 TB verifier is slow and a short
# deadline reads as a crash.
start_server() {
  local tag="$1"
  local logf="$OUT/server-$tag.log"
  echo "[bon] launching server for arm=$tag (port $PORT)"
  local SPEC="{\"method\":\"dspark\",\"model\":\"$D_B5\",\"num_speculative_tokens\":${N_B5:-5},\"attention_backend\":\"FLASHINFER_MLA\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"}"
  vllm serve "$VERIFIER" \
    --served-model-name kimi-k3 --tensor-parallel-size 8 --trust-remote-code \
    --max-model-len 131072 --kv-cache-memory 51539607552 \
    --max-num-seqs 64 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 --port $PORT \
    --enforce-eager --max-logprobs ${MAX_LOGPROBS:-200} \
    --speculative-config "$SPEC" > "$logf" 2>&1 &
  SRV=$!
  local DEADLINE=$(( $(date +%s) + ${READY_TIMEOUT:-7200} ))
  until curl -s "$BASE/v1/models" >/dev/null 2>&1; do
    kill -0 $SRV 2>/dev/null || {
      echo "[bon] FATAL server exited during startup for arm=$tag; tail:"
      tail -80 "$logf"; return 1; }
    [ "$(date +%s)" -gt "$DEADLINE" ] && {
      echo "[bon] FATAL arm=$tag not ready in ${READY_TIMEOUT:-7200}s"
      tail -80 "$logf"; kill $SRV 2>/dev/null; return 1; }
    sleep 10
  done
  echo "[bon] arm=$tag ready after $(( $(date +%s) - (DEADLINE - ${READY_TIMEOUT:-7200}) ))s"
  grep -q "num_spec_tokens=${N_B5:-5}" "$logf" || {
    echo "[bon] FATAL draft not attached at N=${N_B5:-5} for arm=$tag"
    kill $SRV 2>/dev/null; return 1; }
  echo "[bon] draft attached at N=${N_B5:-5}"
  return 0
}

stop_server() {
  local tag="$1"
  kill ${SRV:-0} 2>/dev/null; wait ${SRV:-0} 2>/dev/null
  local W=0
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && [ $W -lt 300 ]; do
    sleep 5; W=$((W+5))
  done
  echo "[bon] arm=$tag released GPUs after ${W}s"
}

RUNPY=${RUNPY:-/etc/job-rw}
COMMON="--rows $OUT/gate-rows.jsonl --base $BASE --model kimi-k3 --out $OUT \
  --bench $RUNPY/bench_k3h_block5.py --max-tokens ${GATE_MAX_TOKENS:-128} \
  --temperature ${TEMP:-1.0} --top-p ${TOPP:-0.95} --seed ${SEED:-101}"

# ---- arm a0u: UNPATCHED reference, run BEFORE the patch exists ----------------
# Order matters. Patching first and then trying to "unpatch" for the reference would
# make the reference depend on the restore being perfect, which is the thing under
# test. This arm runs against untouched source.
echo "[bon] ===== arm a0u (unpatched reference) ====="
grep -q K3BON_PATCH_APPLIED "$SU_PY" && {
  echo "[bon] FATAL $SU_PY already carries the marker -- this tree is not pristine,"
  echo "[bon]   so a0u would not be an unpatched reference."; exit 1; }
start_server a0u || exit 1
python3 $RUNPY/bon_replay.py $COMMON --arm a0u
A0U_RC=$?
stop_server a0u
[ "$A0U_RC" -eq 0 ] || { echo "[bon] FATAL a0u rc=$A0U_RC"; exit 1; }

# ---- GATE 2: the patch must LAND ---------------------------------------------
# Two files, both resolved from the installed vllm. A backup identical to its file
# means the edit never landed, which beats any string the patch prints about itself.
echo "[bon] GATE 2: applying the best-of-N patch"
echo "[bon]   $SU_PY"
echo "[bon]   $ST_PY"
for f in "$SU_PY" "$ST_PY"; do
  [ -w "$f" ] || { echo "[bon] FATAL $f missing or not writable"; exit 1; }
  cp "$f" "$f.pre_k3bon"
done
python3 $RUNPY/patch_k3_bestof.py "$SU_PY" "$ST_PY" 2>&1 | sed 's/^/[bon]   /'
PRC=${PIPESTATUS[0]}
[ "$PRC" -eq 0 ] || { echo "[bon] FATAL patch failed rc=$PRC"; exit 1; }
for f in "$SU_PY" "$ST_PY"; do
  if [ "$(sha256sum < "$f")" = "$(sha256sum < "$f.pre_k3bon")" ]; then
    echo "[bon] FATAL $f is byte-identical to its pre-patch backup -- the edit did not land"
    exit 1
  fi
  grep -q K3BON_PATCH_APPLIED "$f" || { echo "[bon] FATAL marker absent from $f"; exit 1; }
  python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" \
    || { echo "[bon] FATAL patched $f does not parse"; exit 1; }
done
# py_compile cannot catch a DELETED function -- a span-replace has eaten one in this
# project before, leaving a COMPILE-OK file whose watchdog died at first poll. Check
# the two symbols the branch actually calls still exist.
for sym in _k3bon_branch k3bon_reset_slot _k3bon_nucleus_stats; do
  grep -q "def $sym" "$SU_PY" || { echo "[bon] FATAL $sym absent from $SU_PY"; exit 1; }
done
grep -q "k3bon_reset_slot(req_idx)" "$ST_PY" || {
  echo "[bon] FATAL states.py does not call k3bon_reset_slot -- the latch would never"
  echo "[bon]   clear, and at c=1 every request after the first would refuse to branch."
  exit 1; }
echo "[bon] GATE 2 passed: both files changed, carry the marker, parse, keep their symbols"

# ---- arm a0p: patched, C=1. GATE C ------------------------------------------
# K3BON_DUMP_ALL is set on every patched arm. Without it GATE E is VACUOUS: the main
# dump is filtered by [fires], so gate-fired-but-latched appears in no stream and
# "branched == fired" would be true by construction.
echo "[bon] ===== arm a0p (patched, C=1) ====="
export K3BON_C=1
export K3BON_POLICY=engine
export K3BON_H=${K3BON_H:-0.2472}
export K3BON_TOP_P=${TOPP:-0.95}
export K3BON_DUMP="$OUT/bon-a0p.bin"
export K3BON_DUMP_ALL="$OUT/bonall-a0p.bin"
rm -f "$K3BON_DUMP" "$K3BON_DUMP_ALL"
start_server a0p || exit 1
python3 $RUNPY/bon_replay.py $COMMON --arm a0p \
  --dump "$K3BON_DUMP" --dump-all "$K3BON_DUMP_ALL"
A0P_RC=$?
stop_server a0p
[ "$A0P_RC" -eq 0 ] || { echo "[bon] FATAL a0p rc=$A0P_RC"; exit 1; }

# ---- arm b0: patched, C=2, forced candidate 0. GATE D (the KV gate) ----------
echo "[bon] ===== arm b0 (patched, C=2, forced-0) ====="
export K3BON_C=2
# The policy string CARRIES k: forced-0 / forced-1. There is no K3BON_FORCED_K --
# _K3BonConfig parses k out of the policy name and rejects k outside [0, C).
export K3BON_POLICY=forced-0
export K3BON_DUMP="$OUT/bon-b0.bin"
export K3BON_DUMP_ALL="$OUT/bonall-b0.bin"
rm -f "$K3BON_DUMP" "$K3BON_DUMP_ALL"
start_server b0 || exit 1
python3 $RUNPY/bon_replay.py $COMMON --arm b0 \
  --dump "$K3BON_DUMP" --dump-all "$K3BON_DUMP_ALL"
B0_RC=$?
stop_server b0
[ "$B0_RC" -eq 0 ] || { echo "[bon] FATAL b0 rc=$B0_RC"; exit 1; }

# ---- arm b1: patched, C=2, forced candidate 1. GATE F -----------------------
echo "[bon] ===== arm b1 (patched, C=2, forced-1) ====="
export K3BON_C=2
export K3BON_POLICY=forced-1
export K3BON_DUMP="$OUT/bon-b1.bin"
export K3BON_DUMP_ALL="$OUT/bonall-b1.bin"
rm -f "$K3BON_DUMP" "$K3BON_DUMP_ALL"
start_server b1 || exit 1
python3 $RUNPY/bon_replay.py $COMMON --arm b1 \
  --dump "$K3BON_DUMP" --dump-all "$K3BON_DUMP_ALL"
B1_RC=$?
stop_server b1
[ "$B1_RC" -eq 0 ] || { echo "[bon] FATAL b1 rc=$B1_RC"; exit 1; }

# ---- cross-arm gates C, D, F ------------------------------------------------
echo "[bon] ===== cross-arm gates ====="
python3 $RUNPY/bon_gates.py --out "$OUT"
GRC=$?
echo "[bon] cross-arm gates rc=$GRC"

for tag in a0p b0 b1; do
  for f in "$OUT/bon-$tag.bin" "$OUT/bonall-$tag.bin"; do
    B=$(stat -c %s "$f" 2>/dev/null || echo 0)
    echo "[bon] $f: $B B"
  done
done

# ---- restore the tree so a later arm in the same pod is not silently patched --
for f in "$SU_PY" "$ST_PY"; do
  [ -f "$f.pre_k3bon" ] && cp "$f.pre_k3bon" "$f"
done
echo "[bon] restored pristine source from .pre_k3bon backups"

echo "[bon] done=$(date -u +%FT%TZ) gates_rc=$GRC"
echo "[bon] NOTE this run gates the patch only. It does NOT measure the conjecture:"
echo "[bon]   ${GATE_ROWS:-8} rows at ${GATE_MAX_TOKENS:-128} max_tokens is far too small for an"
echo "[bon]   accept_len claim. The B0/B1 pilot is a separate, much longer job."
exit $GRC
