#!/usr/bin/env bash
# Render the diverge-token best-of-N GATES job into a Polaris-submittable file.
#
# Usage:
#   JOB_NAME=k3bon-gates PIN_NODE=host-172-16-1-246 \
#     memo/benchmark-reports/diverge-bestof/render-job-bon-gates.sh > /tmp/job-bon.yaml
#   curl -sS -K /tmp/.polaris.curlrc -X POST \
#     'https://api-inner.polaris.ppio.com/api/v1/jobs' \
#     -H 'Content-Type: application/yaml' --data-binary @/tmp/job-bon.yaml
#
# `curl --data-binary @file` does NOT expand shell variables, so an unrendered
# file registers a job literally named '$JOB_NAME'. Everything is substituted here.
#
# UNPINNED, and the two things that would otherwise force a pin are handled in the
# job rather than by a hostname.
#
# 1. NODE-LOCAL CORPUS. random.jsonl is 102 MB and exists on only two of the nine
#    nodes, which is exactly the condition the eval template names as forcing a pin.
#    So the 8 gate rows ride in the ConfigMap as gzipped binaryData (417 KB, ~40% of
#    the 1 MiB cap) and no node needs the corpus. The `datasets` mount is therefore
#    also dropped -- its hostPath may not exist on every node.
#
# 2. GPUs THAT ARE NOT ACTUALLY FREE. This cluster runs large inference under
#    nerdctl, OUTSIDE k8s, holding VRAM the scheduler cannot see. Measured
#    2026-09-11: host-172-16-0-54 reported 0/8 GPUs requested to k8s while its own
#    nvidia-smi showed 8/8 cards busy, held by leaked processes from a pod already in
#    Error. Volcano can place this job there. run_bon_gates.sh therefore reads real
#    per-card usage first and exits 9 with "PLACEMENT failure" rather than dying in a
#    CUDA OOM several minutes into a TP8 load, which would read as an engine fault.
#
# EXPECT_SHA still names the PARENT corpus, for provenance. It is NOT recomputable
# from the shipped subset, so ROWS_GZ_SHA pins the subset and the job says which of
# the two it actually verified.
set -euo pipefail

: "${JOB_NAME:?set JOB_NAME}"
# Empty by default: the job is submitted unpinned. Set PIN_NODE only to recover from
# an exit-9 placement failure, naming a node verified free by nvidia-smi.
: "${PIN_NODE:=}"
: "${GATE_ROWS:=8}"
: "${GATE_MAX_TOKENS:=128}"
: "${VLLM_PORT:=19541}"
: "${RUN_GENERATION:=$(date -u +%Y%m%dT%H%M%SZ)-bon-gates}"
: "${OUT_DIR:=/data/output/results-bon-gates}"
# r7's corpus sha. Checked in-job so a swapped corpus cannot silently change the run.
: "${EXPECT_SHA:=01a447704f54c98d3ff9ba1d413b0f261e4f5f12b9339d1be677d2589846f8ba}"

# The gzipped gate rows to ship. Built by:
#   ssh <node> 'head -8 /data/datasets/k3-h-v6-apivalid-655-c1sweep/random.jsonl \
#     | gzip -9' > gate-rows.jsonl.gz
: "${ROWS_GZ_FILE:?set ROWS_GZ_FILE to the gzipped gate rows}"
[ -s "$ROWS_GZ_FILE" ] || { echo "[render] FATAL missing $ROWS_GZ_FILE" >&2; exit 1; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# bench_k3h_block5.py provides snapshot/delta over the engine's Prometheus
# counters, which is where accept_len comes from. It lives in the diary repo, so
# this resolves the sibling path when run from there and otherwise needs BENCH_PY.
# A wrong guess here is not silent: the loop below refuses a missing file.
: "${BENCH_PY:=$here/../twoseed/bench_k3h_block5.py}"
bench="$BENCH_PY"
for f in "$here/run_bon_gates.sh" "$here/bon_replay.py" "$here/bon_gates.py" \
         "$here/patch_k3_bestof.py" "$bench"; do
  [ -s "$f" ] || { echo "[render] FATAL missing $f" >&2; exit 1; }
done

# Indent a file into a YAML block scalar at a fixed depth.
emit() { sed 's/^/      /' "$1"; }

ROWS_GZ_SHA="$(sha256sum "$ROWS_GZ_FILE" | cut -d' ' -f1)"
ROWS_B64="$(base64 -w0 "$ROWS_GZ_FILE")"
ROWS_N="$(zcat "$ROWS_GZ_FILE" | wc -l)"
# The rows are the unit of work for every arm, so a mismatch between what was
# packed and what the job will run must be caught here, not read as a short run.
[ "$ROWS_N" -ge "$GATE_ROWS" ] || {
  echo "[render] FATAL $ROWS_GZ_FILE holds $ROWS_N rows but GATE_ROWS=$GATE_ROWS" >&2
  exit 1; }
echo "[render] rows: $ROWS_N packed, $GATE_ROWS will run, sha $ROWS_GZ_SHA" >&2
echo "[render] pin: ${PIN_NODE:-<none, unpinned>}" >&2

cat <<YAML
# Rendered by render-job-bon-gates.sh at $(date -u +%FT%TZ)
# generation: $RUN_GENERATION
cluster: maiyi-b300
configMap:
  apiVersion: v1
  kind: ConfigMap
  metadata:
    name: $JOB_NAME-cm
  # binaryData lands as a real binary file in the mount, so the job needs no base64
  # step and the 1 MiB cap counts the DECODED bytes.
  binaryData:
    gate-rows.jsonl.gz: $ROWS_B64
  data:
    run_bon_gates.sh: |
$(emit "$here/run_bon_gates.sh")
    bon_replay.py: |
$(emit "$here/bon_replay.py")
    bon_gates.py: |
$(emit "$here/bon_gates.py")
    patch_k3_bestof.py: |
$(emit "$here/patch_k3_bestof.py")
    bench_k3h_block5.py: |
$(emit "$bench")
job:
  apiVersion: batch.volcano.sh/v1alpha1
  kind: Job
  metadata:
    name: $JOB_NAME
  spec:
    minAvailable: 1
    tasks:
      - name: bon
        replicas: 1
        template:
          metadata:
            labels:
              polaris.novita.ai/app: eval
          spec:
$(if [ -n "$PIN_NODE" ]; then
    printf '            nodeSelector:\n              kubernetes.io/hostname: %s\n' "$PIN_NODE"
  else
    printf '            # unpinned: placement is the scheduler'"'"'s, and run_bon_gates.sh\n'
    printf '            # exits 9 if the chosen node'"'"'s cards are not actually free.\n'
  fi)
            # MANDATORY. Polaris injects no restartPolicy, so the default is
            # Always: kubelet would restart the finished container in place, the
            # pod would never reach Succeeded, Volcano would never emit
            # TaskCompleted, and the 8 GPUs would stay held after the run.
            restartPolicy: Never
            containers:
              - name: bon
                image: image.paigpu.com/library/vllm-openai:kimi-k3
                imagePullPolicy: IfNotPresent
                workingDir: /data/output
                command:
                  - /bin/bash
                  - -c
                  - |
                    set -u
                    mkdir -p /etc/job-rw
                    # ConfigMap mounts are read-only and the patch script must be
                    # importable/executable from a writable place.
                    cp /etc/job/bon_replay.py /etc/job/bon_gates.py \
                       /etc/job/patch_k3_bestof.py /etc/job/bench_k3h_block5.py \
                       /etc/job-rw/
                    exec bash /etc/job/run_bon_gates.sh
                env:
                  - name: RUN_GENERATION
                    value: "$RUN_GENERATION"
                  - name: OUT_DIR
                    value: "$OUT_DIR"
                  - name: RUNPY
                    value: "/etc/job-rw"
                  - name: VLLM_PORT
                    value: "$VLLM_PORT"
                  - name: GATE_ROWS
                    value: "$GATE_ROWS"
                  - name: GATE_MAX_TOKENS
                    value: "$GATE_MAX_TOKENS"
                  - name: N_B5
                    value: "5"
                  - name: TEMP
                    value: "1.0"
                  - name: TOPP
                    value: "0.95"
                  - name: SEED
                    value: "101"
                  - name: EXPECT_SHA
                    value: "$EXPECT_SHA"
                  - name: ROWS_GZ
                    value: "/etc/job/gate-rows.jsonl.gz"
                  - name: ROWS_GZ_SHA
                    value: "$ROWS_GZ_SHA"
                  - name: MAX_LOGPROBS
                    value: "200"
                  - name: VLLM_LOGGING_LEVEL
                    value: "INFO"
                  # Loopback must not go through a proxy; a proxied 127.0.0.1
                  # returned 502 on every request during the maiyi-9 runs.
                  - name: no_proxy
                    value: "127.0.0.1,localhost"
                  - name: NO_PROXY
                    value: "127.0.0.1,localhost"
                resources:
                  limits:
                    nvidia.com/gpu: 8
                  requests:
                    nvidia.com/gpu: 8
                # The control plane injects the VOLUMES but not the MOUNTS.
                volumeMounts:
                  - name: job-rw
                    mountPath: /etc/job-rw
                  - name: models
                    mountPath: /data/models
                  - name: output
                    mountPath: /data/output
                  - name: job-cm
                    mountPath: /etc/job
                    readOnly: true
                  - name: shm
                    mountPath: /dev/shm
                  - name: infiniband
                    mountPath: /dev/infiniband
YAML
