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
# WHY THIS IS PINNED, against the eval template's default. That template argues for
# unpinned and lists node-local weights as verified on all 9 nodes. Both still hold.
# The reason to pin is the OTHER hazard the same template documents: this cluster
# runs large inference under nerdctl, OUTSIDE k8s, holding VRAM the scheduler cannot
# see. Measured 2026-09-11:
#   host-172-16-1-246   0/8 cards busy   <- genuinely free, and has the corpus
#   host-172-16-0-54    8/8 cards busy   while k8s reports 0/8 requested
#   host-172-16-3-229   0/8 cards busy   but NO corpus at all
# An unpinned 8-GPU job can be placed on 0-54 and contend for VRAM it will not get.
# So the pin is the template's own rule applied, not an exception to it.
#
# The corpus rides on the node, not in the ConfigMap: random.jsonl is 102 MB and
# 1-246's copy is byte-identical to 2-140's (sha 01a44770..., matching r7's
# EXPECT_SHA). Shipping 8 gzipped rows would fit the 1 MiB cap but would throw away
# the other 647 rows the later pilot needs.
set -euo pipefail

: "${JOB_NAME:?set JOB_NAME}"
: "${PIN_NODE:=host-172-16-1-246}"
: "${GATE_ROWS:=8}"
: "${GATE_MAX_TOKENS:=128}"
: "${VLLM_PORT:=19541}"
: "${RUN_GENERATION:=$(date -u +%Y%m%dT%H%M%SZ)-bon-gates}"
: "${OUT_DIR:=/data/output/results-bon-gates}"
# r7's corpus sha. Checked in-job so a swapped corpus cannot silently change the run.
: "${EXPECT_SHA:=01a447704f54c98d3ff9ba1d413b0f261e4f5f12b9339d1be677d2589846f8ba}"

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

cat <<YAML
# Rendered by render-job-bon-gates.sh at $(date -u +%FT%TZ)
# generation: $RUN_GENERATION
cluster: maiyi-b300
configMap:
  apiVersion: v1
  kind: ConfigMap
  metadata:
    name: $JOB_NAME-cm
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
            nodeSelector:
              kubernetes.io/hostname: $PIN_NODE
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
                  - name: PROMPTS
                    value: "/data/datasets/k3-h-v6-apivalid-655-c1sweep"
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
                  - name: datasets
                    mountPath: /data/datasets
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
