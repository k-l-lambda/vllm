# vLLM checkout for the K3 DSpark work

Created 2026-09-11 on host-172-16-2-140 (maiyi-9), via
`export HTTPS_PROXY=http://10.96.0.106:1080`.

## What is here

- `/home/linux/work/vllm-dspark`   full public clone, branch **`k3-dspark-base`**
  at `aaaeda98dcc14af75cd5fbd26386751a0e8b79c9` (2026-07-25).
- `/home/linux/work/vllm-k3-vendor/vllm`  the **installed tree copied out of the
  served image** `image.paigpu.com/library/vllm-openai:kimi-k3`
  (image id `1f7553849e6b8`, layer `sha256:888c3446265e...`),
  `__version__ = 0.1.dev19262+gb6bbf29dd.d20260727`.

## Why the branch is not the image commit

`b6bbf29dd` does not exist on the public remote. Verified against a *fresh* full
clone, not a stale one:

- `git cat-file -t b6bbf29dd` -> `Not a valid object name`
- `git fetch origin b6bbf29dd` -> `couldn't find remote ref`
- no prefix match among the 43163 `refs/pull/*` refs

The version string's `dev19262` counts the **vendor's own history**, not upstream's:
upstream count 19262 matches the installed tree only 74.9%, while the true base
matches 95.00%. So ~160 vendor commits sit on top of the base, and the
`.d20260727` suffix says the image was even built from a dirty tree.

## How the base was pinned

Not by the hash. By blob content, then by minimising the diff:

1. 7 generic files' blobs located in upstream history -> 6 agreed on base count
   `[19028..19102]`; `v1/kv_cache_interface.py` sits at `[19329..19406]`, i.e. the
   vendor cherry-picked that one change.
2. Full comparison of all 2296 vendor `.py` against candidate trees:

   | upstream count | commit | date | match |
   |---|---|---|---|
   | 19082 | 33c4f3551c | 07-24 | 94.26% |
   | 19084 | 6a1acac3fe | 07-25 | 94.90% |
   | **19086** | **70052fb924** | 07-24 | **95.00%** |
   | **19088** | **aaaeda98dc** | 07-25 | **95.00%** |
   | 19090 | 94682b79f4 | 07-24 | 94.85% |

3. The 19086/19088 tie is **not resolvable from the installed tree**: the only
   files differing between them are CI scripts and `tests/`, which an installed
   package does not ship. `70052fb924` is an ancestor of `aaaeda98dc`; the later
   one was taken.

## Vendor delta at this base

- 258 vendor-only `.py`, dominated by the `models/kimi_k3/` family (amd/nvidia/xpu).
- Only 2 upstream-only `.py`.
- 9 differing common files under `worker`/`spec_decode`:
  `v1/spec_decode/llm_base_proposer.py`, `v1/worker/gpu/block_table.py`,
  `cudagraph_utils.py`, `model_runner.py`, `model_states/mamba_hybrid.py`,
  `spec_decode/autoregressive/cudagraph_utils.py`,
  `spec_decode/dflash/speculator.py`, `warmup.py`, `gpu_model_runner.py`.

## Patch targets: 5 of 8 are byte-identical to upstream here

| file | upstream | vendor | |
|---|---|---|---|
| `spec_decode/rejection_sampler_utils.py` | 1129 | 1129 | **identical** |
| `spec_decode/rejection_sampler.py` | 273 | 273 | **identical** |
| `spec_decode/dspark/speculator.py` | 169 | 169 | **identical** |
| `gpu/sample/gumbel.py` | 264 | 264 | **identical** |
| `gpu/states.py` | 129 | 129 | **identical** |
| `gpu/model_runner.py` | 1677 | 1681 | differs by **4 lines** (vendor adds `for_capture=` to the CUDA-graph FULL replay metadata re-stage) |
| `spec_decode/dflash/speculator.py` | 687 | 705 | differs, 31 changed lines |
| `models/kimi_k3/nvidia/dspark_mla.py` | absent | 526 | vendor-only; upstream's `models/deepseek_v4/nvidia/dspark.py` is the same family pattern |

**DSpark itself is upstream code at this commit** (`spec_decode/dspark/` and
`dflash/` both present, `rejection_sampler_utils.py` already 1129 lines with block
verification). It is absent from current `main` (2026-09-11) and from anything
before ~2026-07, which is why a mid-June or a September checkout both look like it
never existed.

## Consequence for the patch

Sites 1, 2, 3 and 5 of `memo/benchmark-reports/diverge-bestof/PLAN.md` can be
developed and diffed against this checkout. Site 4 differs by a known 4 lines. Only
the K3 model file must be read from the vendor copy. The runtime patch still applies
to the *installed* tree with the sha256-vs-backup proof, unchanged.
