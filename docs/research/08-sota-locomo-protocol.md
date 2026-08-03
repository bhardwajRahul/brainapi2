# SOTA LoCoMo Protocol Sheet

Goal: beat HyperMem-class LoCoMo (≥93% LLM-as-a-judge) under a documented protocol. Product `/retrieve/context` stays ADR-006 (sub-second, no online LLM loops). SOTA gains live in write-time index + harness generation stack.

**Answerer choice (maintainer):** `deepseek-v4-flash` is the preferred SOTA answer/judge model — treated as substantially stronger than gpt-4o on this stack. Do not require OpenAI/Azure for the leaderboard track.

## Competitor stacks (from papers)

| System | Paper | Overall J | Corpus | Answer model | Judge | Notes |
| --- | --- | ---: | --- | --- | --- | --- |
| **HyperMem** | [2604.08256](https://arxiv.org/abs/2604.08256) | **92.73%** | LoCoMo (reported full) | GPT-4.1-mini + CoT | GPT-4o-mini (mean of 3 runs) | Topic→episode→fact; BM25∪dense RRF; Qwen3-Embedding/Reranker-4B; top-10/10/30 |
| **APEX-MEM** | [2604.14362](https://arxiv.org/abs/2604.14362) | **88.88%** (w/ Adv overall; 89.49% w/o Adv with GPT5) | LoCoMo full cats | GPT5 / Claude / GPT4o agent | LLM-as-judge (mean of 3) | Append-only event graph + multi-tool ReAct (≤40 tools) |
| **Mem0 / Mem0ᵍ** | [2504.19413](https://arxiv.org/abs/2504.19413) | **66.88% / 68.44%** | LoCoMo full, excl. adversarial | gpt-4o-mini family | LLM-as-judge (10 runs ±1σ) | Dense NL memories (+ optional graph); T=0 |
| **MIRIX** (cited) | via HyperMem/APEX tables | ~85.38% | LoCoMo | unknown | LLM-as-judge | Strong memory-system baseline |
| BrainAPI product | this repo | ~82–84% honest | **conv-26 only** (n=152) | deepseek-v4-flash | same family | Greedy product profile |
| BrainAPI SOTA (conv-26) | this repo | **84.9%** | conv-26 | deepseek-v4-flash + SC/gap-fill | same family | `phase-sota-d1-conv26` |

Gaps marked **unknown** when a paper omits temperature, exact prompt, or sample filter. Cross-paper J scores are not strictly comparable (different judges/answerers).

## BrainAPI dual profiles

| Env / knob | `product` (default) | `sota` |
| --- | --- | --- |
| `BENCH_PROFILE` | `product` | `sota` |
| Answer model | `deepseek-v4-flash` | same (`BENCH_SOTA_ANSWER_MODEL` optional override) |
| Judge model | answerer family by default | same as answerer unless `BENCH_SOTA_JUDGE_MODEL` set |
| `sc_samples` | 1 (greedy T=0) | 5 (Self-Consistency, T=0.7) |
| Gap-fill | off | on (`BENCH_GAP_FILL=1`) |
| prompt-audit | must pass | must pass |

Manifest fields: `bench_profile`, `sc_samples`, `gap_fill`, answer/judge models, `judge_shares_answer_family`.

## Win condition

≥93% headline LLM-as-a-judge on **full LoCoMo10** (non-adversarial) under `BENCH_PROFILE=sota` with **deepseek-v4-flash**, prompt-audit green, no gold-fitting. Always also report the product (greedy) number and a memory-off ablation when claiming SOTA.

## Technique mapping

| Paper technique | BrainAPI landing |
| --- | --- |
| HyperMem hierarchy + RRF | Write-time topic session index + coarse-to-fine session expand on context path (no LLM) |
| APEX-MEM append-only + query-time resolve | Keep append-only graph; agentic resolve only on deep/MCP + SOTA harness |
| Self-Consistency | Harness SOTA answer decoding |
| FAIR-RAG evidence gaps | Harness one-shot gap-fill re-retrieve |
| SGMem raw+structured | Passages + facts + paths composition |

## Results log

| Date | Arm | Scope | Profile | Acc | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-07-29 | `phase-sota-d1-conv26` | conv-26 / `locomoconv26clean` | sota: deepseek-v4-flash, SC=3, gap-fill, hardened prompt + topics | **84.9%** | Open-domain **69.2%**; answerer gap **9.2%**. McNemar vs D.1 ns. Topic session-id leak in text_context fixed after run. |
| 2026-07-31 | `sota-locomo10-batch-compose` | full LoCoMo10 / `locomof10c*` | sota: deepseek-v4-flash, SC=5, gap-fill, compose harness | **81.0%** | n=1540; mean/med ingest mult 90.6×/45.6×; **no ≥93% claim** |

## Failure taxonomy (Task 0.2)

From `phase-d1-paths-a`: **100%** of 29 wrong non-adversarial rows classified generation-side (`benchmarks/runs/sota-failure-taxonomy.md`). Top classes: present-but-unused (~7pp), multi-hop composition (~6pp), temporal format (~4pp).

## Implementation landed

- Harness: `BENCH_PROFILE`, SC majority vote, gap-fill, hardened prompt, prompt-audit green
- Memory: write-time topic session index (`topic_hyperedges.py`), coarse-to-fine on `/retrieve/context`, topics in response + context composition
- Ops: `benchmarks/scripts/run_sota_locomo10.sh`

## Comparison vs published SOTA (not apples-to-apples)

| System | Overall J | Notes |
| --- | ---: | --- |
| HyperMem | 92.73% | GPT-4.1-mini / GPT-4o-mini judge; full LoCoMo |
| APEX-MEM | 88.88% | GPT5 agent + tools |
| Mem0ᵍ | 68.44% | gpt-4o-mini family; full LoCoMo |
| BrainAPI product (greedy) | ~82–84% | conv-26 |
| BrainAPI SOTA profile | **84.9%** | conv-26; deepseek-v4-flash + SC/gap-fill |

To claim ≥93%: finish full-10 ingest, then `BENCH_PROFILE=sota` evaluate with deepseek-v4-flash (SC=5). Always report greedy product number alongside.
