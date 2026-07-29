# Evaluation, Measurement, and Downstream Application Surfaces

Workstream 04. Scope and constraints: `docs/research/00-scope-and-constraints.md`.

Every implementation claim below is anchored to `file:line` as read at commit `649f5e5` plus the uncommitted working-tree changes to `benchmarks/locomo/`. Every research claim carries an arXiv ID. Where I could not determine something, I say so.

A note on freshness: while I was reading, run `locomo-conv26-push75-d` was executing against a working tree whose answer prompt had been edited minutes earlier. That run turned out to be the most informative artifact in the repository, for reasons in section 2.1.

## What this workstream does

### The harness, end to end

`benchmarks/locomo/` is a self-contained CLI that drives the public BrainAPI HTTP surface. It never imports `src/`, so it measures the deployed server as a black box.

**Command surface** (`benchmarks/locomo/cli.py`): `download`, `stats`, `smoke`, `ingest`, `answer-once`, `selftest-metrics`, `evaluate`, `report`.

**Dataset** (`benchmarks/locomo/dataset.py`). `locomo10.json` is fetched from `snap-research/locomo` (`config.py:12-15`), 10 conversations, 1,986 QA pairs. Turns are flattened to `f"{speaker} ({dia_id}): {text}"` (`dataset.py:76`) and each session is prefixed with its wall-clock time and a synthetic `Session id: {session_key}.` line (`dataset.py:87`). That injected session id is what evidence-recall scoring later keys on.

**Ingestion** (`benchmarks/locomo/ingest.py`). One unit per session, submitted to the observation endpoint, then polled to a terminal status (`config.py:20`). Per-unit records capture `submit_latency_ms` and `wait_latency_ms` (`ingest.py:22-36`). They do not capture tokens, extraction counts, or cost. `client.py:92-100` hardcodes the ingestion directive (`observate_for`, `preferred_extraction_entities` = Person/Event/Location/Date), so ingestion is configured for LoCoMo in code, not in the manifest.

**Retrieval + answer** (`benchmarks/locomo/client.py`, `answer.py`). `retrieve_context` posts the question to `/retrieve/context` with `historical_limit`, `max_passages`, `max_facts`, `apply_fact_filter`, `use_ppr`, `sufficiency_retry`. `answer.py:42-46` calls the answer model at `temperature=0` with no seed; `answer.py:54-79` retries once if the model returns empty.

**Judging** (`benchmarks/locomo/judge.py`, `prompts.py`). `judge_answer` (`judge.py:56-87`) sends `JUDGE_SYSTEM` (`prompts.py:20-25`) with a user message containing exactly three fields: question, gold, prediction (`prompts.py:86-90`). `temperature=0`, `response_format=json_object` (`judge.py:68-73`). The judge returns `{correct: bool, reason: str}`.

**Scoring** (`benchmarks/locomo/evaluate.py`, `metrics.py`). Per question the harness writes an `AnswerRecord` (`evaluate.py:30-57`) that is unusually rich: prediction, gold, category, evidence, judge verdict and raw payload, F1, BLEU-1, three separate latencies, both model names, answer and judge token counts, and the full retrieved context (`text_context` capped at 20,000 chars, `evaluate.py:25,194`), source passages, historical context, and the session ids parsed out of them. Because the context blob is persisted per row, most of the offline analysis in section 2 is possible after the fact.

`metrics.aggregate_answers` (`metrics.py:154-245`) computes, per category and overall: judge accuracy with a Wilson 95% interval (`metrics.py:79-90`), mean F1 and BLEU-1, `answerable_rate`, evidence-session recall at full and partial, `answerer_gap`, retrieval latency p50/p95/mean, and `total_llm_tokens`.

**Reporting** (`benchmarks/locomo/report.py`). `build_report` filters out errored rows (`report.py:34`), aggregates, and renders `report.md` plus `report.json` with a headline, a per-category table, latency, tokens, and ingest status.

### What the instrumentation already gets right

This is a better-instrumented harness than most. Specifically:

- **Per-category breakdown exists** and uses the real LoCoMo taxonomy: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial (`config.py:21-27`), surfaced in the report table (`report.py:105-118`).
- **Confidence intervals exist** on every accuracy figure (`metrics.py:175,185-187`). Point estimates are not reported bare.
- **A retrieval ceiling proxy exists.** `gold_in_context` (`metrics.py:133-139`) checks whether at least half the gold tokens longer than two characters appear anywhere in the retrieved blob, and is reported as `answerable_rate`.
- **A first attempt at failure attribution exists.** `answerer_gap = answerable_rate − judge_accuracy` (`metrics.py:211-219`) is intended to separate "memory did not supply it" from "the answerer had it and still failed."
- **Cost and latency sit next to accuracy** in the same report (`report.py:120-129`), which is exactly the right instinct.
- **`selftest-metrics`** (`metrics.py:248-275`, `cli.py:143-150`) unit-tests the metric functions themselves, including a `gold_in_context` and an `evidence_coverage` case.

So the gaps below are not gaps of effort or of instinct. They are gaps of *validity*: the harness measures many things carefully, and almost none of those measurements are currently defended against the ways they can be wrong.

### The unmeasured application surfaces

**Synergies / recommendations.** `EntitySinergyRetriever.retrieve_sibilings` (`src/core/search/entity_sibilings.py`) is reached from `GET /retrieve/entity/synergies` (`src/services/api/routes/retrieve.py:501-526`) via `get_entity_sibilings` (`src/services/api/controllers/entities.py:83-131`). Scoring weights are module-level constants (`entity_sibilings.py:26,29,32,35`: `DIRECT_MULTIPLIER=1.30`, `REMOTE_MULTIPLIER=0.70`, `FACTORS_INCREMENTAL_WEIGHT=0.3`, `NODE_SIM_DESC_INCREMENTAL_WEIGHT=0.5`). The `polarity` argument is declared and documented (`entity_sibilings.py:53,65`) but never read in the method body — the `same`/`opposite` distinction the route advertises (`routes/retrieve.py:515`) does not exist in the implementation. The return is unbounded (`entity_sibilings.py:437`). There is no benchmark, no test, and no metric for this endpoint.

**MCP tool surface.** Five tools (`src/services/mcp/main.py:168,231,272,303,328`). Authorization failures return the bare string `"Unauthorized"` (`main.py:212,266,288,321`) rather than a structured error, so a calling agent cannot distinguish "not allowed" from "no results" without string matching. There is no benchmark for whether an agent equipped with these tools completes tasks better.

**Observability.** Tracing is a bespoke in-process tracker (`src/lib/tracing/tracker.py`), enabled by default (`tracker.py:83`). `start_subscribers` (`tracker.py:353`) is not invoked on the production path, so events stay in an in-memory queue and are dropped when it fills. Spans emit on breach, not on success, so the happy path is invisible. Token usage is logged to stdout only (`src/core/agents/core/invoke_loop.py:51`), never attached to a trace or aggregated. Retrieval and synergies are not instrumented at all.

**Tests as a gate.** 25 files under `tests/`, all `unittest`, no `conftest.py`, no coverage configuration in `pyproject.toml`. No workflow runs them: `staging.yaml:15` is literally `- run: "[ NOT IMPLEMENTED YET ]"`, and `deploy.yaml` / `tags.yaml` build and ship without a test step. There is no quality gate in CI today.

## Guarantees and where they break

The guarantee this workstream is trying to make: **a number in `report.md` moving up means the memory layer got better, and a number moving down means it got worse.** Everything else — cost tracking, attribution, CI gating — exists to make that inference safe.

Here is the recorded history, recomputed from `answers.jsonl` rather than taken from `report.json`, so that duplicate and errored rows are visible:

| run | scored rows | unique QA | judge acc | multi-hop (n=32) | temporal (n=37) | open-domain (n=13) | single-hop (n=70) | p50 ms | p95 ms | harness tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-a | 156 | 152 | 0.0% | 0% | 0% | 0% | 0% | 327 | 540 | 0.08M |
| v3 | 151 | 151 | 38.4% | 22% | 30% | 31% | 52% | 484 | 992 | 1.76M |
| v4-context | 152 | 152 | 44.1% | 22% | 38% | 46% | 57% | 6184 | 8368 | 2.12M |
| v5 | 152 | 152 | 38.2% | 22% | 38% | 15% | 50% | 7111 | 10615 | 2.20M |
| v6 | 152 | 152 | 75.0% | 66% | 81% | 46% | 81% | 4414 | 5694 | 4.01M |
| push75-a | 152 | 152 | 72.4% | 50% | 76% | 46% | 86% | 4081 | 5683 | 4.01M |
| push75-b | 152 | 152 | 75.7% | 56% | 84% | 54% | 84% | 4053 | 5446 | 4.06M |
| push75-c | 152 | 152 | 82.9% | 78% | 86% | 54% | 89% | 4793 | 6333 | 7.74M |
| push75-d | 152 | 152 | 86.2% | 81% | 92% | 69% | 89% | 4848 | 6902 | 7.86M |

The gaps below are ranked by how much they distort that inference, worst first.

### 1. The answer prompt has been fitted to conv-26 gold strings, so the headline now partly measures prompt-to-annotator alignment

This is the most serious problem, it is currently active in the working tree, and it is demonstrable rather than suspected.

`push75-c` and `push75-d` have **byte-identical manifests** apart from `run_id` and timestamps: same brain, same `use_ppr=True`, `max_passages=16`, `max_facts=50`, `apply_fact_filter=True`, `sufficiency_retry=True`, same answer and judge model. The only thing that changed between them is `ANSWER_SYSTEM` in `benchmarks/locomo/prompts.py`, which is uncommitted. Accuracy moved 82.9% → 86.2%.

Now compare the added prompt rules against the gold answers they recover:

| Added rule (`prompts.py`, uncommitted) | conv-26 gold answer | push75-c prediction | push75-d prediction |
| --- | --- | --- | --- |
| L9: `Keep speaker names explicit (e.g. "Melanie's slipper", not "my slipper")` | `In Melanie's slipper` | `In my slipper.` — wrong | `...in Melanie's slipper...` — right |
| L14: `always answer with a short hedged conclusion ... (e.g. "Likely no; ...")` | `LIkely no; though she likes reading...` | `Not mentioned in the conversation.` — wrong | `Likely no; the conversation shows...` — right |
| L13: `Prefer the relative phrasing ... "the week of <date>"` | `The week of 23 August 2023` | `23 August 2023 (during the session...)` — wrong | `...the week of August 23, 2023...` — right |
| L13: `For "how long" questions, prefer "since <year>"` | `Since 2016` | `Seven years.` — wrong | `Seven years (as of September 2023).` — right |
| L12: `ENUMERATION: ... Return ALL matching items` | `Her mentors, family, and friends` | `Melanie, friends, and family.` — wrong | `Melanie, friends, family, and mentors.` — right |

The prompt does not merely describe a formatting preference. Line 9 contains the gold answer string `Melanie's slipper` *and* the exact wrong answer the system gave on the previous run (`my slipper`). Line 14's example hedge `"Likely no; ..."` is the opening of the gold answer for the open-domain items. Line 13's Sunday-versus-Saturday rule resolves one specific question (`When did Melanie run a charity race?`, gold `The sunday before 25 May 2023`). `README.md` warns not to ingest the QA annotations because that leaks the test set; the letter of that rule is respected while its purpose is defeated through the prompt instead.

The `Since 2016` / `Seven years` case deserves separate attention, because it shows the failure is not only leakage. In both runs the memory layer supplied the fact — the system knew the duration. `push75-c` was marked wrong and `push75-d` right for the same underlying retrieval, purely because `push75-d` appended a parenthetical date. Neither prediction contains "2016". That question was never a memory failure in either run, yet it counts as one datapoint of memory quality in both.

Consequence: the headline conflates *did the memory layer supply the fact* with *did the answer prompt phrase it the way LoCoMo's annotator did*. Because the prompt is tuned on the same 152 questions that produce the score, there is no held-out set on which to detect this. The gains from `v5` (38.2%) to `push75-d` (86.2%) are a mixture of genuine retrieval improvement, which section 2.5 shows is real and large, and prompt fitting, which is not, and the harness cannot currently tell you the ratio.

Classification: **bug**, in the sense that it invalidates the measurement. The prompt edits are individually defensible as product behaviour ("never return an empty answer" is a reasonable rule); the problem is that they were derived from and validated on the eval set.

### 2. Every recorded number comes from one conversation out of ten

All ten run directories are `locomo-conv26-*`, and every manifest has `samples: ["conv-26"]`. That is 152 of the dataset's 1,540 non-adversarial questions — 9.9%. Two speakers, Caroline and Melanie, one persona pair, one event graph.

Sample size drives everything downstream. Wilson intervals on the headline are 11-14pp wide, and per category they are far worse. For `push75-c`:

| category | n | accuracy | Wilson 95% CI | width |
| --- | ---: | ---: | --- | ---: |
| multi-hop | 32 | 78.1% | [61.2, 89.0] | 27.7pp |
| temporal | 37 | 86.5% | [72.0, 94.1] | 22.1pp |
| open-domain | 13 | 53.8% | [29.1, 76.8] | 47.7pp |
| single-hop | 70 | 88.6% | [79.0, 94.1] | 15.1pp |

The two categories the maintainer named as the priority — multi-hop and temporal — are measured at n=32 and n=37. A real 10pp improvement in multi-hop is invisible at this sample size. The harness computes and prints these intervals honestly; the problem is that decisions are being taken on differences much smaller than the intervals.

### 3. Configurations are compared with an unpaired test when a paired one is available and free

The runs are paired by construction: the same 152 questions, same ids. The harness compares them by eyeballing two independent Wilson intervals, which throws away the pairing and loses most of the power. Running an exact McNemar test on the recorded rows instead:

| comparison | flipped to correct | flipped to wrong | exact p | verdict |
| --- | ---: | ---: | ---: | --- |
| v6 → push75-a | 7 | 11 | 0.481 | not distinguishable |
| push75-a → push75-b | 12 | 7 | 0.359 | not distinguishable |
| push75-b → push75-c | 20 | 9 | 0.061 | not distinguishable at 0.05 |
| push75-c → push75-d | 11 | 6 | 0.332 | not distinguishable |
| push75-a → push75-c | 24 | 8 | 0.007 | real |
| push75-a → push75-d | 25 | 4 | 0.0001 | real |

**Not one of the individual steps the team took is statistically distinguishable from the step before it.** Only the accumulated a→c and a→d differences are. Meanwhile the churn is large: even between configurations that are statistically identical, 15-20 questions flip in each direction. The `+3.3pp` from `push75-a` to `push75-b` that reads as progress in `report.md` is 12 questions gained and 7 lost, p=0.36.

This also means `v6` → `push75-a` — which looks like a small 2.6pp regression — actually broke 11 questions while fixing 7, and dropped multi-hop from 66% to 50%. A headline that moves 2.6pp hid a 16pp per-category swing.

Classification: **gap**, and the cheapest one to close. McNemar on paired runs needs no new data, only the rows already on disk.

### 4. The judge is unvalidated, is its own answerer, and cannot see the evidence

Three distinct problems, all in `judge.py` and `prompts.py`.

**No human calibration.** There is no file of human labels anywhere in the repo, and no agreement statistic is computed. Judge accuracy *is* the metric; nothing measures the judge. The systematic study in `2606.19544` (21 judges, ~541,000 judgments) finds that exact-match agreement, the metric practitioners default to, overstates discriminative ability, with kappa deflation of 33-41pp against Cohen's kappa on MT-Bench, and that judge rankings shift by up to 14 positions across benchmarks. An uncalibrated judge is an unknown-magnitude bias, not a small one.

**Answerer and judge are the same model.** `config.py:17-18` sets `DEFAULT_ANSWER_MODEL = DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"`, and every manifest confirms both fields are that value. Self-preference bias is measurable and directional (`2410.21819`, `2604.22891`); the mechanism proposed in `2410.21819` is that judges over-reward low-perplexity, familiar-looking text, which is precisely the text their own decoder produces. The `Seven years.` → `Seven years (as of September 2023).` flip in section 2.1 is consistent with the verbosity-flavoured version of this.

**The judge never sees the retrieved context.** `build_judge_messages` passes question, gold, prediction and nothing else (`prompts.py:86-90`); `evaluate.py:205-207` calls it with no context argument. So the judge cannot assess groundedness at all. A prediction that is correct by luck or by parametric knowledge of the base model scores identically to one derived from retrieved evidence. For a memory product this is the wrong thing to be blind to: the entire value proposition is that the answer came from the stored graph.

`temperature=0` (`judge.py:68-73`) is set, which helps, but it does not make the judge unbiased and it does not make it reproducible across provider-side model updates, since `deepseek-v4-flash` is a moving alias with no pinned snapshot.

### 5. Failure attribution stops one step short of being actionable — and the missing step reverses the current priority

`answerer_gap` (`metrics.py:211-219`) is a scalar difference of two aggregate rates. It cannot tell you *which* questions failed for which reason, and it can go negative when the `gold_in_context` proxy under-counts, which it does.

Because `AnswerRecord` persists the full context per row, a proper joint decomposition is computable offline today. Assigning each question to the first failing stage — evidence session never retrieved, or retrieved but gold string absent from the assembled context, or gold present and the answer still wrong:

| run | evidence session missing | gold not in context | generation or judge | correct | context ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 60.9% | 0.7% | 0.0% | 38.4% | 21.9% |
| v5 | 61.2% | 0.7% | 0.0% | 38.2% | 32.9% |
| v6 | 24.3% | 0.7% | 0.0% | 75.0% | 62.5% |
| push75-a | 11.8% | 1.3% | 14.5% | 72.4% | 90.8% |
| push75-b | 9.9% | 1.3% | 13.2% | 75.7% | 90.8% |
| push75-c | 2.0% | 3.9% | 11.2% | 82.9% | 88.8% |

This changes the picture materially. Retrieval coverage went from missing the evidence session on 61% of questions to missing it on 2%. That work was real, large, and is the true story of `v5` → `push75-c`. But **as of `push75-c`, 17 of the 26 remaining failures — 65% — are on the generation-or-judge side of the boundary, not the retrieval side.** The evidence was in the context and the pipeline still did not convert it.

Meanwhile the tuning between `push75-a` and `push75-c` was entirely retrieval-side (`use_ppr` off→on, `max_passages` 8→16, `max_facts` 40→50, `sufficiency_retry` off→on) and cost 93% more tokens (4.01M → 7.74M) to push a component that was already at a 91% ceiling. Without this table on the report, effort keeps flowing to the stage that is already nearly saturated.

Two caveats I want to be explicit about. First, the boundary between the last two columns is only as good as `gold_in_context`, and that proxy is weak: `metrics.py:138` uses `token in blob`, a raw substring test with no word boundary, so gold token `may` matches `maybe` and `2022` matches `20225`. Second, the proxy under-counts as well as over-counts, because the judge accepts paraphrase that shares no gold tokens — visible in `v3`, where accuracy (38.4%) exceeds the measured ceiling (21.9%), which is impossible if the proxy were sound. Treat the split as directionally right and numerically soft; the fix is claim-level entailment rather than substring matching, per section 4.4.

### 6. Adversarial questions — 22% of LoCoMo, and the only test of abstention — are silently excluded, and would be scored wrong if included

`iter_qa_jobs` skips category 5 when `skip_adversarial=True` (`evaluate.py:104,113`), which is the default and is set in every recorded run. So abstention is not measured at all. That matters because abstention is one of the five core abilities LongMemEval was built to isolate (`2410.10813`), and for a memory product "I don't have that" is a correctness requirement, not a nicety.

If the flag is ever flipped, the numbers will be wrong rather than merely absent. In `locomo10.json`, 444 of the 446 category-5 items carry their gold under the key `adversarial_answer`, not `answer`:

```
key signatures across 1,986 QA items:
  1540  ('answer', 'category', 'evidence', 'question')
   444  ('adversarial_answer', 'category', 'evidence', 'question')
     2  ('adversarial_answer', 'answer', 'category', 'evidence', 'question')
```

`evaluate.py:177` reads only `qa.get("answer")`, so gold becomes `""` for those 444. Then `f1_score` returns 1.0 when both prediction and gold are empty (`metrics.py:29-30`) and `bleu1_score` does the same (`metrics.py:45-46`) — an empty prediction scores a perfect 1.0 — while the judge is handed a blank `Gold answer:` line and asked to apply the rule at `prompts.py:24`, which it cannot evaluate without knowing the question was adversarial by design.

This is not a local mistake. `2604.10981` documents the identical defect in the upstream LoCoMo reference implementation, describing "an empty-gold scoring bug in the LOCOMO reference implementation that renders 23% of its corpus unscorable by construction" — matching the 444/1,986 = 22.4% measured here. The bug was inherited, and `--skip-adversarial` currently masks it.

### 7. A run cannot be reproduced from what the run records

The manifest captures `run_id`, `command`, `samples`, `concurrency`, `skip_adversarial`, `categories`, `limit`, `brainapi_url`, `answer_model`, `judge_model`, `historical_limit`, `max_passages`, `max_facts`, `apply_fact_filter`, `use_ppr`, `sufficiency_retry`, and timestamps. Missing:

- **Git SHA.** Nothing ties a run to the code that produced it. The `push75-c` → `push75-d` prompt change is invisible in the manifest, which is why that pair is indistinguishable in the recorded metadata despite being the cleanest demonstration of prompt fitting in the repo.
- **Prompt hash.** Same problem, one level finer. `ANSWER_SYSTEM` and `JUDGE_SYSTEM` are load-bearing measurement instruments and are not versioned.
- **Dataset digest.** `dataset_path` is not hashed. A re-download that changes upstream is undetectable.
- **Model pinning.** `deepseek-v4-flash` is an alias. `temperature=0` is set (`answer.py:42-46`, `judge.py:68-73`) but no seed is requested and no provider snapshot is recorded, so run-to-run and month-to-month drift are both unbounded and invisible.
- **The graph under test.** This is the largest hole. `push75-a/b/c/d`, `v4-context`, and `checkpoint-a` have **no `ingest.jsonl` at all** — they evaluate against a brain that some earlier run populated. The ingestion configuration, extraction model, and graph contents in effect are recorded nowhere in those runs. `v5` (38.2%) and `v6` (75.0%) each did re-ingest, so the single largest accuracy jump in the project's history spans a re-ingestion whose parameters are only partly recoverable.
- **Manifest schema stability.** `checkpoint-a` predates the retrieval flags entirely, so cross-run comparison requires knowing what the code defaulted to at that moment.

There is also a live default-drift hazard between the harness and the server that I should flag as an observation rather than a finding, since retrieval internals are a sibling workstream's scope: the uncommitted diff to `src/services/api/controllers/retrieve.py` reads `getattr(request, "use_ppr", True)` while the Pydantic model in `src/services/api/constants/requests.py` declares `use_ppr: bool = False`. The Pydantic default wins for any client that omits the field. Which means "what the server does by default" and "what the benchmark measured" can diverge without either side erroring.

### 8. Duplicate rows are counted, and a fully failed run renders as a clean report

`completed_qa_keys` (`evaluate.py:89-97`) de-duplicates *jobs* on resume, so `--resume` is safe. But `build_report` reads every non-errored row with no de-duplication (`report.py:34`). The author knew to solve this for ingestion — `_latest_ingest_rows` (`report.py:17-26`) keeps the last status per unit, with the comment "resume appends retries" — and the same treatment was never applied to answers.

`checkpoint-a` is the proof: 176 rows, 20 errored, 156 scored, but only **152 unique** `(sample_id, qa_index)` pairs. Four questions are counted twice, all in the temporal category, which is why that run reports `n=156` with `cat2 n=41` where every other run has 37. And because `report.py:34` drops errored rows silently, that run — in which every single answer failed, 0.0% across all four categories, 82K tokens against the 4M+ of a healthy run — renders as a complete, plausible-looking report with a valid `n`. Nothing in `report.md` says "this run is broken." The only warning path (`report.py:60-63`) triggers on ingest failures, and that run has no ingest rows.

Running `report` mid-flight has the same shape: it will happily aggregate a partial `answers.jsonl` and print a headline.

### 9. Cost is measured for the harness, not for the product

`total_llm_tokens` sums `answer_total_tokens` and `judge_total_tokens` only (`metrics.py:205-209`, labelled "answer+judge" at `report.py:129`). So:

- **Judge tokens are counted as system cost.** They are evaluation overhead and inflate the figure.
- **Server-side tokens are invisible.** Ingestion agent swarm calls, and the `fact_filter` LLM calls on the retrieval path, appear nowhere. `invoke_loop.py:51` logs usage to stdout and drops it. The uncommitted diff to `src/core/search/fact_filter.py` changes work on the retrieval path whose token cost the benchmark structurally cannot see.
- **Ingestion cost is not measured at all.** `IngestRecord` (`ingest.py:22-36`) has two latencies and no token or cost field.
- **There is no cost-per-correct-answer.** From `push75-a` to `push75-c`, accuracy rose 14% relative while harness tokens rose 93%. Cost per correct answer went from roughly 36K to 61K tokens, a 68% increase, and no reported metric shows this.
- **Denominators disagree.** Latency and tokens aggregate over all rows (`metrics.py:200-209`) while accuracy uses the non-adversarial subset (`metrics.py:196`). Identical today because category 5 is always skipped; wrong the moment it isn't.

### 10. Latency is reported but not governed, and the stated budget is being violated

`00-scope-and-constraints.md` fixes the budget for `/retrieve/context` at **sub-second, fast and cheap**. Measured p50 has been between 4.0 and 4.8 seconds in every run since `v6`, with p95 between 5.4 and 6.9 seconds. `push75-c` and `push75-d`, the two best-scoring configurations, are also the two slowest since `v5`.

The harness reports these numbers accurately and nothing acts on them, so accuracy tuning has been silently spending a budget the project has already declared fixed. Two things are worth separating here. Enabling `use_ppr` and `sufficiency_retry` for a benchmark is a legitimate experiment. Reporting the resulting accuracy as *the* headline, without a gate that fails when p95 exceeds the documented budget, is what turns an experiment into a misleading claim.

A measurement detail makes this worse rather than better: `client.py:54-78` resets its timer inside the retry loop, so `retrieve_latency_ms` records the duration of the last attempt only, excluding earlier attempts and backoff sleeps. Reported latency therefore under-states user-visible latency whenever a retry occurred.

### 11. Evidence-session recall is coarser than the evidence annotation it consumes

LoCoMo annotates evidence at dialogue-turn granularity (`D1:12`). `evidence_session_ids` (`metrics.py:93-103`) reduces `D1:12` to `session_1`, and `retrieved_session_ids` falls back to regex-scraping session ids out of the context blob (`metrics.py:106-121`). So a run that retrieves the right session but the wrong turn inside it scores full evidence recall. This is why `evidence_session_recall_full` reads 97.3% for `push75-c` while the finer-grained context ceiling is 88.8% — the coarse metric is the more flattering one, and it is the one on the report's headline.

### 12. The synergies endpoint has no metric, and one advertised feature does not exist

Ranked below the harness issues because it is a smaller surface, but it is the clearest instance of "unmeasured means unknown." `polarity` is accepted, documented in both the route (`routes/retrieve.py:515`) and the retriever (`entity_sibilings.py:65`), and never read (`entity_sibilings.py:53`). A test suite for this endpoint would have caught that on day one. The scoring constants (`entity_sibilings.py:26,29,32,35`) were chosen without any offline metric to choose them against, and the unbounded return (`entity_sibilings.py:437`) means response size is a function of graph density.

### 13. Agent-facing usefulness is unmeasured, and the tool surface makes failure ambiguous

The product goal is to be a substrate for smarter agents. Nothing measures whether an agent holding these five MCP tools completes tasks better than one without them. The `"Unauthorized"` string returns (`main.py:212,266,288,321`) are a measurement problem as much as an ergonomics one: a bare string is indistinguishable from a legitimate empty result, so an eval harness cannot classify tool-call outcomes without string matching.

### 14. No CI gate exists

`staging.yaml:15` is a placeholder. 25 `unittest` files run only when a human runs them. No coverage measurement. Nothing prevents a change that drops multi-hop accuracy 20pp from shipping, and given section 2.3, nothing would reliably detect it even if the benchmark were run, because a 20pp drop on n=32 sits inside the noise band of an unpaired comparison.

## Open questions for the maintainer

1. Were the conv-26-specific rules in `ANSWER_SYSTEM` (`prompts.py:9,13,14`) written by reading conv-26's gold answers, and are you willing to delete them and re-baseline, accepting that the headline will drop?
2. Should the LoCoMo headline be defined on a held-out split of conversations that prompts are never tuned against, and if so which conversations do you want reserved?
3. Is 9.9% of the dataset (conv-26 only) a deliberate cost decision, and what per-run token budget would let us move to all ten conversations?
4. Is `deepseek-v4-flash` acceptable as both answerer and judge, or should the judge move to a different model family to remove self-preference bias?
5. Would you fund a one-time human labelling pass — I estimate 200-300 items, a few hours — to calibrate the judge, given that without it every accuracy number has unknown bias?
6. Is `skip_adversarial=True` a deliberate scoping choice or an artefact, and do you want abstention to count toward the headline once the `adversarial_answer` key is read correctly?
7. Is the sub-second budget for `/retrieve/context` in `00-scope-and-constraints.md` still binding, given measured p50 of 4.0-4.8s in every run since `v6`?
8. Should `use_ppr` and `sufficiency_retry` be treated as the intended production default (in which case the latency budget needs restating) or as benchmark-only settings (in which case the headline should be reported with them off)?
9. Which single number do you want CI to gate on, and what drop are you willing to block a merge for?
10. Should the CI gate run against a live BrainAPI plus a live LLM, or against recorded fixtures, given that the former makes CI cost money and flake on provider outages?
11. Is `polarity` on the synergies endpoint (`entity_sibilings.py:53`) intended to work, and if so what does "opposite" mean concretely enough to write a test for?
12. What is a synergy *for* — a user-facing "you might also care about X", or an internal retrieval expansion signal — since the two need different metrics?
13. Do you have any source of ground-truth judgement about synergy quality (your own labels on a few entities would be enough) or must the first metric be fully unsupervised?
14. Which downstream agent task should define "the memory made the agent better", since we need one concrete task to measure MCP usefulness at all?
15. Is it acceptable for the benchmark to require a fresh ingestion per run — roughly doubling wall-clock — in exchange for runs being reproducible from their own manifest?
16. Should ingestion token cost be plumbed into the API response so the harness can see it, or captured server-side and correlated by run id?

## Frontier techniques

### 4.1 Validate the judge before trusting it: kappa, not exact match

**Mechanism.** Sample a stratified subset of scored questions, have a human label correct/incorrect, then report chance-corrected agreement (Cohen's kappa) between judge and human rather than raw agreement. Separately audit the judge for position and consistency by re-running it with swapped or repeated inputs. `2606.19544` distils this into a "Minimum Viable Validation Protocol" over three protocols: agreement, consistency, bias audit.

**arXiv.** `2606.19544` (Reliability without Validity), 21 judges, ~541,000 judgments. Supporting: `2406.07791` (position bias; introduces repetition stability, position consistency, preference fairness), `2410.21819` and `2604.22891` (self-preference bias quantification; the latter reduces it 31.5% via structured multi-dimensional evaluation).

**Reported gain.** Not an accuracy gain — a validity gain. The headline finding is that exact-match agreement overstates discriminative ability with kappa deflation of 33-41pp on MT-Bench, and that judge rankings shift by up to 14 positions across benchmarks. Two production-deployed judges combined test-retest reliability above 0.95 with position bias above 0.10, so a stable judge is not necessarily an unbiased one.

**Cost.** One human labelling pass of 200-300 items. Then a re-scoring script; no new model. Ongoing cost is one extra judge call per audited item if you add swap-consistency checks.

**Fit.** Direct. The harness already persists `judge_raw` and `judge_reason` per row, so a labelling UI can be a CSV export. Because the judge here is pointwise (not pairwise) with a fixed field order, classical position bias does not apply; the relevant biases are self-preference and the verbosity effect visible in section 2.1.

**Verdict: adopt.** This is the precondition for every other number in this document meaning anything.

### 4.2 Component-wise memory evaluation instead of end-to-end-only

**Mechanism.** Decompose the memory system into representation/storage, extraction, retrieval/routing, and maintenance, then measure each module separately with ablations, rather than reporting a single end-to-end QA score. `2606.24775` does exactly this across 12 memory systems and 11 datasets and argues that end-to-end F1/BLEU treats the system as a monolithic black box, hiding operational cost, architectural trade-offs, and robustness under knowledge updates.

**arXiv.** `2606.24775` (Are We Ready For An Agent-Native Memory System?). Complementary industrial template: `2607.13157` (Oracle Agent Memory), which explicitly complements downstream task accuracy with "memory-centric measures such as evidence retrieval, recall, latency, and estimated token use" — the exact quartet missing from `report.md` on the server side.

**Reported gain.** `2606.24775`'s finding that no single architecture dominates, and that effectiveness depends on whether the memory structure matches the workload bottleneck, is the useful one here: it is the general form of section 2.5's result that BrainAPI's bottleneck has already moved from retrieval to generation. Also finds localized maintenance is more cost-efficient than global reorganization.

**Cost.** Reporting-layer work plus server-side token accounting. No new models.

**Fit.** Strong, and partially already built — `answerer_gap` is a one-dimensional version of this. The blocker is that server-side token usage is logged to stdout (`invoke_loop.py:51`) instead of being returned or traced.

**Verdict: adopt.** This is the framing for the whole measurement stack in section 5.

### 4.3 Add LongMemEval to isolate the five abilities LoCoMo blurs together

**Mechanism.** 500 curated questions over scalable chat histories, targeting five separable abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. Also supplies a three-stage decomposition — indexing, retrieval, reading — for attributing failures.

**arXiv.** `2410.10813` (LongMemEval). Successor for agent environments: `2605.12493` (LongMemEval-V2, 451 questions over trajectories up to 115M tokens, reporting an accuracy-latency Pareto frontier rather than accuracy alone).

**Reported gain.** Commercial assistants and long-context LLMs drop ~30% accuracy on sustained interactions, so the benchmark has headroom where LoCoMo is closer to saturation for this system (single-hop already 89%).

**Why it matters here specifically.** Two of the five abilities are directly on the project's critical path and are currently unmeasured or barely measured: **knowledge updates** is the whole subject of workstream 05 (superseded facts returning as current truth) and LoCoMo has no category for it; **abstention** is skipped entirely (section 2.6). Adding LongMemEval buys measurement of temporal truth, which is otherwise being worked on blind.

**Cost.** A new ingestion path and a new adapter — the largest single item in the plan. Histories are large; expect ingestion cost well above conv-26. Mitigate by starting with the small history configuration.

**Fit.** Good. It is the same shape of task (conversational history in, QA out) so `evaluate.py` needs an adapter, not a rewrite.

**Verdict: adopt, phase 3.** Do it after the harness is trustworthy; adding a second benchmark to an uncalibrated harness doubles the unreliable surface.

### 4.4 Claim-level faithfulness and groundedness in place of substring matching

**Mechanism.** Decompose an answer into atomic claims and check each against the retrieved context with an entailment model, yielding reference-free faithfulness. RAGAS scores faithfulness, context relevance, and answer relevance without ground truth (`2309.15217`). ARES trains lightweight LM judges on synthetic data and — the important part — corrects their errors with prediction-powered inference against a few hundred human annotations (`2311.09476`).

**arXiv.** `2309.15217` (RAGAS), `2311.09476` (ARES), `2601.04196` (RAGVUE, which decomposes into retrieval quality, answer relevance and completeness, strict claim-level faithfulness, and *judge calibration* as a first-class metric, and reports surfacing failures RAGAS misses).

**Reported gain.** ARES evaluates accurately across eight KILT/SuperGLUE/AIS tasks with only a few hundred human annotations, and its judges survive domain shift. For this project the gain is that the section 2.5 attribution boundary stops depending on `token in blob`.

**Cost.** An entailment or judge call per claim. Too expensive to run on every question every time; run it on the failure set and on a fixed sampled subset.

**Fit.** Very good. `text_context`, `source_passages`, and `historical_context` are already persisted per row, so faithfulness can be computed offline from existing `answers.jsonl` files with no re-run.

**Verdict: adopt the faithfulness metric; adopt ARES's prediction-powered inference idea for judge calibration.** Reject the framing that reference-free metrics can replace the gold-answer judge here — LoCoMo has gold answers and they are the stronger signal; use faithfulness as the *second* axis that catches right-for-the-wrong-reason.

### 4.5 Treat multi-hop scores as suspect until shortcut-resistance is tested

**Mechanism.** Establish how much of a multi-hop score survives when the shortcut is removed: run a single-hop-only baseline, and evaluate on adversarial or debiased variants where word-matching the question to one sentence no longer works.

**arXiv.** `1906.02900` (Compositional Questions Do Not Necessitate Multi-hop Reasoning) is canonical: a single-hop BERT model reaches 67 F1 on HotpotQA, comparable to multi-hop models of the time, and humans answer over 80% of questions without being shown all the required paragraphs. `1906.07132` constructs adversarial documents that contradict the shortcut without invalidating the gold answer, and strong baselines drop significantly. `2302.05963` builds four debiased datasets over 2WikiMultiHopQA and HotpotQA-small and shows underlying-reasoning supervision reduces shortcut reliance but does not confer robustness to sub-questions or inverted questions.

**Reported gain.** Diagnostic. The relevant number is how much a score falls when the shortcut is closed.

**Fit and the honest caveat.** LoCoMo's category 1 is "multi-hop" by annotator intent, and the evidence annotations name multiple turns, but nothing verifies that answering actually required combining them. With 32 questions and a context window holding 16 passages plus 40-50 facts, a single passage containing the answer is entirely plausible for a meaningful fraction. A cheap check exists and needs no new data: for each multi-hop question, count how many distinct annotated evidence turns are strictly necessary by re-answering with each one ablated.

**Verdict: adapt.** Do not import HotpotQA — different domain, and the retrieval corpus assumption (Wikipedia paragraphs) does not match a conversational event graph. Do import the methodology as a leave-one-evidence-out ablation on LoCoMo's own multi-hop set.

### 4.6 Multi-hop datasets: which to add and which to refuse

**Adopt, later: MuSiQue-style compositional construction and MultiHop-RAG.** Rejected for now on cost grounds, not merit — a second QA benchmark before the judge is calibrated multiplies unreliable surface.

**Adopt the design principle from `2605.12361` (MedHopQA) immediately, at zero cost.** Its anti-gaming construction embeds 1,000 scored questions inside a public set of 10,000 with answers withheld, explicitly treating saturation resistance and contamination resistance as design constraints. The transferable idea: **a held-out set whose gold answers the prompt author has not read.** That single practice would have prevented section 2.1.

**Reject `2412.17032` (MINTQA)** for this workstream: it targets new and long-tail *world* knowledge in the model's parametric memory, whereas BrainAPI's task is recall from a graph built from supplied conversations. The failure mode it isolates is not the one here.

**Reject `1601.02789`-style n-gram metric surveys and the reliance on BLEU/F1 generally as headline signals.** They remain useful as cheap regression tripwires — a large F1 drop with flat judge accuracy means the answer *format* changed, which is exactly the signal that would have flagged section 2.1 — but they should not be optimized. Note `mean_f1` is 0.474 while judge accuracy is 82.9% for `push75-c`; the two are nearly decoupled.

### 4.7 Attribution and citation verification, for the "did it come from memory" question

**Mechanism.** Require the answer to cite the passage or triple that supports it, then verify the citation independently. `2509.21557` separates generation-time citation (one pass) from post-hoc citation (draft, then attach and verify) and finds retrieval is the dominant driver of attribution quality in both, with post-hoc achieving high coverage at competitive correctness and moderate latency. `2510.11394` (VeriCite) verifies claims with an NLI model, selects supporting evidence, then refines.

**arXiv.** `2509.21557`, `2510.11394`, `2408.12398` (metrics vs humans on fine-grained citation support), `2606.28358` (mechanistic account of when models cite).

**Reported gain.** VeriCite improves citation quality while maintaining answer correctness across five open-source LLMs and four datasets.

**Two caveats I will not paper over.** `2408.12398` finds no single faithfulness metric excels across all evaluations, and that the best metrics specifically struggle to distinguish *partial* support from full or none — which is the interesting case for multi-hop. `2606.28358` shows citation behaviour is a distributed "attributional ensemble" partly disconnected from the model's actual reasoning path, concluding inline citations "can create a false sense of security." So citations are a useful measurement instrument and a weak guarantee.

**Fit.** Good for the deep-navigation surface and for offline evaluation. Not for `/retrieve/context`, whose sub-second budget forbids an extra verification pass.

**Verdict: adopt for evaluation only.** Compute attribution offline on stored rows. Do not add a verification hop to the hot path.

### 4.8 Intrinsic knowledge-graph construction quality, to measure ingestion without gold graphs

**Mechanism.** Score an automatically extracted graph against an "ideal" graph derived from the source text: entity-level completeness, resolution quality, and connectivity; relation-level predicate preservation and multiplicity via lexical similarity, dependency-parse alignment, and negation handling. `2607.10212` (KGCQual) does this, and validates that its scores correlate significantly with downstream link-prediction performance on the same extracted graphs — which is what makes it more than a vanity metric.

**arXiv.** `2607.10212` (KGCQual), `2502.05239` (KG construction evaluation emphasising hallucination and omission with BERTScore graph similarity), `2605.05476` (a dual-purpose benchmark with an expert-curated reference graph as an upper bound, designed to separate "is it the model or the graph").

**Reported gain.** `2607.10212` reliably identifies omissions, redundancy, and structural deviations that existing metrics overlook, on WebNLG, TinyButMighty, and BenchIE.

**Fit.** Conceptually strong: this is the missing ingestion-side metric, and it needs no gold graph. Practically it needs adaptation — these tools target sentence-level triple extraction from encyclopedic text, whereas BrainAPI produces event-centric structures from dialogue with speaker attribution and time. The transferable core is the **omission/hallucination split**: for a session, what fraction of the entities and relations a reader would extract are present, and what fraction of extracted triples are unsupported.

**Verdict: adapt, phase 4.** The cheap version is a fixed 3-5 session gold-annotation set, hand-built once, scored on entity recall and triple precision. Reject wholesale adoption of the published scorers; their linguistic assumptions do not transfer.

### 4.9 Recommendation quality for the synergies endpoint

**Mechanism.** Score a ranked list on beyond-accuracy axes — diversity, novelty, serendipity, coverage — instead of relevance alone. Serendipity is the one that matters for "non-obvious connections", and the validated decomposition is unexpectedness plus relevance plus timeliness plus user curiosity.

**arXiv.** `1906.11431` (User Validation of Recommendation Serendipity Metrics) is the load-bearing citation: over 10,000 users of real feedback, finding that user-profile-based and especially content-based unexpectedness metrics outperform popularity-based ones, and that the full four-component metric indicates serendipity more accurately than any subset. Also `2310.02294` (beyond-accuracy review for GNN recommenders), `2307.14951` (Widespread Flaws in Offline Evaluation), `2309.05892` (report distributions, not point estimates).

**LLM-as-judge for serendipity: adopt with a stated ceiling.** `2508.17571` proposes a universally applicable LLM-based serendipity evaluator, finding chain-of-thought prompting most accurate and, notably, that no serendipity-oriented recommender consistently beat a general one across three datasets. But `2507.17290`'s meta-evaluation against real user studies reports a best-case Pearson correlation of **21.5%** with human judgement. That is a weak instrument. It is still better than the current state, which is no instrument, but it must not be used to gate merges.

**Reject** simulation-based evaluation (`2209.08642`) and prequential streaming evaluation (`1504.08175`): both assume an interaction log and a user-feedback loop that BrainAPI does not have.

**Verdict: adopt the unsupervised structural metrics first** (intra-list diversity, catalogue coverage over entity types, graph distance from the seed entity as an unexpectedness proxy) because they are deterministic, free, and gate-able. Add the LLM serendipity judge as a **reported, non-gating** diagnostic. Fix `polarity` before either, since a parameter that does nothing cannot be measured.

### 4.10 Agent memory utility as downstream task success

**Mechanism.** Define a concrete agent task, run it with and without memory access, and measure task completion. `2605.12493` (LongMemEval-V2) formalizes this for environment experience across five abilities — static state recall, dynamic state tracking, workflow knowledge, environment gotchas, premise awareness — using a context-gathering formulation where the memory system consumes history and returns compact evidence for downstream QA.

**arXiv.** `2605.12493`, and `2607.13157` for the industrial pattern of pairing downstream accuracy with memory-centric measures.

**Reported gain.** Best method 72.5% average accuracy versus 48.5% for the strongest RAG baseline, at substantial latency cost — the paper is explicit that it advances an accuracy-latency Pareto frontier rather than a single number, which is the right reporting shape for BrainAPI's two-tier architecture.

**Fit.** The context-gathering formulation maps almost exactly onto BrainAPI's MCP surface: tools consume the graph and return compact evidence. The obstacle is that BrainAPI has no defined downstream agent task, which is open question 14.

**Verdict: adapt, phase 5, and keep it small.** A 20-30 task suite exercising the five MCP tools, scored on completion with and without memory, plus tool-call success rate and tokens per completed task. Reject building a full agent benchmark; the goal is a tripwire showing the tools are usable, not a leaderboard.

### 4.11 Paired significance testing

**Mechanism.** Use the test that matches the design. For paired binary outcomes on identical items, that is McNemar's exact test, not a comparison of two independent proportions.

**arXiv.** `1809.01448` (Recommended Statistical Significance Tests for NLP Tasks), the appendix to the Dror et al. hitchhiker's guide, which maps task and measure to the valid test. Supporting cautionary evidence on underpowered protocols: `2305.01633` finds only 13% of surveyed NLP human evaluations had low enough barriers and enough information to even attempt reproduction.

**Reported gain.** Free statistical power. Section 2.3 is the demonstration: the same recorded rows, read with the correct test, reverse the conclusion about which steps were real.

**Cost.** About 30 lines of Python. No new data, no new models, no new runs.

**Verdict: adopt first.** Highest ratio of decision quality to effort in this entire document.

## Proposed measurement stack

### Benchmarks and datasets

| Layer | Dataset | Measures | Status | Priority |
| --- | --- | --- | --- | --- |
| Smoke | conv-26, 20 fixed questions | pipeline liveness, latency | exists, needs pinning | P0 |
| Headline | LoCoMo, **4 tuning + 6 held-out conversations** | multi-hop, temporal, open-domain, single-hop, abstention | exists but single-conversation and prompt-contaminated | P0 |
| Abstention | LoCoMo category 5 with `adversarial_answer` read correctly | refusal correctness | broken, masked by a flag | P1 |
| Update / temporal truth | LongMemEval (`2410.10813`) | knowledge updates, multi-session reasoning | absent | P2 |
| Ingestion quality | 3-5 hand-annotated sessions | entity recall, triple precision, omission, hallucination | absent | P3 |
| Recommendations | 20-30 seed entities, structural metrics | diversity, coverage, unexpectedness | absent | P3 |
| Agent utility | 20-30 MCP tasks, with/without memory | task completion, tool-call success | absent | P4 |

The tuning/held-out split is the single most important structural change. Prompts, thresholds, and scoring constants may be tuned on the 4 tuning conversations. The headline is reported on the 6 held-out ones, whose gold answers nobody who edits a prompt has read. Concretely: keep conv-26 in the tuning set (it is already contaminated and cannot be decontaminated), and never tune against the rest.

### Per-component metrics

Ingestion, per session: units submitted / completed / failed; extraction latency; **LLM tokens and cost**; entities and triples written; entity-resolution merge count; omission and hallucination rate against the annotated sessions.

Retrieval, per query: evidence recall at **turn** granularity, not just session (`metrics.py:93-103` is too coarse); context precision, the fraction of returned passages containing any evidence turn; the gold-in-context ceiling, computed by entailment rather than substring (`metrics.py:138`); passages and facts returned; **latency measured across all retry attempts** (`client.py:54-78` currently measures only the last); server-side filter tokens.

Generation, per answer: judge correctness; claim-level faithfulness against the retrieved context; abstention correctness split into correct refusal, wrong refusal, and hallucinated answer; F1 and BLEU-1 retained as *format-drift tripwires only*.

Judge, per release: Cohen's kappa against human labels; self-consistency across repeated calls; swap consistency; the human-label set version.

Run-level: cost per correct answer; total server plus harness tokens; p50 and p95 end-to-end latency; and the full reproducibility tuple — git SHA, prompt hashes, dataset digest, model snapshots, ingestion manifest reference.

### The failure-attribution breakdown

Every question lands in exactly one bucket, evaluated in order, so the buckets sum to 100% and each names one owning workstream:

| # | Bucket | Test | Owner |
| --- | --- | --- | --- |
| A | Not ingested | no annotated evidence turn exists in the graph | ingestion (01) |
| B | Ingested, not retrieved | evidence turn in graph, absent from result set | retrieval (02) |
| C | Retrieved, dropped in assembly | evidence in result set, absent from final context blob | retrieval (02) |
| D | In context, generation failed | context entails gold, answer wrong | generation / prompt |
| E | Answer correct, judge disagreed | human says correct, judge says wrong | judge calibration |
| F | Correct but ungrounded | answer correct, faithfulness fails | measurement validity |
| G | Correct and grounded | — | — |

The current harness collapses A+B+C into one number and D+E+F into another (`answerer_gap`). Splitting A from B is what tells you whether extraction or ranking is at fault. Splitting E out is what stops judge noise being read as a quality regression. Splitting F out is what catches right-for-the-wrong-reason, which for a memory product is a failure even when the score says success.

Report it as a stacked breakdown per category, since section 2.5 shows the mix differs sharply by category, and the aggregate hides that.

### What the CI regression gate asserts

Three tiers, because a benchmark that costs money and needs a live LLM cannot run on every push.

**Tier 1 — every push, no network, target under 2 minutes.**
- All 25 `unittest` files pass. This is the entire content of `staging.yaml:15` today and it is unimplemented.
- `selftest-metrics` passes (`metrics.py:248-275`), extended with cases that currently fail by omission: empty gold must not score F1 = 1.0; `gold_in_context` must respect word boundaries; duplicate rows must not be double-counted.
- Prompt-hash check: if `ANSWER_SYSTEM` or `JUDGE_SYSTEM` changed, the job fails with an instruction to re-baseline. A prompt change is a measurement-instrument change and must not pass silently.
- Static check: the synergies scoring constants and any declared-but-unread parameter (`entity_sibilings.py:53`) are covered by a test.

**Tier 2 — every pull request, live server, roughly 15 minutes, one held-out conversation.**
- Headline accuracy on the held-out conversation has not dropped by a margin that is **significant under McNemar's exact test at p < 0.05** against the stored baseline. Not "has not dropped 2pp" — section 2.3 shows raw thresholds at this n are noise.
- No per-category regression significant at p < 0.05, evaluated on multi-hop and temporal specifically, since those are the declared priorities.
- p95 retrieval latency for `/retrieve/context` is under the documented budget. Given measured p95 of 5.4-6.9s against a stated sub-second target, this gate must start as a **warning with a recorded value** and become blocking once the budget question (open question 7) is settled. A gate that fails on day one gets disabled.
- Cost per correct answer has not risen more than 20% without an explicit override label on the PR. This would have flagged `push75-a` → `push75-c`, which spent 93% more tokens for a 14% relative accuracy gain.
- Zero errored rows, and scored row count equals unique question count. This catches both the `checkpoint-a` class of silent failure and the duplicate-counting bug.
- Abstention correctness has not regressed, once category 5 is scored correctly.

**Tier 3 — nightly or on release tags, full dataset.**
- All 10 conversations, all categories, full attribution breakdown A-G.
- Judge kappa against the frozen human label set is above an agreed floor. If kappa drops, the *judge* regressed, and no accuracy number from that run is admissible.
- Faithfulness on a fixed sampled subset.
- Synergies structural metrics and MCP task-completion suite.
- The full reproducibility tuple recorded, and the run reproducible from its own manifest.

The gate's most important property is what it refuses to do: it does not assert that a number went up. It asserts that a number did not go **significantly** down, that the cost of any gain is visible, and that the measurement instrument itself has not changed underneath the comparison.

## Implementation plan

Sized S (1-2 files) or M (3-5 files). Every task has an acceptance criterion and a command. No task changes retrieval, extraction, or storage behaviour — this workstream measures, it does not tune. Note that `benchmarks/` currently has uncommitted changes; phase 0 assumes those are committed or stashed first so that a baseline is attributable to a SHA.

### Phase 0 — Stop the measurement from lying (no new data, no new models)

**0.1 Record the reproducibility tuple in every manifest.** S — `benchmarks/locomo/cli.py`, `benchmarks/locomo/ingest.py`.
Add git SHA, dirty-tree flag, SHA-256 of `ANSWER_SYSTEM` and `JUDGE_SYSTEM`, SHA-256 of the dataset file, and a reference to the ingest run that populated the brain.
*Acceptance:* a fresh `evaluate` manifest contains all six fields; a run started with a dirty tree is marked.
*Verify:* `python -m locomo evaluate --samples conv-26 --limit 5 --run-id repro-check && python -c "import json;m=json.load(open('benchmarks/runs/repro-check/manifest.json'));assert all(k in m for k in ('git_sha','dirty','answer_prompt_sha','judge_prompt_sha','dataset_sha','ingest_run_id'))"`

**0.2 De-duplicate answers in the report and fail loudly on broken runs.** S — `benchmarks/locomo/report.py`.
Apply the `_latest_ingest_rows` treatment (`report.py:17-26`) to answers keyed on `(sample_id, qa_index)`. Add warnings when the error count is above zero, when duplicates were dropped, or when scored rows are below the expected job count.
*Acceptance:* re-reporting `locomo-conv26-checkpoint-a` yields n=152 not 156, and `report.md` carries a warning naming 20 errored rows.
*Verify:* `python -m locomo report --run-id locomo-conv26-checkpoint-a && grep -c "Warning" benchmarks/runs/locomo-conv26-checkpoint-a/report.md`

**0.3 Add paired significance testing between runs.** S — `benchmarks/locomo/metrics.py`, `benchmarks/locomo/cli.py`.
A `compare` command taking two run ids, emitting McNemar exact p overall and per category, plus the flipped-question lists in both directions.
*Acceptance:* `compare push75-a push75-b` reports p ≈ 0.36 and flags the difference as not significant; `compare push75-a push75-c` reports p ≈ 0.007 and significant.
*Verify:* `python -m locomo compare --baseline locomo-conv26-push75-a --candidate locomo-conv26-push75-b`

**0.4 Fix the adversarial gold key and the empty-gold scoring path.** S — `benchmarks/locomo/evaluate.py`, `benchmarks/locomo/metrics.py`.
Read `adversarial_answer` when `answer` is absent; mark the record as an abstention item; make F1 and BLEU-1 return `None` rather than 1.0 for empty gold (`metrics.py:29-30,45-46`); score abstention items on refusal correctness instead.
*Acceptance:* `--no-skip-adversarial` on 10 category-5 questions produces no F1 = 1.0 rows and an `abstention_accuracy` figure.
*Verify:* `python -m locomo evaluate --samples conv-26 --categories 5 --no-skip-adversarial --limit 10 --run-id adv-check && python -m locomo report --run-id adv-check`

**0.5 Quarantine the contaminated prompt and re-baseline.** S — `benchmarks/locomo/prompts.py`.
Remove the conv-26-derived specifics from `ANSWER_SYSTEM` (`prompts.py:9,13,14`): the `Melanie's slipper` example, the charity-race Sunday rule, the `"Likely no; ..."` hedge template, and the enumeration keyword list that mirrors conv-26 gold answers. Keep the generalizable rules ("never return an empty answer", "prefer relative phrasing when the dialogue uses it", "do not abstain when thematically related evidence exists").
*Acceptance:* no string in `prompts.py` appears in any conv-26 gold answer; the re-baselined score is recorded as the new reference with its drop documented.
*Verify:* a script asserting that no gold answer from `locomo10.json` shares a 4-gram with `ANSWER_SYSTEM`, then `python -m locomo evaluate --samples conv-26 --run-id baseline-clean`

**CHECKPOINT 0.** Re-report all nine historical runs with the fixed reporting path, and publish a corrected history table with McNemar p-values between consecutive runs. Expect the headline to fall from 86.2%. That drop is not a regression; it is the removal of a measurement error. Do not proceed until the maintainer has seen and accepted the corrected baseline — this is where open questions 1, 2, and 6 must be answered.

### Phase 1 — Make the judge an instrument instead of an assumption

Depends on phase 0.

**1.1 Export a human labelling set.** S — `benchmarks/locomo/cli.py`.
A `label-export` command emitting a CSV stratified by category and judge verdict, with question, gold, prediction, judge verdict, judge reason, and a blank human column. Target 250 items weighted toward disagreement-prone categories.
*Acceptance:* CSV has 250 rows with proportional category representation and no leaked judge verdict in the column a human fills.
*Verify:* `python -m locomo label-export --run-id baseline-clean --n 250 --out benchmarks/runs/labels/round1.csv`

**1.2 Compute judge agreement.** S — `benchmarks/locomo/metrics.py`, `benchmarks/locomo/cli.py`.
Ingest completed labels; report raw agreement, Cohen's kappa, per-category kappa, and a confusion matrix. Report both raw and kappa side by side, since `2606.19544` shows the gap between them is the interesting quantity.
*Acceptance:* `judge-validate` prints kappa with a CI and flags any category where kappa is below 0.6.
*Verify:* `python -m locomo judge-validate --labels benchmarks/runs/labels/round1.csv`

**1.3 Audit judge self-consistency and self-preference.** S — `benchmarks/locomo/judge.py`, `benchmarks/locomo/cli.py`.
Re-judge a 100-item subset three times to measure flip rate; re-judge with a different model family to quantify the self-preference delta, given answerer and judge are currently the same model (`config.py:17-18`).
*Acceptance:* a reported flip rate and a cross-family accuracy delta; if the delta exceeds 3pp, the judge model is changed and phase 0's baseline re-run.
*Verify:* `python -m locomo judge-audit --run-id baseline-clean --repeats 3 --alt-model <other-family>`

**CHECKPOINT 1.** If kappa is below 0.6 in any category, that category's historical numbers are not admissible evidence and must be labelled as such in `CHECKPOINT_NOTES.md`. Answers open questions 4 and 5.

### Phase 2 — Attribution and honest cost

Depends on phase 1 for the E bucket.

**2.1 Turn-level evidence recall.** S — `benchmarks/locomo/metrics.py`.
Track `D<session>:<turn>` rather than collapsing to `session_N` (`metrics.py:93-103`), and require the harness to record which turns appear in the returned passages.
*Acceptance:* turn-level recall is reported alongside session-level and is lower than it, confirming the coarse metric was optimistic.
*Verify:* `python -m locomo report --run-id baseline-clean && grep "evidence_turn_recall" benchmarks/runs/baseline-clean/report.json`

**2.2 Entailment-based context ceiling.** M — `benchmarks/locomo/metrics.py`, plus a new judge-side helper.
Replace the `token in blob` substring test (`metrics.py:138`) with a claim-level entailment check on gold against the retrieved context, following RAGAS/RAGVUE (`2309.15217`, `2601.04196`). Retain the substring version as a cheap fallback and report both, so the divergence is visible.
*Acceptance:* the ceiling no longer falls below measured accuracy on any historical run — the `v3` impossibility in section 2.5 disappears.
*Verify:* recompute on `v3` and assert ceiling ≥ accuracy.

**2.3 Full A-G attribution breakdown.** M — `benchmarks/locomo/metrics.py`, `benchmarks/locomo/report.py`.
Implement the seven-bucket assignment, per category, summing to 100%. Bucket A needs a graph-presence probe, which is a new client call; if that endpoint does not exist, merge A into B and say so explicitly in the report rather than guessing.
*Acceptance:* buckets sum to the row count for every historical run; the `push75-c` split reproduces roughly 2% / 4% / 11% for the retrieval, assembly, and generation stages.
*Verify:* `python -m locomo report --run-id locomo-conv26-push75-c && python -c "..."` asserting the buckets sum.

**2.4 Cost per correct answer, including server-side tokens.** M — `benchmarks/locomo/metrics.py`, `benchmarks/locomo/report.py`, `benchmarks/locomo/ingest.py`.
Separate judge tokens from product tokens; add ingestion tokens to `IngestRecord` (`ingest.py:22-36`); report cost per correct answer. Server-side tokens require plumbing from `invoke_loop.py:51`, which is a `src/` change and therefore out of this workstream's scope — until then, report the harness-side figure explicitly labelled as a **lower bound**, not as total cost.
*Acceptance:* `report.md` shows product tokens, judge tokens, and cost per correct answer separately, and the `push75-a` → `push75-c` comparison shows the 68% cost-per-correct increase.
*Verify:* `python -m locomo report --run-id locomo-conv26-push75-c && grep "cost_per_correct" benchmarks/runs/locomo-conv26-push75-c/report.json`

**2.5 Fix the latency measurement.** S — `benchmarks/locomo/client.py`.
Measure from first attempt to final response including backoff (`client.py:54-78`), and report attempt count.
*Acceptance:* a run with an induced retry shows latency exceeding single-attempt duration and a retry count above zero.
*Verify:* `python -m locomo smoke --samples conv-26` against a server returning one 429.

**CHECKPOINT 2.** Publish the attribution breakdown for the corrected baseline and re-rank the sibling workstreams' priorities against it. If the generation bucket still dominates as it does in `push75-c`, the retrieval workstream's next task list needs revisiting before it is executed.

### Phase 3 — Scale the evidence base

Depends on phase 2 so that new data lands in a trustworthy harness.

**3.1 Tuning / held-out conversation split.** S — `benchmarks/locomo/config.py`, `benchmarks/locomo/cli.py`.
Declare 4 tuning conversations (including conv-26) and 6 held-out. Refuse to report a headline from tuning conversations without an explicit flag.
*Acceptance:* `evaluate` on a tuning conversation prints a banner that the result is not a headline; the held-out path forbids prompt-hash mismatch against the recorded baseline.
*Verify:* `python -m locomo evaluate --split tuning --limit 5` and confirm the banner.

**3.2 Full-dataset run.** M — orchestration only.
All 10 conversations, all categories including abstention. This is roughly 13× current volume; at `push75-c`'s rate that is on the order of 75-100M harness tokens, so the cost question (open question 3) must be answered first.
*Acceptance:* a run with n ≈ 1,986, per-category n at least 96, and per-category CI widths under 12pp.
*Verify:* `python -m locomo evaluate --all-samples --no-skip-adversarial --run-id full-v1`

**3.3 Multi-hop shortcut ablation.** M — new module under `benchmarks/locomo/`.
For each multi-hop question, re-answer with each annotated evidence turn ablated, and record how many turns are strictly necessary, per `1906.02900` and `1906.07132`.
*Acceptance:* a distribution of necessary-evidence counts; the fraction of "multi-hop" questions answerable from a single turn is reported.
*Verify:* `python -m locomo shortcut-probe --run-id full-v1 --categories 1`

**3.4 LongMemEval adapter.** M — new dataset adapter plus an ingestion path.
Start with the smallest history configuration. Map its five abilities onto the existing category machinery.
*Acceptance:* a LongMemEval run produces a report with per-ability breakdown, including knowledge-updates, which LoCoMo cannot measure at all.
*Verify:* `python -m locomo evaluate --dataset longmemeval --limit 50 --run-id lme-smoke`

**CHECKPOINT 3.** With full-dataset CIs and the shortcut probe in hand, the multi-hop and temporal claims can be stated with real confidence for the first time. This is the earliest point at which "the memory layer is X% accurate" is a defensible sentence.

### Phase 4 — Gate it

Depends on phase 3 for stable baselines. Tier 1 can land after phase 0.

**4.1 Tier 1 gate.** S — `.github/workflows/`.
Replace `staging.yaml:15` with a real job: `unittest` discovery, `selftest-metrics`, the prompt-hash check, and a no-gold-4-gram-in-prompt assertion.
*Acceptance:* a PR that edits `ANSWER_SYSTEM` fails with a re-baseline instruction; a PR that breaks a metric function fails.
*Verify:* `python -m unittest discover -s tests -v && python -m locomo selftest-metrics`

**4.2 Tier 2 gate.** M — workflow plus a comparison harness.
One held-out conversation against a live server, asserting no significant McNemar regression overall or on multi-hop and temporal, plus zero errored rows, unique-equals-scored, and cost-per-correct within 20%. Latency starts as a recorded warning.
*Acceptance:* an artificially degraded retrieval config fails the gate; a no-op PR passes; the job stays under 20 minutes.
*Verify:* `python -m locomo gate --baseline <ref> --candidate <run> --tier 2`

**4.3 Tier 3 nightly.** S — workflow.
Full dataset, attribution breakdown, judge kappa floor, faithfulness sample.
*Acceptance:* nightly artefact contains all sections; a judge-kappa drop marks the run inadmissible rather than reporting an accuracy delta.
*Verify:* manual dispatch of the workflow.

**CHECKPOINT 4.** Every subsequent workstream's claims are now falsifiable in CI. This is the point at which the roadmap in `00-scope-and-constraints.md` becomes provable rather than plausible.

### Phase 5 — The unmeasured surfaces

Depends on phase 4 only for CI plumbing; can run in parallel with phase 3.

**5.1 Synergies test suite.** S — `tests/`.
First tests for `EntitySinergyRetriever`, covering the scoring constants (`entity_sibilings.py:26,29,32,35`) and asserting the `polarity` contract. The `polarity` test will fail against current behaviour (`entity_sibilings.py:53`); land it as an expected failure with a reference to open question 11 rather than deleting the assertion.
*Acceptance:* tests exist and run in tier 1; the polarity gap is recorded as a known failure, not hidden.
*Verify:* `python -m unittest tests.test_entity_synergies -v`

**5.2 Synergies structural metrics.** M — new benchmark module.
20-30 seed entities; report intra-list diversity, catalogue coverage over entity types, mean graph distance from seed as an unexpectedness proxy, and result-set size distribution (relevant given the unbounded return at `entity_sibilings.py:437`).
*Acceptance:* a report with all four, and a stable baseline for a future gate.
*Verify:* `python -m locomo synergy-report --brain locomoconv26 --seeds benchmarks/data/synergy_seeds.json`

**5.3 LLM serendipity diagnostic.** S — extends 5.2.
Chain-of-thought serendipity judging per `2508.17571`, reported and explicitly **non-gating**, annotated with the 21.5% human correlation ceiling from `2507.17290` so nobody later mistakes it for ground truth.
*Acceptance:* the metric appears in the report with its stated ceiling next to it.
*Verify:* `python -m locomo synergy-report --with-llm-judge`

**5.4 MCP agent-task suite.** M — new module.
20-30 tasks over the five tools, run with and without memory access. Report task completion, tool-call success rate, and tokens per completed task.
*Acceptance:* a with-versus-without delta; ambiguous `"Unauthorized"` string returns (`main.py:212,266,288,321`) are counted separately from empty results and reported as a measurability defect.
*Verify:* `python -m locomo agent-eval --tasks benchmarks/data/mcp_tasks.json`

**5.5 Ingestion quality set.** M — new annotation plus scorer.
Hand-annotate entities and relations for 3-5 sessions; score entity recall, triple precision, omission, and hallucination, adapting KGCQual's split (`2607.10212`).
*Acceptance:* a per-session scorecard, and a demonstration that the scores separate two different extraction configurations.
*Verify:* `python -m locomo kg-quality --sessions benchmarks/data/kg_gold/`

### Ranked by expected value per unit cost

| Rank | Task | Cost | Why |
| --- | --- | --- | --- |
| 1 | 0.3 paired significance | S | Reverses conclusions about past work using data already on disk |
| 2 | 0.5 decontaminate the prompt | S | Without it the headline is not a memory metric |
| 3 | 0.2 de-duplicate and fail loudly | S | A fully broken run currently renders as a clean report |
| 4 | 0.1 reproducibility tuple | S | Nothing is attributable to code today |
| 5 | 2.3 attribution breakdown | M | Redirects every other workstream's effort |
| 6 | 1.1-1.2 judge calibration | S+S | Bounds the error on every number |
| 7 | 3.1 tuning/held-out split | S | Structurally prevents recurrence of finding 2.1 |
| 8 | 4.1-4.2 CI gates | S+M | Makes all of the above permanent |
| 9 | 3.2 full dataset | M | Halves CI widths; largest token cost |
| 10 | 2.4 cost per correct answer | M | Reveals cost regressions currently invisible |
| 11 | 3.4 LongMemEval | M | Only way to measure knowledge updates |
| 12 | 5.1-5.2 synergies | S+M | Unblocks an unmeasured product surface |
| 13 | 5.4 MCP agent tasks | M | Highest ambiguity about what to measure |
| 14 | 5.5 ingestion quality | M | Most annotation effort per unit of insight |

Explicitly **not worth it here**, despite being fashionable: adding HotpotQA or MuSiQue as headline benchmarks (wrong domain, and a second unreliable surface before the judge is calibrated); simulation-based recommender evaluation (`2209.08642`, no interaction log exists); inline citation verification on the `/retrieve/context` hot path (`2606.28358` shows citations are a weak guarantee, and the sub-second budget forbids the extra pass); and replacing the gold-answer judge with reference-free RAGAS metrics (LoCoMo has gold answers, which are the stronger signal — faithfulness belongs alongside, not instead).

## Risks

**Re-baselining will look like a large regression.** Phase 0 removes prompt contamination, counts duplicates correctly, and scores abstention. The headline will fall from 86.2%, possibly substantially. *Detection and mitigation:* publish the corrected history table at checkpoint 0 with the drop attributed line by line to each fix, and record in `benchmarks/runs/CHECKPOINT_NOTES.md` that pre-phase-0 numbers are not comparable to post-phase-0 numbers. The risk is not the drop; it is someone later comparing across the boundary.

**The judge may fail calibration, invalidating the recorded history.** If kappa comes back below 0.6 for multi-hop or temporal, the categories the project cares most about have no usable history. *Detection:* phase 1.2. *Mitigation:* label 250 items before committing to any accuracy target, and treat the labelled set as a permanent asset that is re-used at every judge change.

**A stricter gate gets disabled rather than satisfied.** A latency gate that fails on day one — which it would, at p95 of 5.4-6.9s against a sub-second budget — invites a bypass that then covers real regressions. *Mitigation:* latency starts as a recorded warning; only accuracy and correctness gates block initially; every override requires an explicit PR label so bypasses are countable.

**Full-dataset runs may be unaffordable.** Extrapolating `push75-c`, 10 conversations is on the order of 75-100M tokens per full run. *Detection:* cost per correct answer from 2.4. *Mitigation:* tier 2 uses one held-out conversation; the full run is nightly or release-only; the judge is the cheapest model that passes calibration.

**Attribution may be miscalibrated and misdirect effort.** The A-G buckets depend on a graph-presence probe (bucket A) that may not have an endpoint, and on entailment quality (buckets C/D). A wrong split sends sibling workstreams after the wrong stage — precisely the failure this document accuses the current `answerer_gap` of. *Detection:* cross-check the split against manual inspection of 30 failures. *Mitigation:* if the probe does not exist, merge A into B and label it as merged in the report, rather than inventing a boundary.

**Measuring more surfaces raises maintenance cost enough that measurement stops.** Six benchmark families is a lot for one project. *Mitigation:* only tiers 1 and 2 are on the critical path of a PR; everything else is nightly, and a nightly suite that breaks does not block shipping. Keep the synergies and MCP suites at 20-30 items — deliberately small tripwires, not leaderboards.

**A live-server gate makes CI flaky.** Tier 2 depends on a running BrainAPI and an LLM provider. *Detection:* track gate failure causes separately from gate failures. *Mitigation:* distinguish infrastructure failure from quality failure in the exit code, retry infrastructure failures, and never let an infrastructure failure read as a quality pass.

**Optimizing the new metrics instead of the product.** Faithfulness, diversity, and serendipity are all gameable, and the pattern this document documents in section 2.1 — tuning a prompt until a metric moves — will recur. *Mitigation:* the tuning/held-out split from 3.1 is the structural defence, and it must apply to every new metric, not just to accuracy.

**The uncommitted working tree makes the current state unattributable.** Seven harness files and three `src/` files are modified but uncommitted, and `push75-d` was produced from that tree. *Detection:* the dirty-tree flag from 0.1. *Mitigation:* commit or stash before establishing any baseline; treat every existing run in `benchmarks/runs/` as provenance-unknown until re-run.
