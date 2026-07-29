# Cross-Workstream Roadmap

Synthesis of `01`-`05`, ranked. Constraints in `00-scope-and-constraints.md` are binding. Every claim below was verified against the code or the run artifacts; where a claim could not be verified it is marked as such.

## Reconciliation with ADR-005

`docs/decisions/005-hybrid-one-shot-context-retrieval.md` (Accepted, 2026-07-28) was written by the maintainer during this research and post-dates the analysts' reading of ADRs 001-004, all four of which are empty stubs. **ADR-005 is the authoritative statement of the current retrieval design and it already settles several points this roadmap's source documents treated as open.** Read it before acting on Phases 1-3. Specifically:

- **The one-shot vs agentic question is decided**, not open: keep the one-shot `GetContext` API and enrich a single call with hybrid retrieval; agentic navigation "stays open for workloads where one sufficiency retry is not enough; it is not the default retrieval model." The retrieval analyst reached the same conclusion independently, which is corroboration rather than new information.
- **The answerer prompt changes were deliberate and documented** (decision item 7), not inadvertent leakage. The maintainer explicitly rejected changing the *judge* prompt on the grounds that it "breaks comparability with v5/v6 and hides real system quality," while treating answerer-side enumeration, temporal phrasing, and hedged open-domain rules as legitimate tuning. That distinction is defensible in principle; the problem is narrower than "leakage" and specific: several of the shipped rules encode conv-26 gold answer strings and one resolves a single question, which fits the test set regardless of intent. Task 0.2 stands, and the maintainer has authorised it.
- **The truncated-logs problem was already known.** Consequence: "Eval must log full `source_passages` / `historical_context` / `retrieved_session_ids`; truncated context logs misdiagnose retrieval." The 20,000-character cap nevertheless still truncates, and no `triples` field is logged at all — so Task 0.1 implements an intent the ADR already recorded rather than introducing a new one.
- **The ~5 s p50 was accepted knowingly** ("Latency and token cost rise with larger passage budgets and optional PPR/sufficiency"). Phase 3 changed that accepted decision: dossier deletion is now live at **~1.4 s p50**; ADR-005 should be superseded to reflect the new latency posture and the remaining ~400 ms gap to sub-second.

**ADR-005 line 29 is now stale.** It states that defaults keep `use_ppr` and `sufficiency_retry` off for API compatibility. Task 2.2 flipped `use_ppr` to `True` (with `sufficiency_retry` left off, because Phase 3 removes it from this path). The ADR needs an amendment or a superseding ADR; leaving it as-is documents a default the code no longer has.

## The finding that reordered everything

**The event graph — the product's entire differentiator — had never been measured** until Phase 0/1.

`benchmarks/locomo/evaluate.py` previously capped logged `text_context` at 20,000 characters, and `retrieve.py` wrote passages into that field before graph facts. Verified against `locomo-conv26-push75-d/answers.jsonl`: all 152 of 152 rows were exactly 20,000 characters, **0 of 152 contained a graph fact line**, and no `triples` field was logged at all.

Both quality metrics — `answerable_rate` and `evidence_session_recall_full` — were derived from that truncated field. They therefore measured the **passage channel only**. Every headline number in `benchmarks/runs/` through push75 was consistent with the event graph contributing nothing whatsoever.

**Phase 0 fixed the harness; Phase 1 answered the product question.** Graph EvR is logged (~53% point estimate on the clean brain) but **not yet a reliable metric** — identical-config runs agree on graph session recall for only ~39–42% of questions (passage recall stays deterministic). See Phase 0 follow-up below. The channel is visible and does **not** move judge accuracy above the resolution floor vs passages-only. Everything after Phase 1 is downstream of that result — see Phase 4 gate.

## What is already fixed

| Defect | Status |
| --- | --- |
| Event-leg supersession deprecating valid actor attributions (`ingestion.py:465-505`) | **Fixed**, 3 tests, 2 fail before / pass after. Existing brains **not** repaired — no backfill authorised. |
| Relative-date resolution inert; recency scorer permanently 1.0 | **Fixed**, 27 tests. LoCoMo timestamps and ISO variants now parse; recency varies instead of being pinned at 1.0. |
| Test suite aborted collection entirely (`retrieve.py:40` imports `kg_agent`, stale stub in `tests/test_architecture_refactors.py`) | **Fixed.** No tests ran at all before this; every earlier "141 passed" baseline was measured with collection interrupted. Suite now runs. |
| Unreachable config defaults + no-op fact filter (Task 2.2) | **Fixed.** `use_ppr` now defaults `True`; all `getattr` fallbacks removed so the schema is the only source of defaults; filter genuinely gated on a caller-supplied adapter. |
| Type-named placeholder super-nodes (Task 2.3) | **Fixed.** Five demonstrated patterns and the Scout instructions to emit Unit entities removed; quantities travel as the relationship `amount` property, which the Janitor already prescribed. Existing graphs still affected — repair pass described, not authorised. |

| Phase 0 harness (all six tasks) | **Done and verified against the live stack.** Graph channel logged, prompt de-fitted, paired McNemar, adversarial scoring, reproducible manifests, judge separable. |
| `benchmarks/locomo/client.py` dropped `use_ppr` when false, so with the new server default an arm configured `use_ppr=false` would silently run **with** PPR | **Fixed.** Both flags now sent unconditionally, so the manifest matches the wire. |
| Dossier channel on `/retrieve/context` (~79% of wall / ~3918 ms p50) | **Fixed and measured live** after sync+restart. Context path no longer imports/calls `EventSynergyRetriever` / `retrieve_matches` and does not emit `[dossier:` lines. Synergies retained on entity routes (`/entity/info`, `/entity/synergies`). Live p50 **~1.4 s** (was ~4.8 s). |

## The honest baseline

`baseline-clean-a` — same config as `push75-c/-d`, conv-26 gold rules removed from `ANSWER_SYSTEM`, measured on the **pre-fix** (`locomoconv26`) brain:

| | Prompt-fitted (`push75-d`) | De-fitted (`baseline-clean-a`) |
| --- | --- | --- |
| Headline | 86.2% | **82.2%** [75.4, 87.5] |
| Multi-hop | 88.6% | 71.9% |
| Temporal | 91.9% | 86.5% |
| Open-domain | — | 46.2% |
| Single-hop | — | 91.4% |
| Evidence recall (passages) | 97.3% | 97.3% |

Paired McNemar: 4 flipped right, 10 flipped wrong, **p = 0.18**. The drop is real as a point estimate and not statistically distinguishable at n=152 — which is the same resolution problem as before, now applied honestly. Evidence recall is identical, exactly as expected when only the prompt changed. Open-domain lost 3 and gained 0, consistent with removing the hedge template.

**Use 82.2% as the reference point** for the old brain. A `prompt-audit` command now fails if any 3-gram of `ANSWER_SYSTEM` appears in a gold answer, so the fitting cannot silently recur.

## Phase 1 — done (live, after sync+restart)

Clean brain `locomoconv26clean`: **19/19** conv-26 sessions ingested (not 35 — that was a miscount of `date_time` keys). Extraction fixes confirmed in-graph vs the old brain:

| Signal | Clean | Old |
| --- | --- | --- |
| Deprecated EVENT-touching edges | **0** | 219 |
| Type-named placeholders | **0** | 3 |
| Nodes / rels / events | **537 / 752 / 410** | 411 / 584 / 281 |

Eval arms (de-fitted prompt, same config as `baseline-clean-a`):

| Arm | Acc | Multi-hop | Graph EvR | Passage EvR | p50 ms |
| --- | --- | --- | --- | --- | --- |
| `baseline-clean-a` (old brain) | 82.2% | 71.9% | 0%\* | 97.3% | 4776 |
| `clean-brain-eval-a` | 83.6% | 78.1% | 54.0% | 96.7% | 1372 |
| `clean-brain-eval-b` | 81.6% | 71.9% | 52.0% | 96.7% | 1423 |
| `ablate-passages-a` (`max_facts=0`) | 82.2% | 75.0% | 0% | 96.7% | 1337 |
| `ablate-passages-b` | 81.6% | 81.2% | 0% | 97.3% | 1306 |

\*Old baseline graph EvR was unmeasurable due to logging, not necessarily empty.

McNemar all ns: full vs baseline p≈0.77–1.0; ablate vs full p≈0.77–1.0.

### Verdicts

1. **Extraction fixes earn their keep on graph hygiene**, not on detectable judge accuracy at n=152.
2. **The graph channel does not move judge accuracy above the resolution floor** vs passages-only. Graph evidence recall is logged (~53% point estimate) — the harness can see the channel; it is not yet converting into answers. Treat that EvR figure as provisional until fact selection is order-stable (below).
3. **Dossier deletion earns its keep on latency**: ~4.8 s → ~1.4 s p50, no accuracy loss. Sub-second is not yet met; residual is likely embedding fanout (Phase 3.2). Gap to budget ≈ **400 ms**.

### Follow-up findings (second clean-brain measurement agent)

1. **Graph EvR is non-deterministic across identical-config runs** — open defect / Phase 0 follow-up. Even with `graph_session_ids` emitted, identical-config runs on the same brain agree on graph session recall for only ~39–42% of questions. Passage recall stays deterministic. Until fact selection is made order-stable, **do not treat graph EvR as a reliable metric** — it is measurable but not yet a measurement.
2. **The clean graph is larger and more fragmented, which hurts under a fixed `max_facts` budget.** Clean: 537 nodes / 410 events vs old 411 / 281. Same `max_facts=50` therefore covers fewer distinct sessions. Extraction correctly stopped collapsing history; **fact selection / budget is now the binding constraint** if the graph channel is to pay off — a better next lever than more extraction fidelity *if* graph work continues.
3. **Process:** two agents ingested the same brain id / run directory concurrently; `ingest.jsonl` was interleaved and needed dedup. Detail: `benchmarks/runs/clean-brain-ingest/NOTES.md`. Recommend: one owner per brain id; the manifest should record the running server's code identity, not only the harness git SHA (a SHA is useless when the TUI install is a different checkout).
4. **Provenance correction:** both old and clean brains already have `source_chunk_ids` on 100% of nodes and relationships — re-ingest is not required for provenance stamping. See `benchmarks/runs/CHECKPOINT_NOTES.md`.

## Latency — measured, not inferred (2026-07-28)

### Pre-deletion attribution (old path)

Live measurement against `locomoconv26` at the push75-d config, n=20 questions, in-process, matching the harness p50 within noise (4813 ms vs 4835 ms). Instrumentation: `profile_stages=true` on `/retrieve/context` → `stage_timings`, or `TRACE_STAGE_PROFILER_ENABLED=true` for a `LATENCY` trace event. Off by default; measured overhead below noise.

| Stage | Claimed | Measured p50 | Share | Verdict |
| --- | --- | --- | --- | --- |
| dossiers | hundreds of per-edge fetches | **3918 ms** | **79%** | right, and understated |
| passages | 12–24 serial | 829 ms (682 embed) | 17% | cost is embeddings, not search |
| facts (concurrent) | depth 4 | 481 ms union / 2622 sum | 10% | correct |
| historical | concurrent | 355 ms, fully overlapped | 0% of wall | free; delete for cost only |
| PPR | loop-bound | 35.9 ms | 0.7% | irrelevant |
| spaCy | 2 passes on loop | 13.1 ms | 0.3% | irrelevant |
| sufficiency retry | depth 24 | **fired 0/20** | 0% | unpaid on this brain |
| fact filter | 0 | 0 | 0% | correct |

### Post-deletion (live, after sync+restart)

Dossier deletion is **deployed and measured** on the clean-brain eval arms: harness p50 **~1372–1423 ms** (full) / **~1306–1337 ms** (passages-only), versus **4776 ms** on the old-brain baseline that still paid for dossiers. That is a ~3.4 s win with no detectable accuracy loss. Remaining gap to the 1000 ms budget is **~300–400 ms**, consistent with the predicted embedding-fanout residual.

### Consequences for Phase 3

1. **Dossier channel deleted from `/retrieve/context` — done and live.** Predicted remaining ~895 ms; measured remaining ~1.4 s. The component's own docstring called its output "raw, unprecise," and `_MAX_DOSSIER_ENTITIES = 5` was unconditional. Dominant cost inside it was **10 serial embedding HTTP calls**, not the per-edge vector fetches.
2. **Next: batch embeddings across the fanout (Phase 3.2).** Passages alone make three serial ~200 ms embed calls. One shared embedding is the leading candidate for closing the remaining ~400 ms gap.
3. **Do not spend effort on spaCy dedup, PPR off-loop moves, or the sufficiency retry** for latency — combined they are under 50 ms, and the retry never fired in 20/20 questions at both configs, including the arm whose manifest claims `sufficiency_retry=true`. That means `push75-d` did not measure the knob its manifest says it varies.
4. **Event-loop blocking is ~50 ms p50 (~1%).** Nearly everything is I/O wait or off-loop Python. One exception worth fixing when touching the file: `_build_adjacency_from_seeds` runs a synchronous graph query on the loop (28.5 ms) — small alone, a queuing multiplier under concurrency.
5. **Historical context recovers cost, not latency** — 91.5% duplicate of source passages and fully overlapped.

Sub-second now depends on embedding batching, not further channel deletion. This brain is tiny (19 sessions); vector/Cypher times will grow with corpus size while embedding latency will not.

## Phase 1 blockers — resolved and live

1. **`max_facts=0` honouring.** `retrieve.py` uses `max(0, request.max_facts)`; schema `Field(40, ge=0)`. Verified by `tests/test_context_max_facts_and_provenance.py` (6/6) and by the live ablate arms (graph EvR = 0%).
2. **Graph session provenance.** Response exposes per-triple `source_chunk_ids` / `source_session_ids` and top-level `graph_session_ids`. Fact lines also get `(session_N)` annotations so the existing harness scraper can score graph recall. Live graph EvR ~52–54% on clean-brain full arms confirms the channel is logged and scored — but **cross-run agreement is only ~39–42%**, so the number is not yet trustworthy (Phase 0 follow-up: order-stable fact selection).
3. **Sync + restart.** Workspace → `~/.brainapi/source` sync and TUI/API restart completed before the clean-brain eval and ablation runs above.

## Dataset subtlety worth recording

LoCoMo's `adversarial_answer` is **not** a refusal marker. There are 439 distinct values across 446 questions, all substantive: it is the **trap** answer produced by accepting the question's false premise, usually attributing an event to the wrong speaker. So F1/BLEU overlap against it is an *inverted* signal — high overlap means the system fell for the trap. The harness now returns `None` for lexical metrics on adversarial rows and instructs the judge that correct means declining or correcting the premise. Any future work on abstention must not treat this field as a gold refusal string.

Both were found during research, not by the benchmark. That is itself the argument for Phase 0.

---

## Phase 0 — Make the benchmark able to detect a real change

Nothing after this phase is verifiable without it. Every task here is small and touches only `benchmarks/`.

### Task 0.1 — Log the graph channel

**Why:** without it the graph's contribution is unmeasurable, and the passages-only ablation is meaningless.
**Do:** log `triples` as its own field; either raise `_TEXT_CONTEXT_CAP` or write graph facts before passages; recompute `retrieved_session_ids` from the untruncated response.
**Acceptance:** a graph fact line appears in the logged artifact for >0 questions; `evidence_session_recall` is computable separately for the graph and passage channels.
**Verification:** re-run `evaluate` on the existing brain; assert both channels non-empty in `answers.jsonl`.
**Size:** S. **Depends on:** none.

### Task 0.2 — Delete the conv-26-fitted rules from `ANSWER_SYSTEM` and re-baseline

**Why:** the uncommitted `benchmarks/locomo/prompts.py` diff contains gold answer strings verbatim — `Keep speaker names explicit (e.g. "Melanie's slipper", not "my slipper")`, the phrasings `"the week of <date>"` and `"since <year>"`, the open-domain template `"Likely no; ..."`, and `For "last weekend" / charity race timing relative to a Thursday/Friday session, prefer the preceding Sunday over Saturday`. That last rule resolves one specific question. `README.md` forbids ingesting QA annotations because it leaks the test set; routing the same information through the answer prompt defeats that rule while observing its letter.

`push75-c` → `push75-d` is the proof: byte-identical manifests, same brain, same flags, same models, **only** the answer prompt differs, and accuracy moved 82.9% → 86.2%. The gap between those two runs is prompt fitting, not memory quality.
**Do:** remove question-specific rules; keep only format guidance that is defensible for an arbitrary conversation. Re-baseline and accept the drop.
**Acceptance:** no gold-answer string or single-question rule remains in the prompt; a new baseline is recorded.
**Verification:** diff review plus a fresh run recorded as the reference point.
**Size:** S. **Depends on:** none. **Expect the headline to fall.**

### Task 0.3 — Paired significance testing

**Why:** the harness compares independent Wilson intervals, discarding the pairing on identical question sets and most of the statistical power. **Passage** retrieval is deterministic — passage `evidence_session_recall` is bit-identical across independent runs within a config arm (`0.826667` in both a and b; `0.973333` in both c and d) — while 11-12% of questions flip correctness between identical-config runs at `temperature=0`. **Graph** session recall is *not* deterministic yet (~39–42% cross-run agreement on identical-config clean-brain arms); do not treat graph EvR deltas as signal until fact selection is order-stable. Judge deltas need a paired test.
**Do:** exact McNemar between runs; report flipped-right and flipped-wrong counts, not just the delta.
**Acceptance:** every run comparison emits a p-value and a flip table.
**Verification:** reproduce the known results — `push75-b`→`push75-c` p≈0.06 single-run; pooled two-run arms p≈9×10⁻⁴.
**Size:** S. **Depends on:** none.

### Task 0.4 — Fix adversarial scoring, then decide whether to enable it

**Why:** 444 of 446 adversarial questions carry gold under `adversarial_answer`, but `evaluate.py:177` reads only `answer`; `metrics.py:29-30` returns F1 = 1.0 for empty prediction against empty gold. Abstention — the only defence against confident wrong answers, and the thing a memory product is judged on in production — is both unscorable and skipped by default.
**Acceptance:** adversarial questions score correctly against `adversarial_answer`; empty-vs-empty no longer scores 1.0.
**Size:** S. **Depends on:** none.

### Task 0.5 — Make runs reproducible

**Why:** no manifest records a git SHA, prompt hash, dataset digest, or pinned model, and `push75-a/b/c/d` have no `ingest.jsonl`, so the graph under test is recorded nowhere. Separately `report.py:34` does not de-duplicate, so `checkpoint-a` scored 156 rows from 152 questions — and that run, 0.0% across every category with every answer failing, rendered as a clean plausible report. Concurrent agents sharing a brain id / run directory also interleaved `ingest.jsonl` (see `benchmarks/runs/clean-brain-ingest/NOTES.md`).
**Acceptance:** manifest captures harness SHA **and the running server's code identity** (TUI install may be a different checkout — harness SHA alone is useless), prompt hash, dataset digest, model IDs; one owner per brain id; a run that fails wholesale is visibly distinguishable from a run that scored badly.
**Size:** S.

### Checkpoint 0
- Graph and passage channels measured separately
- Prompt no longer fitted to conv-26; honest baseline recorded
- Paired tests in place; flip counts reported
- **Do not proceed until the honest baseline exists.** Everything after this is judged against it.

---

## Phase 1 — Answer the product question — **DONE**

### Task 1.1 — Passages-only ablation on a re-ingested clean brain — **DONE**

**Why:** this decides whether the graph earns its cost. It had to run on a **freshly ingested** brain: all four `push75` runs showed `ingest.n = 0`, so every one reused a pre-fix brain carrying wrongly-deprecated actor edges.
**Done:** re-ingested `locomoconv26clean` (19/19 sessions), then compared `max_facts=0` against the full config with everything else fixed (de-fitted prompt, same flags as `baseline-clean-a`). Two runs per accuracy arm.
**Result:** no contribution detectable above the resolution floor — McNemar ablate vs full p≈0.77–1.0; judge accuracy within noise of the old-brain baseline. Graph EvR logged at ~53% but cross-run agreement only ~39–42% — not yet a reliable metric. Full table and verdicts in the Phase 1 results section above.

**This result hardens the Phase 4 gate** (see below): graph design investment is no longer justified by LoCoMo judge accuracy alone. If graph work continues, prefer fact-selection / `max_facts` budget over further extraction fidelity.

---

## Phase 2 — Confirmed defects with cheap fixes, independent of measurement

Ranked by gain per unit of cost. None require the benchmark to be trustworthy, because each is a provable defect rather than a quality hypothesis.

### Task 2.1 — Stop billing passages three times
The same passage text reaches the answerer three times: inlined into `text_context` (`retrieve.py:1057`), again as `## Source passages` (`prompts.py:56-59`), and again as `## Historical context` (`prompts.py:64-67`), where 91.5% of entries are byte-identical to a source passage. 183,978 characters per question where 61,955 suffice — a **66.3% cut with no information loss**, accounting for essentially the entire 51.4k-token prompt. Improves cost, latency, and the budget simultaneously. **Size:** S. **Highest gain per hour in this document.**

### Task 2.2 — Make the intended config reachable, and fix the no-op filter
`use_ppr` and `sufficiency_retry` are declared `False` in `GetContextRequestBody` while `retrieve.py:898` and `:967` read them as `getattr(request, ..., True)` — Pydantic always materialises the field, so the `True` fallback is dead and the better config is off for every HTTP caller. Separately `filter_relevant_facts` is called at `retrieve.py:917-921` with no `llm_adapter`, which under the current diff returns every index, making `apply_fact_filter=True` provably identical to the `else` branch it guards; the LLM prompt inside it never executes. Decide intent, then make the code say it. **Size:** S.

### Task 2.3 — Fix the type-name super-node collapse
`src/constants/prompts/architect_agent.py:111` instructs the model to emit `{"type": "MONEY", "name": "Money"}`. Node identity is `sha256(lower(name)|lower(type))` (`src/core/saving/identity.py:13-25`), so **every monetary amount ever ingested collapses onto one shared node**, manufacturing false multi-hop paths between unrelated facts. Verified: `stable_node_id('Money','MONEY')` is constant. This actively fabricates connections, which is worse than missing them for a product whose promise is a traceable path. **Size:** S-M.

### Task 2.4 — Index the graph, and stop silently skipping the vector index
Nothing in the repository creates a single Neo4j index or constraint; every uuid lookup is a full scan, twice per relationship write, using bracket syntax that would defeat a property index if one existed. And `_vector_index_ddl` returns no DDL above 2000 dimensions (`src/lib/postgresql/vectors.py:62-63`) while `.env.example:43-47` ships 3072, so no HNSW index is ever created on a default install. Decide the embedding question by recall-vs-latency measurement per `00`; `halfvec` indexes up to 4000 dims and is the option that keeps 3072. **Size:** S each.

### Task 2.5 — Give the Janitor a veto
Relationships are queued for persistence at `ArchitectAgentCreateRelationshipTool.py:550-551` and only *then* is `wrong_relationships` checked, so a rejected edge **and** its replacement are both written. And `run_atomic_janitor` returns `"OK"` on success and `None` on parse failure while callers use `getattr`, making a crashed Janitor indistinguishable from an approving one. The validation loop the README advertises does not currently gate anything. **Size:** M.

### Checkpoint 2
- Token cost per question down ~66% with accuracy unchanged within the resolution floor
- No shared type-level super-nodes; false path count measurable
- A failed Janitor is distinguishable from an approving one

---

## Phase 3 — Meet the latency budget (context path)

`/retrieve/context` must be sub-second. **Pre-deletion** measured p50 was 4053–4793 ms; **post-deletion live** p50 is **~1.4 s** on clean-brain full arms (~1.3 s passages-only). Gap to budget ≈ **400 ms**. Latency does not scale with query length (r = 0.089), so bounding fanout is a cost win, not a latency win.

Dossier deletion (the predicted dominant win) is **done and measured**. The remaining path is embedding fanout plus a handful of serial I/O operations.

- **3.1** Instrument per-stage timing — **done**. Live attribution confirmed dossiers at 79% of wall; post-deletion residual matches the embedding-fanout prediction.
- **3.2** Batch embeddings across the fanout (leading residual); optionally move to write time: PPR adjacency table, per-entity gists, passage compression. **This is the remaining work to close ~400 ms.**
- **3.3** Move to the deep/MCP tier: the sufficiency **retry** (keep the cheap lexical check on the context path, but only to return an explicit insufficiency signal to the caller — which also closes the "silently returns weak context" gap), listwise LLM and cross-encoder reranking, beam search, and all iterative retrieve-then-reason.
- **3.4** Context-tier ranker: a linear or small-GBDT model over features already computed — vector distance, PPR score, recency, provenance overlap, variant strength — trained on labels in `answers.jsonl`. Microseconds, no model server. A cross-encoder costs 50-200 ms on CPU for a component that is not the bottleneck; it is the deep-tier escalation path, not the context-path answer.

**Checkpoint 3:** p50 under 1000 ms with per-category evidence recall held. Dossier deletion got from ~4.8 s to ~1.4 s; embedding batching is the next gate. If the budget cannot be met, escalate with numbers rather than quietly missing it.

---

## Phase 4 — Design work, gated on Phase 1 (tension now explicit)

Phase 1 reported: **the graph channel does not move LoCoMo judge accuracy above the resolution floor** vs passages-only (McNemar ns). Graph design investment is therefore **harder to justify on LoCoMo judge accuracy alone.**

The remaining case for Phase 4 work is not "the ablation proved the graph wins on LoCoMo," but:

1. **Stabilize then raise graph EvR** — first make fact selection order-stable (Phase 0 follow-up; current ~39–42% cross-run agreement makes the ~53% figure provisional), then lift coverage under the fixed `max_facts` budget (clean graph is larger/more fragmented, so 50 facts cover fewer sessions). Converting retrieval into answers remains open.
2. **Multi-hop / composition where passages fail** — product priority #1; LoCoMo multi-hop moved within noise, not past the floor.
3. **Non-LoCoMo product surfaces** — synergies, entity routes, agent substrates — where the graph is the product, not an optional evidence channel.

**Active GraphRAG track:** `07-event-graphrag-improvement-plan.md` — event-hub-first plan (determinism → diversified `max_facts` / typed PPR → tiered cross-event composition per ADR-006). Prefer that doc for retrieval GraphRAG task breakdown; do not start 4.x as a LoCoMo accuracy programme. Start only when one of (a)–(c) is the explicit goal, and measure that goal directly.

**SOTA LoCoMo track:** `08-sota-locomo-protocol.md` — dual product/SOTA profiles to compete with HyperMem/Mem0/APEX-MEM (≥93% LLM-as-judge on full LoCoMo10). Generation + hierarchical topic retrieval; not more GraphRAG EvR alone.

**Phase A (2026-07-29):** order-stable fact selection + harness agreement. Live: graph-session stability **99.3%** on `graph-stable-a/b`; graph EvR measurable at **47.3%** (`graph-stable-a` p50 ~1351 ms).

**Phase B (2026-07-29):** hub/session MMR, intra-hub completeness, typed PPR, path-shaped single-hub format, optional `temporal_conflicts`. Live: stability held; graph EvR **flat 47.3%** vs graph-stable-a — **Checkpoint B EvR lift NOT met**. Sequential p50 ~948 ms. Conclusion recorded in `07`: re-ranking the same intra-hub pool cannot raise session coverage; Phase C cross-event hub-bridge expansion is required.

**Phase C (2026-07-29):** write-time hub-bridge index + context ≤1-bridge expansion (backfill 13 660 bridges on `locomoconv26clean`). Live: stability **98.7%**; graph EvR **47.3–48.0%** (noise vs 47.3%; +1 question on arm a); cat1 still 25%. Sequential p50 ~1340–1391 ms. **Checkpoint C EvR lift NOT met.** Novel-session bridge ranking (C2) also null on EvR (+~500 ms). Diagnostic: 91% of incomplete QAs are 1-bridge-reachable; curated set never kept the missing session → retention, not substrate reachability.

**Phase C3 + ANN (2026-07-29):** reserved curated slots for novel-session bridge facts (`reserve_hub_cap=12`) **plus** ANN seed stability (over-fetch+uuid order, seed stabilize, 3dp distance quantization). **Dual gate PASS:** agreement **99.3%**, graph EvR **56.0%** (was 47.3%), cat1 **40.6%** (was 25%), p50 ~1320 ms. Detail and ship setting in `07` Checkpoint C3.

**Judge secondary (same day):** full evaluate `phase-c3-judge-a/b` → **77.6% / 78.3%** (multi-hop **65.6%** both). Graph EvR **56.0%** and agreement **99.3%** held on full evaluate. McNemar pooled vs `baseline-clean-a`: 9↑/15↓ p=0.31; vs `graph-stable-a`: 10↑/16↓ p=0.33 — **ns**. EvR lift did **not** convert to answers (G4); answerer gap ~18–19%.

**Checkpoint D.1 (2026-07-29):** paths into answerer on `locomoconv26clean` (mf=50, `use_ppr`) → **80.9% / 80.3%** acc (multi-hop 71.9% / 75.0%); Graph EvR **56.0%** held; A↔B **100%**; `prompt-audit` passed. Pooled **80.6%** vs C3 **78.0%**: McNemar 17↑/9↓ p=0.17 ns; vs `baseline-clean-a` 82.2%: 10↑/16↓ p=0.33 ns. **Soft convert** (directionally up, not significant); answerer gap ~16%. Measured with harness `enrich_paths_from_triples` after TUI restart was initially blocked; later sync+restart confirmed native `_paths_for_curated` legs. Detail in `07`. **Next (user fork — do not auto-start D.2):** D.2 feature reranker **vs** Checkpoint D product-surface write-up (graph value = EvR/provenance, not LoCoMo judge).
- **4.1 Bitemporal representation.** Store validity as typed, range-queryable properties on the event hub plus an appended `SUPERSEDES` edge, with a `statement_key` and a `cardinality` marker so an interval closes only when the claim is genuinely single-valued and the existing assertion is strictly older. This fixes the remaining over-firing the hotfix did not cover — there is still no cardinality model, so a legitimately one-to-many predicate between two non-event nodes is still blindly deprecated — and the out-of-order case, where ingesting older data invalidates the *newer* fact. Adopt with `2501.13956`; cite `2606.01435` for the latency argument only, per the caveat in `00`.
- **4.2 Apply the current-truth filter to every channel.** It is called in exactly one place (`retrieve.py:787`) and guards only the triple channel; the passage channel and `historical_context` have no temporal predicate, so an invalidated edge and the sentence asserting it ship in the same payload. Per the maintainer's decision, current-truth becomes the default with opt-in history. Passage annotation must come from a write-time rollup — no LLM call on the query path.
- **4.3 Canonicalize types and predicates *after* open extraction.** `2402.10744` found that constraining an LLM to a fixed relation set *causes* hallucination, so post-hoc canonicalization is the right architecture, not schema-constrained extraction. Backing: `2605.29168`, `2505.23628`.
- **4.4 Ground each fact in a verified source span.** The Janitor's revision protocol is entirely structural; nothing asks whether a relationship is supported by the source text, so a fluent hallucination passes. `2510.00276`.
- **4.5 Contradiction-aware dedup.** `2606.26511` measured cosine similarity separating a contradicted fact from a duplicate at AUROC 0.59 — near chance — so the current similarity-based dedup preferentially swallows the very updates it should act on.
- **4.6** Reuse `KGAgent.verify_entity_existence` (`kg_agent.py:219`), which already implements candidate-generation-plus-LLM-adjudication, on the free-text path's abstain-on-ambiguity branches (`ingestion.py:290-291`), which currently create a duplicate instead of asking it.

---

## Deliberately deferred

- **Judge validation against human labels.** No human labels exist and no agreement statistic is computed, so judge accuracy *is* the metric while nothing measures the judge. `config.py:17-18` uses the same model as answerer and judge, and the judge never sees the retrieved context (`prompts.py:86-90`), so it cannot assess groundedness — the one property that matters for a memory product. A 200-300 item labelling pass would calibrate it. **Recommended; awaiting the maintainer's decision.**
- **Supersession sub-benchmark.** Explicitly deferred by the maintainer. Consequence: temporal fixes are unit-testable but their end-to-end effect is unmeasurable. LoCoMo category 2 scores 0.86 while measuring event *localization*; roughly 12 of 1,986 questions probe current truth.
- **Scaling to all ten conversations.** Currently 152 of 1,540 non-adversarial questions (9.9%), one conversation. The two priority categories are measured at n=32 (multi-hop, CI width 27.7pp) and n=37 (temporal, 22.1pp); open-domain is n=13 (47.7pp). Power analysis: a *concentrated* effect that repairs a broken question class is detectable at 1-2 runs per arm, but a **diffuse uniform +5-point improvement reaches only 0.14 power at two runs and 0.52 at six**. This benchmark can validate targeted fixes and cannot validate broad polish at any affordable run count.
- **Backfill for wrongly-deprecated edges.** Not authorised. Mechanical when wanted: clear `deprecated`/`invalid_at` where either endpoint carries the `EVENT` label.

## Risks

| Risk | Impact | Detection / mitigation |
| --- | --- | --- |
| The ablation shows the graph adds little | High — **realized:** no LoCoMo judge gain above floor; graph EvR ~53% provisional and not converting | Do not justify Phase 4 on LoCoMo accuracy alone; stabilize EvR first, then fact-budget / selection, multi-hop composition, or non-LoCoMo surfaces |
| Graph EvR treated as a measurement before order-stability | High — identical-config runs agree on only ~39–42% of questions | Phase 0 follow-up: order-stable fact selection; do not A/B on graph EvR until then |
| Concurrent agents share a brain id / run dir | Medium — interleaved `ingest.jsonl`, duplicate extractions | One owner per brain id; manifest records server code identity (see clean-brain-ingest/NOTES.md) |
| Removing fitted prompt rules drops the headline and looks like regression | Medium | Re-baseline explicitly and label the old numbers as fitted |
| Sub-second and accuracy are irreconcilable on one path | Medium | Already mitigated by the two-tier split; escalate with numbers |
| Latency attribution is wrong because it was inferred, not measured | Medium | **Mitigated:** 3.1 measured; dossier deletion confirmed live (~4.8 s → ~1.4 s) |
| Cardinality still missing after the hotfix | Medium | One-to-many non-event predicates still wrongly deprecated; Task 4.1 |
| Unrepaired brains silently confound every comparison | High | Record `ingest.n` per run; require a fresh ingest for any graph-side claim |
