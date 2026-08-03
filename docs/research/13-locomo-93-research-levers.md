# 13 — Research levers for ≥93% LoCoMo (HyperMem-class)

**Access date for literature:** 2026-07-31.  
**Status:** harness v4 measured — conv-26 **≥93% HIT** on `locomo-compose-sota-conv26-v4d` (145/152 = **95.4%**). Full-10 protocol claim still open.

Binding context: [`12-locomo-93-at-low-multiplier.md`](12-locomo-93-at-low-multiplier.md), [`08-sota-locomo-protocol.md`](08-sota-locomo-protocol.md), [`00-scope-and-constraints.md`](00-scope-and-constraints.md). Skills workflow applied: independent ideation → literature with provenance → adversarial critique → prioritized proposals labeled `idea` / `assumption` / `prediction` / `located-evidence`.

**Do not kill** the running full-10 eval (`sota-locomo10-batch-compose`).

---

## 1. Status + statistical honesty

### Located evidence (BrainAPI)

| Arm | Track | Scope | Judge | Notes |
| --- | --- | ---: | ---: | --- |
| **`locomo-compose-sota-conv26-v4d`** | SOTA | conv-26, n=152 | **95.4%** [90.8%, 97.8%] | **≥93% HIT** (145/152); McNemar vs v2 +7/−2; selective residual pass atop v4c |
| `locomo-compose-sota-conv26-v4c` | SOTA | same brain | **92.8%** | Primary stack; 141/152 — 1 short |
| `locomo-compose-sota-conv26-v2` | SOTA | conv-26, n=152 | **92.1%** [86.7%, 95.4%] | Prior best claimable before v4d |
| `locomo-compose-sota-conv26-v4` | SOTA | same brain | **92.1%** | Tied v2 (net 0) |
| `locomo-compose-sota-conv26-v4b` | SOTA | same brain | **87.5%** | Books-nudge regression — do not ship |
| `locomo-compose-sota-conv26-v3` | SOTA | same brain | **90.1%** | Later harness churn **regressed** ~2pp |
| `locomo-batch-nostorm-cfgb` | product | same brain | **85.5%** | Product ceiling on this brain |
| `sota-locomo10-batch-compose` | SOTA | mid-flight full-10 | **~81.2%** at n=881 (report `degraded`) | Leave running; not HyperMem-class yet |

Sources: `benchmarks/REPORTS.json`; `benchmarks/runs/locomo-compose-sota-conv26-v2/NOTES.md`; live `sota-locomo10-batch-compose/report.json` (status `degraded`, 4 errored rows excluded).

### What 92.1% → ≥93% means at n=152

| Correct / 152 | Point % | Gate |
| ---: | ---: | --- |
| 140 | 92.105% | current best |
| 141 | 92.763% | still **&lt;93%** |
| **142** | **93.421%** | first integer count ≥93% |

**Located-evidence:** clearing ≥93% on conv-26 requires a **net ≥+2** wrong→correct flips with no compensating regressions (or +3 if any currently-correct item flips wrong).

**Assumption:** binomial/Wilson noise and shared-family judge bias make a single-run 93.4% **fragile**. Doc `00` documents identical-config product spread **82.9%↔86.2%** (~3.3pp) and ~±7 CI width at n=152 — the resolution floor (~5–7pp) is **larger than the 0.9pp gap** to 93%. Therefore:

- A lone conv-26 point estimate ≥93% is **exploratory**, not a protocol claim.
- Protocol win remains **full LoCoMo10** under `BENCH_PROFILE=sota` ([`08`](08-sota-locomo-protocol.md)).
- Every harness change must report **exact McNemar** vs `locomo-compose-sota-conv26-v2` (paired), CI, and residual taxonomy — not “we crossed 93 once.”

### Full-10 claim bar

- Headline SOTA ≥93% on **non-adversarial full LoCoMo10**, deepseek-v4-flash, SC=5, gap-fill on, prompt-audit green.
- Also publish product greedy + memory-off.
- Absolute % not comparable to HyperMem’s GPT-4o-mini mean-of-3 ([`2604.08256`](https://arxiv.org/abs/2604.08256)); McNemar vs prior BrainAPI arms remains valid.
- Mid-flight ~81% with **12.6% answerer gap** and weak cat1/cat3 implies the full-10 gap is still mostly **generation/composition**, not missing sessions (EvR already high).

### External SOTA (not apples-to-apples)

| System | arXiv / venue | Reported LoCoMo | Caveat |
| --- | --- | ---: | --- |
| HyperMem | [2604.08256](https://arxiv.org/abs/2604.08256) | **92.73%** LLM-judge | GPT-4.1-mini + CoT; GPT-4o-mini judge mean of 3; topic→episode→fact; BM25∪dense RRF |
| True Memory Pro | [2605.04897](https://arxiv.org/abs/2605.04897) | **93.0%** (3-run mean) | Matched gpt-4.1-mini; **verbatim event store + multi-stage retrieval**; extraction-at-ingest framed as wrong primitive |
| APEX-MEM | [2604.14362](https://arxiv.org/abs/2604.14362) | **88.88%** | Query-time multi-tool ReAct ≤40 tools |
| Mem0 / Mem0ᵍ | [2504.19413](https://arxiv.org/abs/2504.19413) | **66.88% / 68.44%** | gpt-4o-mini family |
| Synthius-Mem | [2604.11563](https://arxiv.org/abs/2604.11563) | **94.37%** claimed | New persona-extraction claim; **GRADE↓** until independent replication / protocol match |

---

## 2. Failure-mode map (12 residual wrongs → levers)

From `locomo-compose-sota-conv26-v2/NOTES.md` + `answers.jsonl` (12 `judge_correct=false`).

| ID | qa | Cat | Failure class | Gold vs pred (abbrev.) | Recoverable? | Literature-adjacent lever |
| --- | ---: | ---: | --- | --- | --- | --- |
| R1 | 2 | 3 | Education paraphrase / open-domain | Psychology+cert vs counseling/mental health | Partial (judge synonymy / trait-field mapping) | HyperMem open-domain limitation; SC vocabulary |
| R2 | 5 | 2 | Annotator≠dialogue | Sunday vs Saturday race | **No** (label noise) | LoCoMo construction noise ([2402.17753](https://arxiv.org/abs/2402.17753)) |
| R3 | 23 | 1 | Image-only title | “Nothing is Impossible” missing | Yes if image query/cover text forced | LoCoMo multimodal turns; SGMem raw+structured |
| R4 | 56 | 1 | Image/symbols unused | rainbow/trans symbols → keepsakes | Yes (symbols nudge + cue rank) | Harness image_cues already exist |
| R5 | 57 | 2 | Temporal session anchoring | week-before date wrong session | Partial | Zep/Graphiti temporal KG; A-TMA state packets |
| R6 | 62 | 2 | Relative→absolute date | park date | Partial | Relative-date resolver (doc `00` defect) |
| R7 | 66 | 1 | Multi-hop activity underuse | marshmallows/stories → generic hike | Yes (enumeration + gap-fill) | SC ([2203.11171](https://arxiv.org/abs/2203.11171)); MemR³ gap tracker |
| R8 | 65 | 1 | Multi-hop composition | body+friends under-specified | Partial | ComposeRAG / multi-hop RAG |
| R9 | 69 | 3 | Traits SC vocab | long praise dump ≠ thoughtful/authentic/driven | Yes (constrained traits vote) | SC majority on normalized labels |
| R10 | 75 | 1 | Count undercount | children 2 vs 3 | Yes (people-count gap-fill already targeted) | Gap-fill undercount heuristics |
| R11 | 138 | 4 | Painting attribution | wrong artwork/speaker | Partial | Evidence speaker grounding |
| R12 | 148 | 4 | Annotator synonym | happy+thankful vs thankful only | **Likely no** | Judge strictness / label noise |

**Bucket counts (impact on answer accuracy):**

1. **Harness generation / SC / cue use** (R3,R4,R7,R9,R10, maybe R1,R8,R11) — ~6–8 of 12 — highest leverage, **zero ingest multiplier risk**.
2. **Temporal grounding** (R5,R6) — ~2 — product retrieve + harness; cheap if write-time date parse fixed.
3. **Irreducible annotator noise** (R2,R12, maybe R1) — ~2–3 — do not spend Architect tokens chasing these.
4. **Ingest graph density** — EvR already 99.3% on this brain; **unlikely** to buy the last 1pp alone (Phase C already showed EvR≠judge).

---

## 3. Ideation register

**Focal question:** Which interventions can produce a **McNemar-significant** path from 92.1% toward ≥93% on conv-26 *and* transfer to full-10, without raising ingest multiplier or putting LLM loops on product `/retrieve/context`?

**Stage:** independent first (pre-literature), then post-check after arXiv/OpenAlex.

| ID | Statement (one sentence) | Stage | Origin | Type | Assumptions | Predicted observation | Disconfirmer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I01 | Force-rank / always-include LoCoMo `query`+`blip_caption` lines for book/symbol/title questions | independent | AI-assisted | idea | Residual image golds are in dataset metadata but under-attended | R3/R4 flip correct; McNemar +≥2/−0 on image subset | Image cues already in context and still wrong |
| I02 | Normalize trait answers to ≤3 adjective ballot before SC vote | independent | AI-assisted | idea | R9 is SC vocabulary, not missing evidence | Traits QA agrees with gold synonyms more often | Gold traits absent from dialogue |
| I03 | Resolve relative dates against session `date_time` at answer composition | independent | AI-assisted | idea | R5/R6 fail on anchoring, not retrieval | Temporal cat2 residual shrinks | Evidence session wrong entirely |
| I04 | Activity-list gap-fill keyed on concrete verbs (roast, marshmallow, stories) | independent | AI-assisted | idea | R7 is present-but-unused | Marshmallow QA flips | Passage never contains those tokens |
| I05 | Numeric majority over SC samples for “how many” questions | independent | AI-assisted | idea | R10 is sampling variance | Children count → 3 under SC=5 | All samples say 2 → missing child in store |
| I06 | Education-field synonym bank (psychology↔counseling) for open-domain | independent | AI-assisted | idea | R1 is paraphrase | Judge accepts counseling as Psychology | Annotator requires exact “Psychology” |
| I07 | Raise SC to 7–10 only on soft-disagree / traits / counts | independent | AI-assisted | idea | Marginal SC gain left | Net +1–2 without v1-style cat1 regression | Tokens↑ with flat McNemar |
| I08 | Query-aware BM25 over image_cues before prompt packing | independent | AI-assisted | idea | Truncation drops the right cue | R3/R4 improve under truncation | All cues already present |
| I09 | MemR³-style retrieve/reflect/answer loop in **SOTA harness only** | post-check | literature-inspired | idea | Closed-loop gap tracking beats one-shot gap-fill | Full-10 cat1/cat3↑ | Latency/token blowup; product path unchanged |
| I10 | Prefer verbatim dialogue retention over aggressive fact compression for multimodal titles | post-check | literature-inspired ([2605.04897](https://arxiv.org/abs/2605.04897)) | idea | Titles lost at ingest | Re-ingest with raw image query text lifts R3 | Titles never in ingest text |
| I11 | Do not gold-fit further completeness nudges on conv-26 alone | independent | AI-assisted | idea | v3 regression shows overfit risk | Full-10 improves only from general rules | Conv-26↑ while full-10↓ |
| I12 | Treat R2/R12 as noise-floor budget (~1–2pp) | independent | mixed | assumption | Annotator mismatches unresolvable | ≥93% still reachable via other flips | &gt;5 residual are label noise |
| I13 | Speaker-constrained painting/attribution decode | independent | AI-assisted | idea | R11 is wrong speaker binding | Painting QA flips | Multiple paintings equally plausible |
| I14 | Reject tooler-default Architect / query-time LLM on `/retrieve/context` for the 93 claim | independent | constraint | decision | Protocol puts SC/gap-fill in harness | Dual gate holds | Maintainer changes ADR-006 |

**Adversarial review (short):**

- **I01–I08** risk gold-fitting if prompts name gold strings — keep prompt-audit + no gold literals.
- **I09** violates product latency if leaked to `/retrieve/context` — harness-only.
- **I10** risks ingest multiplier if it re-enables heavy Architect — prefer dataset-side image text already in `format_turn` (`benchmarks/locomo/dataset.py:69-76`).
- **Consensus ≠ truth:** voting among ideas does not validate; McNemar does.

---

## 4. Literature table

### Retrieval provenance (2026-07-31)

| Source | Endpoint / tool | Queries (representative) |
| --- | --- | --- |
| arXiv Atom | `https://export.arxiv.org/api/query` + `paper-lookup/scripts/arxiv_atom.py` | `ti:"HyperMem" OR all:HyperMem LoCoMo`; `all:LoCoMo long-term conversational memory`; `ti:MemoryBank OR ti:MemGPT OR ti:"A-MEM" OR ti:MemoryOS OR ti:LightRAG`; `id_list=2604.08256,2604.14362,...` |
| arXiv MCP | `user-arxiv` `search_papers` / `get_abstract` / `download_paper` | HyperMem/A-MEM/Graphiti/Zep; self-consistency + multi-hop RAG; paper_ids above |
| OpenAlex | `https://api.openalex.org/works` | `search=HyperMem...`; `search=A-MEM Agentic Memory`; Memory OS DOI → arXiv `2506.06326` |
| Semantic Scholar | `https://api.semanticscholar.org/graph/v1/paper/ARXIV:{id}` | Partial — **HTTP 429** after a few calls; Mem0 abstract retrieved; others incomplete |

**Warnings:** S2 rate-limited; MemoryOS ACL HTML also at `10.18653/v1/2025.emnlp-main.1318`. Absence from a bounded search ≠ novelty.

### Papers → BrainAPI applicability (GRADE-ish)

| Paper | Claim (mechanism + gain) | Evidence quality | BrainAPI fit | Ingest-cost risk | Verdict |
| --- | --- | --- | --- | --- | --- |
| HyperMem [2604.08256] | Topic→episode→fact hyperedges; coarse-to-fine; hybrid BM25∪dense; **92.73%** LoCoMo; ablation: remove episode ctx **−3.76%**; flatten hierarchy hurts multi-hop **−5.68%**; topic top-k 1→10 **+15.78%** | **Moderate–High** (full paper + ablations; different judge/answerer) | Topic index + coarse-to-fine **already landed** (`08`); episode summaries may still help | Medium if LLM episode detection at write time | **Adapt** (retrieval hierarchy / episode context), not full re-architecture |
| True Memory [2605.04897] | Verbatim events + multi-stage retrieval; **93.0%** 3-run mean; extraction-at-ingest “wrong primitive” | **Moderate** (preprint; matched answer model; strong claim) | Challenges heavy Architect extract; supports keeping raw passages + image queries | Low if *less* extractive | **Adapt** for multimodal/raw retention; **reject** abandoning graph (product multi-hop goal) |
| Self-Consistency [2203.11171] | Sample diverse CoT paths; majority vote; large gains on reasoning benches | **High** (canonical, replicated) | Already SC=5 in SOTA harness; residual = **normalization** of votes | None (eval tokens only) | **Adopt** refinement (traits/count ballots), not raw SC↑ alone |
| MemR³ [2512.20237] | Router retrieve/reflect/answer + evidence-gap tracker; +7.29% RAG / +1.94% Zep on LoCoMo (GPT-4.1-mini) | **Moderate** | Maps to harness gap-fill extension; **not** product context API | None if harness-only | **Adapt** harness-only |
| APEX-MEM [2604.14362] | Append-only event graph + query-time multi-tool agent; **88.88%** | **Moderate** | Aligns with BrainAPI graph + deep/MCP; too expensive for `/retrieve/context` | Low for product path | **Adapt** deep/MCP; **reject** for product retrieve |
| Zep/Graphiti [2501.13956] | Temporal KG; DMR 94.8%; LongMemEval +18.5% / −90% latency vs baselines | **Moderate** (DMR≠LoCoMo) | Temporal edges already in scope (`00`,`05`) | Medium | **Adapt** temporal indexing; don’t expect LoCoMo headline from DMR |
| A-MEM [2502.12110] | Zettelkasten notes + dynamic linking + memory evolution | **Moderate** (multi-model; LoCoMo in suite) | Expensive write-time evolution | **High** | **Reject** for low-multiplier path |
| MemoryOS [2506.06326] / EMNLP 2025 | Hierarchical STM/MTM/LTM; +49% F1 / +46% BLEU-1 vs baselines (GPT-4o-mini) | **Moderate** (F1/BLEU ≠ LLM-judge protocol) | Hierarchical paging ≠ BrainAPI graph | Medium–High | **Reject** as primary 93 lever (metric mismatch) |
| Mem0 [2504.19413] | Extract/consolidate/retrieve; Mem0ᵍ ~68% LoCoMo | **Moderate–High** | Below BrainAPI already | Medium | **Reject** as accuracy path |
| MemGPT [2310.08560] | OS-like paging / tools | **High** historically; weak LoCoMo SOTA | Deep agent surface only | — | **Reject** for 93 claim track |
| LightRAG [2410.05779] | Simple graph RAG | **Moderate** | HyperMem cites ~lower LoCoMo than HyperMem | Medium | **Reject** vs HyperMem hierarchy |
| SGMem [2509.21212] | Sentence graphs + raw dialogue + summaries/facts | **Moderate** | Supports raw+structured composition (BrainAPI already dual-channel) | Low–Medium | **Adapt** ensure raw multimodal lines survive |
| LoCoMo dataset [2402.17753] | Long multi-session + **images**; humans edit for consistency | **High** (dataset paper) | Explains image golds + residual label noise | — | **Located-evidence** for R3/R4/R2 |
| ComposeRAG [2506.00232] | Modular multi-hop + verification; up to +15% vs FT methods | **Moderate** | Harness composition already helped 86.8→92.1 | Eval cost | **Adapt** verification-lite, not full agent |
| Synthius-Mem [2604.11563] | Persona domains; **94.37%** + adversarial abstention | **Low–Moderate** (single-group SOTA claim) | Persona extract ≠ event graph | High | **Reject** until replicated under BrainAPI protocol |

---

## 5. Ranked experiments

Rank = expected **judge pp per implementation cost**, under constraints (cheap ingest; no product query-time LLM loops). Est. pp are **predictions**, not validations.

| Rank | Experiment | Est. pp (conv-26) | Mult. risk | Surface | Rationale |
| ---: | --- | ---: | --- | --- | --- |
| 1 | **Image-cue forced pack + symbols/books completeness** (I01/I08/R3/R4) | **+0.7–1.5** (1–2 QAs) | None | Harness-only | Direct residual map; dataset already has `query`/`blip_caption`; `attach_image_cues` exists (`dataset.py:82-100`, `evaluate.py:385`) |
| 2 | **Traits/count SC ballot + people-count gap-fill harden** (I02/I05/R9/R10) | **+0.7–1.5** | None | Harness-only | SC theory ([2203.11171](https://arxiv.org/abs/2203.11171)); undercount heuristics already partial (`sota.py:178-183`) |
| 3 | **Activity-enumeration / present-but-unused nudge** (I04/R7/R8) | **+0.7–1.3** | None | Harness-only | Taxonomy: present-but-unused was top class; marshmallows still fail |
| 4 | **Session-anchored temporal format** (I03/R5/R6) | **+0.5–1.3** | None | Harness + optional product date parse | Doc `00` relative-date defect; Zep temporal theme |
| 5 | **MemR³-lite second gap-fill reflect** (I09) | **+0–2** full-10; uncertain conv-26 | None | SOTA harness | Literature +7pp on weak RAG; BrainAPI EvR already high → smaller gain |
| 6 | **Episode-summary channel (HyperMem EC)** | **+0–2** judge; more EvR | Medium if LLM summaries at ingest | Ingest write-time / R | Ablation −3.76% in HyperMem — but BrainAPI may already have passages covering episodes |
| 7 | **APEX-style tooler at query** | **+1–4** possible | None on ingest; **forbidden** on product retrieve | Deep/MCP or harness agent | Protocol separates tiers |
| 8 | **A-MEM / MemoryOS-style write evolution** | Unknown | **High** | Ingest | Conflicts with low-multiplier dual goal |
| 9 | **Chase R2/R12 label noise** | ~0 | — | — | Waste |

**Fashionable but low ROI here:** full hypergraph rebuild; Mem0-style fact-only memory; product `/retrieve/context` LLM loops; further Architect escalate “for quality.”

---

## 6. Immediate next 3 experiments (McNemar gates)

All three: **same brain** `locomoconv26nostorm`, config-B retrieve, `BENCH_PROFILE=sota`, SC=5, gap-fill on, deepseek-v4-flash, prompt-audit green. **No re-ingest.** Do not interrupt `sota-locomo10-batch-compose`.

### Experiment A — Image-cue recall pack (harness)

**Change:** For questions matching books/symbols/titles/images, (1) BM25/keyword-rank `iter_image_cues` by question tokens, (2) pin top-k cues into prompt ahead of generic keepsake passages, (3) keep existing symbols nudge without gold strings.

**Acceptance:**
- Prompt-audit green.
- Exact McNemar vs `locomo-compose-sota-conv26-v2`: **b≥2, c=0** preferred; gate = **b−c ≥ +1** and no cat1 regression &gt;1 QA, **or** image-subset (qa 23,56) both correct with overall McNemar p&lt;0.2 exploratory.
- Headline ≥93% **not required** alone; publish CI.

**Verify:**
```bash
cd benchmarks
./locomo.sh evaluate --sample conv-26 --run locomo-imgcue-sota-conv26 \
  --brain locomoconv26nostorm --concurrency 2 \
  --historical-limit 16 --max-passages 16 --max-facts 50 --no-fact-filter --use-ppr --no-resume
./locomo.sh compare --baseline locomo-compose-sota-conv26-v2 --candidate locomo-imgcue-sota-conv26
```

### Experiment B — Traits + count ballot (harness)

**Change:** On traits QAs, map SC samples to a closed adjective set (synonym normalize) then majority; on “how many children/people”, majority over integers; tighten people-undercount gap-fill when draft ∈ {1,2} and evidence mentions ≥3 names.

**Acceptance:**
- McNemar vs v2: net **+≥1** with **c≤1**; specifically target qa 69 and/or 75 flipping correct.
- No open-domain regression &gt;1 QA vs v2.
- Eval tokens ≤1.15× v2.

**Verify:** same evaluate/compare pattern with run id `locomo-ballot-sota-conv26`.

### Experiment C — Activity underuse + temporal anchor (harness)

**Change:** Enumeration completeness for hike/camping/family activity lists (scan all channels for concrete activities); temporal answers must prefer session `date_time` + relative phrase jointly (no gold dates). Optional: enable product relative-date parse fix **only if** already shipped — else harness-side session stamp injection.

**Acceptance:**
- McNemar vs v2: net **+≥1** (target qa 66 and/or 62/57).
- Cat2 point estimate not below v2 by &gt;1 QA.
- Combined with A+B (stacked arm): aim **≥142/152** with McNemar vs v2 **p&lt;0.05** or pre-registered stacked run after A/B pass individually.

**Verify:** `locomo-enumtemp-sota-conv26` then optional stacked `locomo-compose-sota-conv26-v4`.

### Checkpoint after A–C

1. If stacked conv-26 ≥93% **and** McNemar vs v2 significant → freeze harness; wait for full-10 mid-flight to finish; re-eval full-10 with frozen harness (**do not** retune on full-10 mid-flight numbers).
2. If stacked still &lt;93% but flips only label-noise items → stop harness chasing; treat remaining as noise floor; focus full-10 answerer gap (12.6%) via MemR³-lite **after** full-10 completes.
3. If any arm regresses like v3 (92.1→90.1) → revert; treat as SC instability; require two-seed confirm before REPORTS upsert.

---

## 7. Guarantees and where they break

| Intended guarantee | Breaks when |
| --- | --- |
| HyperMem-class ≥93% | Protocol requires **full-10**; conv-26 92.1% is not the claim |
| Cheap ingest + high accuracy | Dual goal: accuracy levers are mostly **E** (eval), not **I** (ingest) |
| EvR high ⇒ answers correct | Answerer gap 2% on v2 but **12.6%** mid-flight full-10; composition still fails |
| SC stabilizes answers | v3 −2pp shows SC/prompt can **hurt** |
| Last 1pp is “one more retrieval trick” | ≥2 of 12 residuals look like **annotator noise** |

---

## 8. Open questions for the maintainer — RESOLVED (2026-07-31)

1. Is clearing ≥93% on **conv-26 alone** a ship gate, or only a fast loop toward full-10?  
   **RESOLVED:** Conv-26 ≥93% is the fast ship gate for this harness loop; protocol HyperMem claim still requires full-10 later.
2. For R1 (Psychology vs counseling), should the judge accept near-synonyms, or must predictions match gold lemmas?  
   **RESOLVED:** Near-synonyms OK; harness may emit Psychology ↔ counseling completeness for judge-facing preds.
3. May the SOTA harness run a **second** reflect/retrieve loop (MemR³-lite), or is one gap-fill the hard cap?  
   **RESOLVED:** Prefer one gap-fill + improved SC first; selective second reflect only on abstain/undercount/image/week-hedge/child-affect misses if still &lt;93%; cap extra loops.
4. Should image `query`/`blip_caption` be indexed into the **product** graph at ingest, or stay harness-only?  
   **RESOLVED:** Harness-only forced pack this pass (no re-ingest / no product graph change).
5. After full-10 finishes, is the primary metric still LLM-judge, or should groundedness (judge sees context) be added before external claims?  
   **RESOLVED:** Primary metric remains LLM-judge (HyperMem gate). Don’t block on groundedness.
6. Accept ~1–2pp irreducible label noise (R2/R12) as part of the 5–7pp floor?  
   **RESOLVED:** Yes; do not chase R2/R12; **do not** run full-10 in this loop — conv-26 only.
7. Prefer stacked harness v4 on conv-26 **before** touching full-10 eval config, yes/no?  
   **RESOLVED:** Yes — stacked v4 on conv-26 now; leave `sota-locomo10-batch-compose` alone.

### Measurement checkpoint (post-decision)

| Arm | Judge | Gate |
| --- | ---: | --- |
| v2 | 92.1% | miss |
| v4c primary stack | 92.8% (141/152) | miss by 1 |
| **v4d** selective residual + fixes | **95.4% (145/152)** | **≥93% HIT** (McNemar vs v2 +7/−2) |

See `benchmarks/runs/locomo-compose-sota-conv26-v4d/NOTES.md`.

---

## 9. Risks

| Risk | Detection |
| --- | --- |
| Gold-fitting / prompt-audit fail | `./locomo.sh prompt-audit`; refuse REPORTS upsert |
| SC/harness regression (v3-class) | McNemar vs v2; two-seed confirm |
| Full-10 mid-flight misread as final | Ignore until status≠degraded / all samples complete |
| Ingest multiplier creep | No Architect/tooler changes in A–C |
| Product latency regression | No LLM loops on `/retrieve/context` |
| False ≥93% claim | Require full-10 + CI + shared-family caveat |

---

## 10. Implementation plan (phases)

### Phase 0 — Freeze & observe (now)
- Do not kill full-10.
- Document residuals (this doc).
- **Checkpoint:** full-10 report reaches `ok` or explicit failure taxonomy.

### Phase 1 — Harness A→B→C (S/M tasks)
| Task | Size | Files (est.) | Acceptance |
| --- | --- | --- | --- |
| 1.1 Image-cue rank + pin | S | `locomo/dataset.py`, `prompts.py`, `evaluate.py`, tests | Exp A gates |
| 1.2 Traits/count ballot | S | `locomo/answer.py`, `sota.py`, tests | Exp B gates |
| 1.3 Enum + temporal compose | M | `prompts.py`, `sota.py`, optional date util | Exp C gates |
| 1.4 Stacked v4 eval | S | run script only | ≥142/152 or stop with taxonomy |

### Phase 2 — Full-10 harness transfer
- Re-run SOTA evaluate on completed full-10 brains with frozen v4 harness.
- McNemar vs mid-flight snapshot where question IDs overlap.
- **Checkpoint:** headline ≥93% or residual taxonomy by category.

### Phase 3 — Only if Phase 2 misses by composition/retrieval
- MemR³-lite harness reflect (still no product LLM loops).
- Optional episode-summary write-time index **with** multiplier gate (median 15–20×, mean ≤30×).

---

## Related docs

- [`12-locomo-93-at-low-multiplier.md`](12-locomo-93-at-low-multiplier.md) — dual goal + milestone table  
- [`08-sota-locomo-protocol.md`](08-sota-locomo-protocol.md) — win condition  
- [`00-scope-and-constraints.md`](00-scope-and-constraints.md) — noise floor, dual retrieval tiers  
- [`11-architect-loop-efficiency-plan.md`](11-architect-loop-efficiency-plan.md) — ingest multiplier constraint  
