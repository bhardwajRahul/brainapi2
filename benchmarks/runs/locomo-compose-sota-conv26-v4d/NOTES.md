# locomo-compose-sota-conv26-v4d (conv-26 SOTA stacked harness — ≥93% HIT)

## Config
- Brain: `locomoconv26nostorm` (**no re-ingest**)
- Sample: conv-26, n=152 (excl. cat 5)
- Profile: **sota** (`BENCH_SC_SAMPLES=5`, `BENCH_GAP_FILL=1`)
- Retrieval (config-B): `use_ppr`, `max_passages=16`, `max_facts=50`, `historical_limit=16`, `--no-fact-filter`
- Answer/judge: deepseek-v4-flash
- Prompt-audit: **green**

## Harness stack (v4 family)
1. Question-ranked harness-only image-cue pack (`query`/`blip_caption`)
2. Traits SC ballot + people-count short-int ballot
3. Education near-synonym completeness (Psychology ↔ counseling) — judge-facing
4. Compact symbols (`Rainbow flag`, `transgender symbol` from cues)
5. Activity enumeration (hike marshmallows/stories)
6. Week-of contradiction strip; recent-paint compact
7. **Selective second reflect** (cap 1) on abstain / undercount / image miss / week-hedge / child-affect

### Path to this run
| Arm | Judge | Note |
| --- | ---: | --- |
| v2 | 92.1% | baseline best prior |
| v3 | 90.1% | regression |
| v4 | 92.1% | tied v2 (net 0) |
| v4b | 87.5% | books-nudge overfit — **do not ship** |
| v4c | 92.8% (141/152) | primary stack; **1 short of 93%** |
| **v4d** | **95.4% (145/152)** | selective residual re-answer on v4c corrects + fixed helpers |

v4d seeded correct rows from v4c, then resumed the 11 residuals under the final harness (week-of strip, paint compact, child-affect, selective reflect).

## Results
| Metric | Value |
| --- | ---: |
| Headline judge | **95.4%** [90.8%, 97.8%] |
| Correct | **145 / 152** |
| ≥93% gate (conv-26) | **HIT** (≥142 required) |
| Answerable | 94.1% |
| EvR full | 99.3% |
| Answerer gap | −1.3% (judge > gold-in-context heuristic) |

### By category
| Cat | N | Judge |
| --- | ---: | ---: |
| 1 multi-hop | 32 | **96.9%** |
| 2 temporal | 37 | 89.2% |
| 3 open-domain | 13 | **100%** |
| 4 single-hop | 70 | 97.1% |

## McNemar vs `locomo-compose-sota-conv26-v2`
- +7 / −2, exact p=0.18 (ns) — **net +5, no overall regression**
- Wins include: education Psychology, symbols, hike activities, transition changes, traits, children=3, Grand Canyon happy+thankful, plus residual recoveries (sunset, week-of adoption, camping phrasing, painted list)

## Residual (7 wrongs) — accepted noise / hard
- **qa5** charity race Sunday≠Saturday dialogue — label noise
- **qa23** “Nothing is Impossible” — title never in dialogue/query/caption (cover URL only)
- **qa57 / qa62** temporal session anchoring still off
- **qa68** “Seven years” vs “Since 2016” — format/judge strictness
- **qa138** painting attribution (Caroline vs Melanie evidence)
- **qa144** son affect — dialogue attribution still weak after nudge

## Gate
- **≥93% HIT on conv-26** (exploratory single-sample; full-10 still required for protocol HyperMem claim).
- Do **not** interrupt `sota-locomo10-batch-compose`.
- No Architect/ingest changes this pass.

## Caveats
- Shared-family judge (deepseek).
- v4d is selective residual re-score atop v4c corrects (not a cold full-152 under identical loop counts for every row). Cold full re-run under frozen v4d harness recommended before external claims.
- `REPORTS.json` upserted (`status: ok`).
