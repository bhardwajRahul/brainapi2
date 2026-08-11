# Deep Interconnected E-Commerce Recommender Systems — Landscape Index

Workstream: external curriculum / tutorial / industrial index for graph-structured e-commerce recommendation, and how it maps (or does not map) onto BrainAPI.

**Status of this document:** Research **index and critical framing**, not a finding and not an implementation plan. Primary source material was provided by the maintainer as a compiled review titled *Deep Interconnected E-Commerce Recommender Systems: A Comprehensive Review of Academic Lecture Series, Tutorial Syllabi, and Industrial Frameworks* (ingested 2026-08-05). Claims below are labeled:

| Label | Meaning |
| --- | --- |
| **source material** | Restated from the provided review / its cited URLs |
| **located evidence** | Independently checked (arXiv ID, course URL, or codebase path) |
| **idea / proposal** | BrainAPI-facing implication; not validated |
| **decision** | Locked elsewhere (plan / ADR); restated for cross-link only |

Related internal docs: [`04-evaluation-and-applications.md`](04-evaluation-and-applications.md) §4.9 (synergies metrics), synergies/recsys feasibility plan (Phases 0–4), [`00-scope-and-constraints.md`](00-scope-and-constraints.md) (recommenders as a product substrate).

---

## 1. What the source material argues

**source material.** E-commerce recommendation has moved from matrix factorization toward deep, graph-structured, multi-stage pipelines because catalogs are sparse, scale-free, and multi-behavior. Isolated interaction instances miss structural relationships in user–item graphs. The review indexes:

1. Graduate curricula (Stanford CS224W, Yale CPSC 583, Minnesota LensKit specialization).
2. Conference tutorials (IJCAI 2021 DeepRecSys, WSDM 2022 GNN-RecSys, LoG 2023, KDD 2020/2022).
3. Production GPU / RL pipelines (NVIDIA Merlin, RecSys 2022 Ray/RLlib).
4. Industrial architectures (Alibaba ComiRec/MIND, Pinterest PinSage/PinnerSage, Amazon asymmetric complementary GNN, Uber Eats GraphSAGE).
5. Three architectural directives: decouple graph propagation from deep feature interaction; multi-interest over single-vector users; topology-aware offline evaluation against popularity bias.

**Critical note (methodology).** The provided text is an **analytical syllabus index**, not a primary empirical study. Citation tags `[cite: N]` in the source are opaque (no bibliography keyed to those numbers). Treat numerical claims from industrial blogs (e.g. Uber AUC 78%→87%) as **unverified secondary reports** until traced to a primary paper or engineering post with methods.

---

## 2. Academic curricular foundations

### 2.1 Stanford CS224W — Machine Learning with Graphs

**source material.** Instructors associated with Jure Leskovec / Rex Ying. User–item interactions modeled as bipartite \(G=(U,V,E)\). Lectures cover NGCF → LightGCN, GIN expressiveness, and scale methods (neighbor sampling, Cluster-GCN, simplified/precomputed GCN). Neural subgraph matching maps motifs into order-embedding cones.

**LightGCN propagation (as stated in source; matches literature):**

\[
e_u^{(k+1)}=\sum_{i\in N_u}\frac{1}{\sqrt{|N_u||N_i|}}\,e_i^{(k)},\quad
e_i^{(k+1)}=\sum_{u\in N_i}\frac{1}{\sqrt{|N_i||N_u|}}\,e_u^{(k)}
\]

Final embeddings are layer averages \(\sum_{k=0}^{K}\alpha_k e^{(k)}\). No feature transforms or nonlinearities inside the graph steps.

**located evidence.** LightGCN paper: He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*, [arXiv:2002.02126](https://arxiv.org/abs/2002.02126). Course hub: [web.stanford.edu/class/cs224w/](https://web.stanford.edu/class/cs224w/). RecSys slides example: [snap.stanford.edu/.../13-recsys.pdf](http://snap.stanford.edu/class/cs224w-2021/slides/13-recsys.pdf).

### 2.2 Yale CPSC 583 — Deep Learning on Graph-Structured Data

**source material.** Rex Ying. Theory half: distributed embeddings, generative graph models, hyperbolic spaces for hierarchies. Applied half: recommenders, biology, multi-agent reasoning; GNNExplainer for subgraph-level explanation.

**located evidence.** Course listing / instructor site: [cs.yale.edu/homes/ying-rex/](https://www.cs.yale.edu/homes/ying-rex/).

### 2.3 University of Minnesota — Advanced Recommender Systems

**source material.** Joseph A. Konstan, Michael D. Ekstrand. Matrix factorization, hybrids, LensKit; emphasis on metrics, evaluation methodology, and production-minded CF pipelines rather than GNN theory.

| Curriculum | Focus (source material) | Competency claimed |
| --- | --- | --- |
| Stanford CS224W | NGCF, LightGCN, GIN, GNN scaling | Expressive scalable graph convolutions |
| Yale CPSC 583 | Inductive SAGE, hyperbolic, GNNExplainer | Explainable / non-Euclidean graph models |
| Minnesota specialization | MF, hybrids, LensKit | Production collaborative pipelines + eval |

---

## 3. Conference tutorials (syllabus index)

### 3.1 IJCAI 2021 — Deep Learning for Recommendations

**source material / located evidence.** Tutorial site: [advanced-recommender-systems.github.io/ijcai2021-tutorial/](https://advanced-recommender-systems.github.io/ijcai2021-tutorial/). Speakers cited: Wenqi Fan, Xiangyu Zhao, Dawei Yin, Jiliang Tang.

Syllabus segments (source): intro; deep RS fundamentals; RL for RS; GNNs for recommendations; AutoML for RS; adversarial attacks; trustworthy RecSys (fairness / popularity bias).

### 3.2 WSDM 2022 — GNNs for Recommender Systems

**source material.** Chen Gao, Xiang Wang, Xiangnan He, Yong Li. Single-behavior → multi-behavior GNNs (e.g. MBRec): browse / cart / favorite / purchase hierarchies; social + temporal dynamics; sparsity via multi-typed propagation.

**located evidence (related).** Multi-behavior GNN survey/work often cited in this line: e.g. [arXiv:2302.08678](https://arxiv.org/pdf/2302.08678) (*Multi-Behavior Graph Neural Networks for Recommender System*). Slides linked in source: [staff.ustc.edu.cn/~hexn/slides/wsdm22_tutorial_gnn_rec.pdf](http://staff.ustc.edu.cn/~hexn/slides/wsdm22_tutorial_gnn_rec.pdf).

### 3.3 LoG 2023 — Reproducibility, Topology, Node Representation

**source material.** Tommaso Di Noia, Claudio Pomo, Daniele Malitesta. Three themes:

1. **Reproducibility** — Elliot framework; complex GNNs sometimes fail to beat strong shallow MF.
2. **Topology** — power-law degree bias → popular items dominate message passing → diversity collapse.
3. **Node representation** — from scratch vs multimodal pretrained item descriptors.

**located evidence.** [arXiv:2310.11270](https://arxiv.org/abs/2310.11270) (*Graph Neural Networks for Recommendation: Reproducibility, Graph Topology, and Node Representation*).

### 3.4 KDD 2022 — Graph Representation Learning for Web-Scale RecSys

**source material.** Ahmed El-Kishky, Michael Bronstein, Ying Xiao, Aria Haghighi. Homogeneous vs heterogeneous graphs; random-walk embeddings (node2vec DFS/BFS); L1 (direct co-purchase) vs L2 (shared neighbor) proximity; embeddings as inputs to candidate generation / ranking.

**located evidence.** Talk PDF cited in source: [ahelk.github.io/talks/kdd22/kdd_talk.pdf](https://ahelk.github.io/talks/kdd22/kdd_talk.pdf).

### 3.5 KDD 2020 — Marketplace & Automated RecSys

**source material.** Part A (Mehrotra, Carterette / Spotify): multi-stakeholder marketplace, Pareto scalarization, multi-objective bandits. Part B (Yong Li, Quanming Yao): AutoML for feature interaction, GraphNAS, automated KG embeddings.

**located evidence.** [sites.google.com/view/kdd20-marketplace-autorecsys/](https://sites.google.com/view/kdd20-marketplace-autorecsys/).

| Tutorial | Speakers (source) | Primary content |
| --- | --- | --- |
| IJCAI 2021 | Fan, Zhao, Yin, Tang | Deep RS, RL, GNN, AutoML, attacks, trust |
| WSDM 2022 | Gao, Wang, He, Li | Multi-behavior GNN, social, temporal |
| LoG 2023 | Di Noia, Pomo, Malitesta | Elliot repro, topology bias, multimodal |
| KDD 2022 | El-Kishky et al. | Walks, proximity, web-scale graph learning |
| KDD 2020 | Mehrotra/Carterette; Li/Yao | Marketplace MOO; AutoML / GraphNAS |

---

## 4. Production-scale pipelines

### 4.1 Multi-stage serving pattern

**source material.** Canonical industrial funnel:

```
Interaction log streams
  → GPU ETL (NVTabular)
  → Candidate generation (two-tower / GNN ANN)
  → Deep ranking (Wide&Deep / DeepFM / DIN)
  → Re-rank / business rules / diversity
  → Triton (or equivalent) low-latency serve
```

Phases: retrieval → filter/feature-store → rank → order.

### 4.2 RecSys 2022 — Ray / RLlib

**source material.** Treat recommend→feedback as an MDP; optimize long-horizon engagement with distributed multi-agent RL rather than static pointwise targets. Tutorial listing: [recsys.acm.org/recsys22/tutorials/](https://recsys.acm.org/recsys22/tutorials/).

### 4.3 NVIDIA Merlin stack

**source material.**

| Component | Role | Pipeline stage |
| --- | --- | --- |
| NVTabular | GPU ETL, high-cardinality categoricals | Preprocess |
| Merlin Models | NCF, VAE-CF, Wide&Deep, DLRM | Rank |
| Transformers4Rec | Session / sequential transformers | Seq retrieve / match |
| Triton | Concurrent models, dynamic batching | Online serve |

**KDD Cup 2023 (Amazon multilingual next-product)** — source claims Merlin team win via: Stage 1 co-visitation + contrastive multilingual product embeddings; Stage 2 transformer + GBDT rerank on titles/descriptions/history. Treat as secondary case study until methods write-up verified.

---

## 5. Industrial case studies (architectural mechanisms)

### 5.1 Alibaba MIND / ComiRec — multi-interest

**source material.** Single-vector sequential history fails under diverse interests. ComiRec extracts \(K\) interest vectors via self-attention (SA) or capsule dynamic routing (DR); training picks \(v_t=\arg\max_k v_k^\top e_i\) nearest the positive item; sampled-softmax NLL; inference runs \(K\) parallel retrievals then diversity aggregation.

Open discussion / blog cited: [Alibaba Cloud — Controllable Multi-Interest Framework](https://www.alibabacloud.com/blog/596749).

### 5.2 Pinterest PinSage / PinnerSage

**source material.** PinSage: importance-sampled random-walk neighborhoods + hard negatives on ~3B pins / 18B edges scale. PinnerSage: Ward hierarchical clustering of a user’s pin embeddings; medoids as multi-topic user vectors.

**located evidence (secondary).** PinSage engineering post: [Pinterest Engineering — PinSage](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48). PinnerSage PDF: [cs.stanford.edu/.../pinnersage-kdd20.pdf](https://cs.stanford.edu/people/jure/pubs/pinnersage-kdd20.pdf). KDD 2018 PinSage paper listing: [kdd.org/.../graph-convolutional-neural-networks-for-web-scale-recommender-systems](https://www.kdd.org/kdd2018/accepted-papers/view/graph-convolutional-neural-networks-for-web-scale-recommender-systems).

### 5.3 Amazon — asymmetric complementary GNN

**source material.** Complementary edges are directed (phone→case ≠ case→phone). Dual embeddings per item \(h^{\mathrm{src}}\), \(h^{\mathrm{tgt}}\); score \(\langle h_u^{\mathrm{src}}, h_v^{\mathrm{tgt}}\rangle\); contrastive loss favors outbound edges, penalizes inbound negatives.

**located evidence (secondary).** [Amazon Science — Using GNNs to recommend related products](https://www.amazon.science/blog/using-graph-neural-networks-to-recommend-related-products). Related arXiv HTML cited in source list: [arXiv HTML 2508.14059](https://arxiv.org/html/2508.14059) (Amazon co-purchase GNN — verify relevance before citing as the Amazon complementary paper).

### 5.4 Uber Eats — heterogeneous GraphSAGE

**source material.** Users / restaurants / cuisines / dishes; inductive GraphSAGE; reported AUC lift 78%→87% vs non-graph baselines (**unverified secondary claim**).

| Platform | Model | Mechanism (source) | Claimed advantage | Scaling limit (source) |
| --- | --- | --- | --- | --- |
| Alibaba | ComiRec | Multi-interest SA / capsules | Breaks single-vector bottleneck | Sequence length |
| Pinterest | PinSage | Walk-localized conv + hard neg | Bounded neighborhoods | Static bipartite scale |
| Pinterest | PinnerSage | Ward medoids | Multi-topic users | Update frequency |
| Amazon | Asymmetric GNN | Dual src/tgt embeddings | Directed complementary | 2× embedding storage |
| Uber Eats | GraphSAGE | Heterogeneous inductive agg | Local + spatial signals | Regional density |

---

## 6. Architectural strategic directions (from source)

**source material — three directives:**

1. **Decouple** linear/simplified graph propagation from nonlinear feature-interaction rankers (LightGCN-style embeddings → DeepFM/DIN).
2. **Multi-interest** user representations (ComiRec / PinnerSage-class) instead of one vector.
3. **Topology-aware evaluation** — stratify by degree / popularity so models are not scored only on head items (LoG 2023 theme).

---

## 7. Critical mapping onto BrainAPI

This section is **analysis**, not from the source review.

### 7.1 Category difference (claim evaluation)

| Dimension | Landscape above | BrainAPI core (**located evidence**) |
| --- | --- | --- |
| Job | Rank items for users from interaction logs | Event KG from text/triples + evidence retrieval |
| Graph | User–item bipartite / multi-behavior | Actor `-[:MADE]->` Event `-[:TARGETED]->` Object |
| Learning | Offline train (GNN, two-tower, sampled softmax) | Encode-only embeddings; synergies = heuristic graph+cosine ([`entity_sibilings.py`](../../src/core/search/entity_sibilings.py)) |
| Serve | Retrieval → rank → re-rank | `/retrieve/context`, `/retrieve/entity/synergies` |
| Metrics | NDCG / CTR / next-item | LoCoMo / BEAM memory QA; synergies unmeasured ([`04`](04-evaluation-and-applications.md) §12) |

**Conclusion supported by evidence:** BrainAPI synergies are **not** LightGCN/Merlin. Equating them is a category error.

### 7.2 What transfers as metaphor (ideas — not validated)

| Industrial idea | Plausible BrainAPI analogue (**idea**) |
| --- | --- |
| Candidate generation | Synergies + neighbors + hops |
| Asymmetric complementary | Directed event-hub walks (query as source vs target) |
| Multi-interest | Cluster recent event neighborhoods / medoids |
| Decouple propagate vs rank | Keep graph walk cheap; rank with association_score + diversity |
| Topology-aware eval | Stratify synergy seeds by node degree / type frequency |
| Merlin train/serve | **Plugin-only** parallel pipeline — plugins are additive, cannot overwrite `/ingest/` or synergies |

### 7.3 What does **not** transfer without new data + new stack

- Training LightGCN/PinSage inside default ingest (no interaction-tensor train loop; would risk LoCoMo/BEAM if forced onto `/ingest/`).
- NVTabular / Triton as core dependencies.
- Marketplace multi-objective bandits without supplier/consumer logs.
- Claiming GNN SOTA by shipping synergies heuristics.

### 7.4 Adversarial / minority points preserved from LoG 2023

- Complex GNNs can lose to strong MF under Elliot — **do not** assume deeper = better.
- Power-law bias harms diversity — any BrainAPI recommend API should measure long-tail coverage, not only association_score on hubs.
- Multimodal item features matter industrially; BrainAPI already has text descriptions — fusion is optional enrichment, not a reason to train PinSage in core.

### 7.5 Decision log (cross-link)

| Decision | Rationale | Rejected alternative |
| --- | --- | --- |
| Core path = event-KG relatedness + deterministic `/ingest/structured` | Matches BrainAPI data model; preserves benches | Fold LightGCN into `/ingest/` |
| Industrial GNN/Merlin = optional **plugin** parallel train/serve | Plugin API is additive only | Overwrite synergies / default ingest |
| Measure synergies with structural metrics first | [`04`](04-evaluation-and-applications.md) §4.9 | Gate on LLM serendipity alone |

---

## 8. Source URL registry (from provided material)

Access date for this index: **2026-08-05**. URLs are as supplied; not all were re-fetched end-to-end.

### Tutorials & courses

- IJCAI 2021 DeepRecSys — https://advanced-recommender-systems.github.io/ijcai2021-tutorial/
- CS224W — https://web.stanford.edu/class/cs224w/
- CS224W recsys slides — http://snap.stanford.edu/class/cs224w-2021/slides/13-recsys.pdf
- Yale Rex Ying — https://www.cs.yale.edu/homes/ying-rex/
- KDD 2020 marketplace/AutoRecSys — https://sites.google.com/view/kdd20-marketplace-autorecsys/
- KDD 2022 talk PDF — https://ahelk.github.io/talks/kdd22/kdd_talk.pdf
- WSDM 2022 GNN-Rec slides — http://staff.ustc.edu.cn/~hexn/slides/wsdm22_tutorial_gnn_rec.pdf
- RecSys 2022 tutorials — https://recsys.acm.org/recsys22/tutorials/

### Papers / arXiv (verified or commonly cited)

- LightGCN — https://arxiv.org/abs/2002.02126
- LoG GNN-Rec tutorial paper — https://arxiv.org/abs/2310.11270
- Multi-behavior GNN RS — https://arxiv.org/pdf/2302.08678
- Amazon co-purchase GNN (HTML) — https://arxiv.org/html/2508.14059
- PinnerSage — https://cs.stanford.edu/people/jure/pubs/pinnersage-kdd20.pdf
- PinSage KDD 2018 listing — https://www.kdd.org/kdd2018/accepted-papers/view/graph-convolutional-neural-networks-for-web-scale-recommender-systems

### Industrial / engineering posts

- PinSage (Pinterest Eng) — https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48
- Amazon complementary GNN — https://www.amazon.science/blog/using-graph-neural-networks-to-recommend-related-products
- Alibaba ComiRec blog — https://www.alibabacloud.com/blog/596749
- NVIDIA Merlin overview — https://developer.nvidia.com/blog/how-to-build-a-winning-recommendation-system-part-2-deep-learning-for-recommender-systems/
- Meta Intelligence CF→Deep overview — https://www.meta-intelligence.tech/en/insight-recommender-systems

### Additional URLs from the source list (not individually verified this pass)

YouTube CS224W lectures (17.1 scale, 9.2 GIN, 2023 GNN-RecSys, Trustworthy Graph AI); Coursera recommender courses; RecSys 2021/2024 pages; NVIDIA Merlin preprocessing Medium post; Alibaba M2GRL blog; YouTube DNN recommendations paper; AssemblyAI / Plasticity GNN trend posts; alphaXiv mirrors of 2310.11270; NSF PAR “How Does Message Passing Improve Collaborative Filtering?”; Emerald incremental graph RS survey.

**Search limits:** Opaque `[cite: N]` keys in the source were not resolved to a formal bibliography. Industrial AUC / catalog-size figures remain secondary until primary methods sections are checked. This document does **not** establish novelty of any BrainAPI design.

---

## 9. Next actions (labeled)

| Action | Type |
| --- | --- |
| Keep this file as the external landscape index for recommenders | **decision** (documentation) |
| Core BrainAPI work: structured dual-mode ingest + synergies harden + `/retrieve/recommend` | **proposal** — see feasibility plan Phases 0–3 |
| Optional `plugins/recsys-gnn/` for LightGCN/Merlin-class train/serve | **proposal** — Phase 4; additive only |
| `plugins/recsys-gnn` LightGCN export→train→`/recsys/recommend` (BrainAPI = KB) | **decision** — see [`16-recsys-eval-protocol.md`](16-recsys-eval-protocol.md) |
| Do not change default `/ingest/` or LoCoMo/BEAM brains for this landscape | **decision** |
| If citing Uber/Amazon metrics in papers or README, verify primary sources first | **further search** |
