# TopicGPT v1.0 — Modernization Plan

> Full rewrite, BERTopic-style modular pipeline, OpenAI-latest defaults (`gpt-5.4-mini` for chat, `text-embedding-3-small` for embeddings), evaluation suite, Pydantic-typed structured outputs. Python 3.13. uv-managed.

## Goals

1. Modular, pluggable pipeline: `embed → reduce → cluster → vectorize → represent → label`
2. Type-safe end-to-end: Pydantic configs, Protocol interfaces, mypy --strict clean
3. OpenAI structured outputs (`response_format=BaseModel`) replace legacy function-calling
4. Evaluation suite (NPMI, Cv, UMass, diversity, OCTIS-style bench)
5. 80%+ test coverage, CI, ruff + mypy clean
6. CLI for fit / inspect / eval / visualize
7. Old code archived under `legacy/`; new package under `src/topicgpt/`

## Non-Goals (this revision)

- Multi-provider (litellm) — explicitly chosen OpenAI-only
- LLM-direct topic generation (Pham 2024) — clustering-based path only
- Multimodal — text-only
- Online / streaming topics — batch only

## Target Layout

```
topicgpt/
├── pyproject.toml          # uv, ruff, mypy, pytest configured
├── README.md               # rewritten
├── MIGRATION.md            # 0.0.6 → 1.0 guide
├── PLAN.md                 # this file
├── src/topicgpt/
│   ├── __init__.py         # public API: TopicModel, Topic, etc.
│   ├── config.py           # pydantic Settings + per-stage configs
│   ├── exceptions.py
│   ├── topic.py            # Topic dataclass (frozen)
│   ├── pipeline.py         # TopicModel orchestrator
│   ├── embeddings/
│   │   ├── base.py         # Embedder Protocol
│   │   ├── openai.py       # text-embedding-3-small, batched, cached
│   │   └── sbert.py        # sentence-transformers fallback
│   ├── reduction/
│   │   ├── base.py         # DimReducer Protocol
│   │   ├── umap_reducer.py
│   │   └── pca.py
│   ├── clustering/
│   │   ├── base.py         # Clusterer Protocol
│   │   ├── hdbscan_clusterer.py
│   │   ├── kmeans.py
│   │   └── agglomerative.py
│   ├── vectorization/
│   │   ├── base.py         # Vectorizer Protocol
│   │   ├── ctfidf.py       # class-based TF-IDF
│   │   └── cosine.py
│   ├── representation/
│   │   ├── base.py         # Representer Protocol
│   │   ├── keybert.py      # KeyBERT-style MMR keyword extraction
│   │   ├── llm.py          # OpenAI chat + structured outputs
│   │   └── schemas.py      # Pydantic schemas for LLM outputs
│   ├── evaluation/
│   │   ├── coherence.py    # NPMI, Cv, UMass via gensim
│   │   ├── diversity.py    # topic diversity, inverted RBO
│   │   └── bench.py        # OCTIS-style harness
│   ├── visualize/
│   │   └── plots.py        # plotly: barchart, scatter, dendrogram, heatmap
│   └── cli.py              # typer entrypoint
├── tests/
│   ├── unit/
│   └── integration/
├── examples/
│   ├── quickstart.py
│   ├── eval_20newsgroups.py
│   └── custom_pipeline.py
└── legacy/                 # archived v0.0.6 files (read-only)
    └── src/                # original Clustering.py, TopicGPT.py, etc.
```

## Public API (target)

```python
from topicgpt import TopicModel, OpenAIConfig

model = TopicModel.from_config(
    embed="openai:text-embedding-3-small",
    reduce="umap",
    cluster="hdbscan",
    represent=["ctfidf", "llm:gpt-5.4-mini"],
)
topics = model.fit_transform(documents)

# Inspect
model.get_topic_info()           # DataFrame
model.get_topic(0)               # Topic
model.find_topics("climate")     # similarity search

# Hierarchies
hierarchy = model.hierarchical_topics()
model.reduce_topics(n=10)

# Eval
from topicgpt.evaluation import coherence, diversity
coherence.npmi(model, documents)
diversity.score(model)

# Visualize
model.visualize_topics()         # plotly Figure
model.visualize_hierarchy()
model.visualize_barchart()

# Persist
model.save("./out")
TopicModel.load("./out")
```

## Phases

Each phase = one PR-sized chunk with explicit acceptance criteria. Gate = `ruff check && mypy --strict && pytest -q` passes.

### Phase 1 — Skeleton + Tooling
**Files**: `pyproject.toml`, `.github/workflows/ci.yml`, `.gitignore`, `src/topicgpt/__init__.py`, `legacy/` move.
- uv-managed, Python 3.13, deps: pydantic, openai, tiktoken, numpy, pandas, scikit-learn, hdbscan, umap-learn, gensim, sentence-transformers, plotly, typer, tqdm, hishel.
- Dev deps: ruff, mypy, pytest, pytest-cov, pytest-mock, hypothesis, bandit.
- Configure ruff (line-length 100, all rules sane), mypy --strict, pytest with `--cov=src/topicgpt --cov-fail-under=80`.
- CI: matrix on Python 3.13, run lint+type+test.
- Archive `src/*.py` → `legacy/src/` (verbatim, untouched).
- **Accept**: `uv sync && uv run pytest` runs (zero tests yet but exits 0 with `--cov-fail-under=0` override for now).

### Phase 2 — Core types + config + exceptions
**Files**: `src/topicgpt/config.py`, `src/topicgpt/topic.py`, `src/topicgpt/exceptions.py`, tests.
- `Settings(BaseSettings)` reads `OPENAI_API_KEY`, `TOPICGPT_CACHE_DIR`, etc. from env.
- `OpenAIConfig`, `UMAPConfig`, `HDBSCANConfig`, `LLMRepresentationConfig` — frozen `BaseModel`s.
- `@dataclass(frozen=True) class Topic` with `id`, `keywords`, `representative_docs`, `label`, `description`, `size`, `centroid`.
- Domain exceptions: `TopicGPTError`, `EmbeddingError`, `ClusteringError`, `RepresentationError`.
- **Accept**: 100% coverage on these files. Hypothesis-based tests for Topic equality / hash.

### Phase 3 — Embedding layer
**Files**: `embeddings/{base,openai,sbert}.py`, tests.
- `Embedder` Protocol: `embed(texts: Sequence[str]) -> NDArray[float32]`, `dim: int`.
- `OpenAIEmbedder`: default `text-embedding-3-small`, configurable `dimensions`, batched (max 2048 per request), retries via tenacity, optional on-disk cache (hishel keyed by hash(text, model, dim)).
- `SBERTEmbedder`: wraps sentence-transformers, default `BAAI/bge-large-en-v1.5`.
- **Accept**: unit tests with mocked OpenAI client; cache hit/miss test; batching boundary test.

### Phase 4 — Reduction + Clustering
**Files**: `reduction/{base,umap_reducer,pca}.py`, `clustering/{base,hdbscan_clusterer,kmeans,agglomerative}.py`, tests.
- Protocols `DimReducer.fit_transform`, `Clusterer.fit_predict` returning `NDArray[int]` (-1 = outlier).
- UMAP defaults: 5 dims, 15 neighbors, cosine; HDBSCAN defaults: min_cluster_size=30, euclidean.
- KMeans + Agglomerative for fixed-k scenarios.
- **Accept**: synthetic gaussian-blob test data, assert recovered cluster count within tolerance.

### Phase 5 — Vectorization + Representation
**Files**: `vectorization/{base,ctfidf,cosine}.py`, `representation/{base,keybert,llm,schemas}.py`, tests.
- `CTFIDF`: BERTopic-style class-based TF-IDF (concat docs per cluster → TF-IDF over class corpus). Returns sparse matrix + vocab.
- `KeyBERTRepresenter`: MMR over candidate words ranked by cosine to topic centroid.
- `LLMRepresenter`: calls `gpt-5.4-mini` with structured output:
  ```python
  class TopicLabel(BaseModel):
      label: str = Field(max_length=80)
      description: str = Field(max_length=400)
      keywords: list[str] = Field(min_length=3, max_length=10)
  ```
  Uses `client.responses.parse(model="gpt-5.4-mini", response_format=TopicLabel, ...)`. Prompt template includes top words + N representative docs. Model name overridable via `LLMRepresentationConfig.model` (e.g., bump to `gpt-5.4` for higher-quality runs).
- **Accept**: mock OpenAI; verify Pydantic-validated response, prompt construction, MMR diversity.

### Phase 6 — Pipeline orchestrator + public API
**Files**: `pipeline.py`, `__init__.py`, integration tests.
- `TopicModel` composes Embedder, DimReducer, Clusterer, Vectorizer, list[Representer].
- `fit(documents)`, `fit_transform(documents)`, `transform(documents)` (assign new docs to existing topics via nearest centroid).
- `get_topic_info() -> pd.DataFrame`, `get_topic(id) -> Topic`, `find_topics(query, top_k) -> list[tuple[Topic, float]]`.
- `hierarchical_topics() -> pd.DataFrame` via scipy linkage on c-TF-IDF; `reduce_topics(n_or_threshold)`.
- `save(path)`, `load(path)` — uses joblib for sklearn, pickle for the rest, json for config.
- **Accept**: end-to-end test on 20newsgroups-mini (200 docs) with mocked LLM; round-trip save/load.

### Phase 7 — Evaluation suite
**Files**: `evaluation/{coherence,diversity,bench}.py`, tests.
- `coherence.npmi`, `coherence.cv`, `coherence.umass` via gensim CoherenceModel.
- `diversity.proportion_unique_words`, `diversity.inverted_rbo`.
- `bench.run_bench(model_factory, dataset_loader, metrics) -> BenchResult` with seeds, mean ± std.
- **Accept**: deterministic synthetic test computes NPMI by hand and matches gensim output within tolerance.

### Phase 8 — Visualization
**Files**: `visualize/plots.py`, smoke tests.
- `visualize_topics()` — UMAP-2D scatter (recomputed for viz; not the clustering UMAP).
- `visualize_barchart(top_n=8)` — top-N keywords per topic.
- `visualize_hierarchy()` — dendrogram.
- `visualize_heatmap()` — topic-topic similarity.
- All return `plotly.graph_objects.Figure`.
- **Accept**: each fn returns a Figure with expected trace count given a fixture model.

### Phase 9 — CLI + docs + examples
**Files**: `cli.py`, `README.md`, `MIGRATION.md`, `examples/*.py`.
- `topicgpt fit --input docs.txt --out ./model --n-topics 20`
- `topicgpt info ./model` → topic table
- `topicgpt eval ./model --docs docs.txt --metrics npmi,diversity`
- `topicgpt viz ./model --kind barchart --out chart.html`
- README: new quickstart, comparison table, citation, model defaults.
- MIGRATION.md: name mapping from v0.0.6 → v1.0.
- **Accept**: CLI smoke tests via typer.testing.CliRunner.

### Phase 10 — Polish + verify
- `mypy --strict src/topicgpt` clean.
- `bandit -r src/topicgpt` clean.
- `pytest --cov` ≥ 80%.
- `ruff check && ruff format --check`.
- Sentrux `health` re-baseline. Compare to old.
- README badges (CI, coverage, Python).
- Bump version to `1.0.0` in pyproject.toml.

## Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| OpenAI structured-outputs API surface drift | Pin `openai>=1.55`, use `responses.parse`; one adapter module isolates SDK calls |
| HDBSCAN wheels broken on Py 3.13 | Pin `hdbscan>=0.8.40` (3.13 wheels shipped Q2 2025); fall back to `sklearn.cluster.HDBSCAN` if import fails |
| UMAP slow on 100k+ docs | Document `n_neighbors`/`low_memory=True`; expose as config |
| LLM cost in tests | All Phase-5+ tests mock OpenAI; live calls only in `examples/` (opt-in) |
| Coverage gate too tight on viz | Mark `visualize/` as `# pragma: no cover` for the rendering branches; assert structure only |

## Deliverables Checklist

- [ ] `pyproject.toml` rewritten (uv, deps, ruff, mypy, pytest config)
- [ ] CI workflow green
- [ ] `src/topicgpt/` package with all modules
- [ ] `legacy/` archive
- [ ] `tests/` ≥ 80% cov
- [ ] `README.md` rewritten
- [ ] `MIGRATION.md` written
- [ ] `examples/quickstart.py` runs (with API key)
- [ ] CLI works (`topicgpt --help`)
- [ ] Tag `v1.0.0-rc1` ready to cut

## Order of Operations

Phases 1 → 10 sequential. Each phase committed separately. Gate at every phase: `uv run ruff check && uv run mypy src/topicgpt && uv run pytest`.
