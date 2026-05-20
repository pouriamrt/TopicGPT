# Migration: TopicGPT 0.0.6 → 1.0

> v1.0 is a full rewrite. The old code is preserved under [`legacy/src/`](./legacy/src/) for reference but is no longer imported by the package. This guide maps the v0.0.6 surface to the v1.0 surface.

## Defaults that changed

| | v0.0.6 | v1.0 |
|---|---|---|
| Chat model | `gpt-3.5-turbo-16k` (retired) | `gpt-5.4-mini` |
| Embedding model | `text-embedding-ada-002` (retired) | `text-embedding-3-small` |
| Python | 3.11 | **3.13** |
| Build | `setuptools` + `setup.py` | `hatchling` + `pyproject.toml` (uv-managed) |
| LLM I/O | free-text completions | Pydantic structured outputs (`responses.parse`) |
| Tests | none | `pytest`, 137+ tests, 95% coverage |
| Lint / types | none | `ruff` + `mypy --strict` |

## Class / module map

| v0.0.6 | v1.0 |
|---|---|
| `TopicGPT.TopicGPT(...)` | `topicgpt.TopicModel(embedder=..., reducer=..., clusterer=..., ...)` |
| `Clustering.Clustering_and_DimRed(...)` | `topicgpt.UMAPReducer(...)` + `topicgpt.HDBSCANClusterer(...)` (separated) |
| `GetEmbeddingsOpenAI.GetEmbeddingsOpenAI(...)` | `topicgpt.OpenAIEmbedder(OpenAIEmbeddingConfig(...))` |
| `ExtractTopWords` (TF-IDF + cosine) | `topicgpt.CTFIDFVectorizer` + `topicgpt.CosineSimilarityScorer` + `topicgpt.KeyBERTRepresenter` |
| `TopwordEnhancement` (gpt-3.5 free text) | `topicgpt.LLMRepresenter` (gpt-5.4-mini, Pydantic-validated `TopicLabel`) |
| `TopicPrompting` (legacy function-calling) | (removed — replace with direct LLM calls + structured outputs) |
| `TopicRepresentation.Topic` | `topicgpt.Topic` (frozen dataclass) |
| `tm.print_topics()` | `print(tm.get_topic_info().to_string(index=False))` |
| `tm.fit(corpus)` | `tm.fit(documents)` |
| `tm.prompt(query)` | `tm.find_topics(query, top_k=...)` |
| `tm.visualize_clusters()` | `topicgpt.visualize.visualize_topics(tm)` |

## Side-by-side

### v0.0.6

```python
from topicgpt.TopicGPT import TopicGPT

tm = TopicGPT(
    openai_api_key="sk-...",
    n_topics=20,
    openai_prompting_model="gpt-3.5-turbo-16k",
    embedding_model="text-embedding-ada-002",
)
tm.fit(corpus)
tm.print_topics()
ans, _ = tm.prompt("which topic talks about climate change?")
```

### v1.0

```python
from topicgpt import (
    CTFIDFVectorizer,
    KMeansClusterer, KMeansConfig,
    KeyBERTRepresenter,
    LLMRepresenter,
    OpenAIEmbedder, OpenAIEmbeddingConfig,
    TopicModel,
    UMAPReducer,
)

tm = TopicModel(
    embedder=OpenAIEmbedder(OpenAIEmbeddingConfig(model="text-embedding-3-small")),
    reducer=UMAPReducer(),
    clusterer=KMeansClusterer(KMeansConfig(n_clusters=20)),
    vectorizer=CTFIDFVectorizer(),
    representers=[KeyBERTRepresenter(), LLMRepresenter()],
)
tm.fit(corpus)
print(tm.get_topic_info())

# Search topics by semantic similarity to a query (returns top-k):
hits = tm.find_topics("climate change", top_k=3)
for topic, score in hits:
    print(score, topic.label, topic.keywords)
```

## Things explicitly *removed*

- **`TopicPrompting`** (1281 lines): legacy OpenAI function-calling. The same use-cases are now solved by:
    - **Split / merge / delete topics**: re-run the pipeline with adjusted `n_clusters`, or post-process the labels array yourself. Simpler and reproducible.
    - **Compare / describe topics**: `tm.get_topic(id)` + `tm.find_topics(query)`.
    - **General Q&A over topics**: call the OpenAI chat API directly with topic context — there's no single right wrapper.
- **Vocab embeddings via OpenAI**: previously, every unique word was embedded. v1.0 uses c-TF-IDF by default (much cheaper) and exposes `CosineSimilarityScorer` if you still want the twin-embedding recipe.

## Environment

Set `OPENAI_API_KEY` (or any `TOPICGPT_*` env var):

```bash
export OPENAI_API_KEY=sk-...
# optional:
export TOPICGPT_CACHE_DIR=.topicgpt_cache
export TOPICGPT_MAX_RETRIES=4
export TOPICGPT_REQUEST_TIMEOUT_S=60
```

## CLI

v0.0.6 had no CLI. v1.0 ships one:

```bash
topicgpt fit  --input docs.txt --out ./model --n-topics 20
topicgpt info ./model
topicgpt eval ./model --docs docs.txt --top-n 10
topicgpt viz  ./model --out chart.html --kind barchart
```
