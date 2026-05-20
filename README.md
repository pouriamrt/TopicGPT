# TopicGPT

> Modular LLM-driven topic modelling. Embed → reduce → cluster → vectorize → represent → label. OpenAI-latest defaults (`gpt-5.4-mini` for chat, `text-embedding-3-small` for embeddings).

**Status:** v1.0 rewrite in progress. See [PLAN.md](./PLAN.md) for the phased roadmap.

While traditional topic models extract topics as flat lists of top-words (`["lion", "leopard", "rhino", "elephant"]`), TopicGPT produces rich, named topics with LLM-generated labels, descriptions, and a typed Python API.

## Core Capabilities (v1.0 target)

- Modular pipeline — swap embedder / reducer / clusterer / vectorizer / representer independently
- OpenAI structured outputs (Pydantic schemas) for every LLM call
- Class-based TF-IDF (BERTopic-style) + KeyBERT-MMR top-word extraction
- Hierarchical + reduced topic views
- Evaluation suite: NPMI, Cv, UMass, topic diversity, OCTIS-style bench
- Plotly visualisations: scatter, barchart, dendrogram, heatmap
- CLI: `topicgpt fit | info | eval | viz`
- Type-safe end-to-end: `mypy --strict` clean, Pydantic configs

## Install (once v1.0 lands)

```bash
uv add topicgpt
# or
pip install topicgpt
```

## Quickstart (planned API)

```python
from topicgpt import TopicModel

model = TopicModel.from_config(
    embed="openai:text-embedding-3-small",
    reduce="umap",
    cluster="hdbscan",
    represent=["ctfidf", "llm:gpt-5.4-mini"],
)
topics = model.fit_transform(documents)
print(model.get_topic_info())
```

## Development

```bash
uv sync --all-extras --group dev
uv run ruff check .
uv run mypy src/topicgpt
uv run pytest
```

## Migrating from 0.0.6

See [MIGRATION.md](./MIGRATION.md) (lands in Phase 9). Original 0.0.6 code is preserved under [`legacy/src/`](./legacy/src/) for reference.

## References

- Original (Reuter et al.): https://github.com/ArikReuter/TopicGPT
- BERTopic: https://maartengr.github.io/BERTopic/
- TopicGPT (Pham et al., 2024): https://arxiv.org/abs/2311.01449
