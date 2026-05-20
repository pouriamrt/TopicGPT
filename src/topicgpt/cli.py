"""TopicGPT command-line interface.

Subcommands:
    fit      Train a model on a text file (one document per line).
    info     Print topic-info table from a saved model directory.
    eval     Compute NPMI / diversity over a fitted model.
    viz      Render a single visualisation to HTML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from topicgpt import (
    CTFIDFVectorizer,
    HDBSCANClusterer,
    HDBSCANConfig,
    KeyBERTRepresenter,
    KMeansClusterer,
    KMeansConfig,
    OpenAIEmbedder,
    OpenAIEmbeddingConfig,
    TopicModel,
    UMAPConfig,
    UMAPReducer,
)
from topicgpt.clustering import Clusterer
from topicgpt.evaluation import inverted_rbo, npmi, proportion_unique_words
from topicgpt.visualize import (
    visualize_barchart,
    visualize_heatmap,
    visualize_hierarchy,
    visualize_topics,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="TopicGPT v1.0 — LLM-driven modular topic modelling.",
)
_console = Console()


@app.command()
def fit(
    input_file: Annotated[Path, typer.Option("--input", help="One document per line.")],
    out: Annotated[Path, typer.Option(help="Directory to save the fitted model.")],
    n_topics: Annotated[int | None, typer.Option(help="Force fixed number of topics.")] = None,
    embedding_model: Annotated[
        str, typer.Option(help="OpenAI embedding model.")
    ] = "text-embedding-3-large",
    embedding_dim: Annotated[
        int | None, typer.Option(help="Output dim override.")
    ] = None,
) -> None:
    """Train a TopicModel and persist it to disk."""
    docs = _read_lines(input_file)
    _console.print(f"[bold]Loaded[/bold] {len(docs)} documents")

    embedder = OpenAIEmbedder(
        OpenAIEmbeddingConfig(model=embedding_model, dimensions=embedding_dim)
    )
    clusterer: Clusterer = (
        HDBSCANClusterer(HDBSCANConfig())
        if n_topics is None
        else KMeansClusterer(KMeansConfig(n_clusters=n_topics))
    )

    model = TopicModel(
        embedder=embedder,
        reducer=UMAPReducer(UMAPConfig()),
        clusterer=clusterer,
        vectorizer=CTFIDFVectorizer(),
        representers=[KeyBERTRepresenter()],
    )
    model.fit(docs)
    model.save(out)
    _console.print(f"[green]Saved model to {out}[/green]")
    _console.print(model.get_topic_info().to_string(index=False))


@app.command()
def info(
    model_dir: Annotated[Path, typer.Argument(help="Path to saved model.")],
) -> None:
    """Print topic info from a saved model (uses a stub embedder for load)."""
    model = TopicModel.load(model_dir, embedder=_make_default_embedder())
    _console.print(model.get_topic_info().to_string(index=False))


@app.command(name="eval")
def eval_cmd(
    model_dir: Annotated[Path, typer.Argument(help="Saved model directory.")],
    docs: Annotated[Path, typer.Option("--docs", help="Reference corpus.")],
    top_n: Annotated[int, typer.Option(help="Words per topic to score.")] = 10,
) -> None:
    """Compute coherence + diversity metrics."""
    model = TopicModel.load(model_dir, embedder=_make_default_embedder())
    reference = _read_lines(docs)
    topic_words = [list(t.keywords[:top_n]) for t in model.topics_]
    _console.print(f"NPMI:      {npmi(reference, topic_words, top_n=top_n):+.4f}")
    _console.print(f"Diversity: {proportion_unique_words(topic_words, top_n=top_n):.4f}")
    _console.print(f"RBO:       {inverted_rbo(topic_words, top_n=top_n):.4f}")


@app.command()
def viz(
    model_dir: Annotated[Path, typer.Argument(help="Saved model directory.")],
    out: Annotated[Path, typer.Option(help="Output HTML path.")],
    kind: Annotated[
        str, typer.Option(help="One of: scatter | barchart | hierarchy | heatmap.")
    ] = "barchart",
) -> None:
    """Render a single visualisation to standalone HTML."""
    model = TopicModel.load(model_dir, embedder=_make_default_embedder())
    if kind == "scatter":
        fig = visualize_topics(model)
    elif kind == "barchart":
        fig = visualize_barchart(model)
    elif kind == "hierarchy":
        fig = visualize_hierarchy(model)
    elif kind == "heatmap":
        fig = visualize_heatmap(model)
    else:
        raise typer.BadParameter(f"Unknown kind: {kind}")
    fig.write_html(str(out))
    _console.print(f"[green]Wrote[/green] {out}")


# ----------------------------------------------------------------------
# Helpers


def _read_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def _make_default_embedder() -> OpenAIEmbedder:
    """Default embedder used when loading a saved model from CLI."""
    return OpenAIEmbedder()


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":
    main()
