#!/usr/bin/env python3
"""Create metadata-based visualizations from saved BERTopic outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import plotly.express as px


ROOT = Path(__file__).resolve().parents[1]
STUDENT_PALETTE = px.colors.qualitative.Safe + px.colors.qualitative.Bold + px.colors.qualitative.Pastel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "bertopic_sample_nomic_sensitive_lemmatized"),
        help="Directory containing topic_assignments.csv and topic_labels_llm.csv.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of largest non-outlier topics to show in filtered visualizations.",
    )
    return parser.parse_args()


def clean_label(label: object, fallback: str) -> str:
    value = str(label).strip()
    return value if value and value.lower() != "nan" else fallback


def short_title(title: str, max_len: int = 70) -> str:
    title = re.sub(r"\s+", " ", str(title)).strip()
    return title if len(title) <= max_len else f"{title[: max_len - 3]}..."


def load_data(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(output_dir / "topic_assignments.csv")
    labels_path = output_dir / "topic_labels_llm.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
    else:
        labels = pd.DataFrame(columns=["topic", "label", "top_words"])

    assignments["publication_year"] = pd.to_numeric(assignments["publication_year"], errors="coerce")
    assignments["decade"] = (assignments["publication_year"] // 10 * 10).astype("Int64")

    for column in ["topic", "label", "top_words"]:
        if column not in labels.columns:
            labels[column] = ""

    labels = labels[["topic", "label", "top_words"]].drop_duplicates("topic")
    data = assignments.merge(labels[["topic", "label"]], on="topic", how="left")
    data["topic_label"] = data.apply(
        lambda row: clean_label(row.get("label"), f"Topic {row['topic']}"),
        axis=1,
    )
    data.loc[data["topic"] == -1, "topic_label"] = "Outlier"
    data["topic_display"] = data.apply(
        lambda row: f"{row['topic']}: {row['topic_label']}",
        axis=1,
    )
    data["short_title"] = data["title"].map(short_title)
    return data, labels


def save_topic_review_table(data: pd.DataFrame, labels: pd.DataFrame, output_dir: Path) -> None:
    """Write a student-friendly topic table with labels, topic words, and metadata clues."""
    topic_data = data[data["topic"] != -1].copy()
    if topic_data.empty:
        return

    counts = (
        topic_data.groupby(["topic", "topic_label"])
        .size()
        .reset_index(name="chunks")
    )
    years = (
        topic_data.dropna(subset=["publication_year"])
        .groupby("topic")["publication_year"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "earliest_publication_year", "max": "latest_publication_year"})
    )
    doc_counts = (
        topic_data.groupby(["topic", "short_title"])
        .size()
        .reset_index(name="chunks_from_document")
        .sort_values(["topic", "chunks_from_document"], ascending=[True, False])
    )
    top_documents = (
        doc_counts.groupby("topic")
        .head(3)
        .groupby("topic")
        .apply(
            lambda rows: "; ".join(
                f"{row.short_title} ({int(row.chunks_from_document)} chunks)"
                for row in rows.itertuples(index=False)
            )
        )
        .reset_index(name="top_documents")
    )

    review = (
        counts.merge(labels[["topic", "top_words"]], on="topic", how="left")
        .merge(years, on="topic", how="left")
        .merge(top_documents, on="topic", how="left")
        .sort_values("chunks", ascending=False)
    )
    review = review[
        [
            "topic",
            "topic_label",
            "chunks",
            "top_words",
            "earliest_publication_year",
            "latest_publication_year",
            "top_documents",
        ]
    ]
    review.to_csv(output_dir / "topic_review_table.csv", index=False)


def top_topics(data: pd.DataFrame, top_n: int) -> list[int]:
    counts = data[data["topic"] != -1]["topic"].value_counts()
    return counts.head(top_n).index.tolist()


def save_topic_prevalence_by_decade(data: pd.DataFrame, viz_dir: Path, top_n: int) -> None:
    selected_topics = top_topics(data, min(top_n, 8))
    time_data = data[data["topic"].isin(selected_topics)].dropna(subset=["decade"]).copy()
    if time_data.empty:
        return

    counts = (
        time_data.groupby(["decade", "topic", "topic_display"])
        .size()
        .reset_index(name="chunks")
        .sort_values(["decade", "chunks"], ascending=[True, False])
    )
    totals = counts.groupby("decade")["chunks"].transform("sum")
    counts["share"] = counts["chunks"] / totals

    fig = px.line(
        counts,
        x="decade",
        y="share",
        facet_col="topic_display",
        facet_col_wrap=2,
        markers=True,
        labels={
            "decade": "Publication decade",
            "share": "Share of modeled chunks",
            "topic_display": "Topic",
        },
        title=f"Topic Trends by Publication Decade: Top {len(selected_topics)} Topics",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(line_color="#3366CC", marker_color="#3366CC")
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    fig.update_layout(showlegend=False, height=max(700, 230 * ((len(selected_topics) + 1) // 2)))
    fig.write_html(viz_dir / "topic_prevalence_by_decade.html", include_plotlyjs="cdn")


def save_topic_prevalence_grouped_bars(data: pd.DataFrame, viz_dir: Path, top_n: int) -> None:
    selected_topics = top_topics(data, min(top_n, 8))
    time_data = data[data["topic"].isin(selected_topics)].dropna(subset=["decade"]).copy()
    if time_data.empty:
        return

    counts = (
        time_data.groupby(["decade", "topic", "topic_display"])
        .size()
        .reset_index(name="chunks")
        .sort_values(["decade", "chunks"], ascending=[True, False])
    )
    totals = counts.groupby("decade")["chunks"].transform("sum")
    counts["share"] = counts["chunks"] / totals

    fig = px.bar(
        counts,
        x="decade",
        y="share",
        color="topic_display",
        barmode="group",
        color_discrete_sequence=STUDENT_PALETTE,
        labels={
            "decade": "Publication decade",
            "share": "Share of modeled chunks",
            "topic_display": "Topic",
        },
        title=f"Topic Shares by Publication Decade: Top {len(selected_topics)} Topics",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.write_html(viz_dir / "topic_prevalence_grouped_bars.html", include_plotlyjs="cdn")


def save_topic_decade_heatmap(data: pd.DataFrame, viz_dir: Path, top_n: int) -> None:
    selected_topics = top_topics(data, top_n)
    heat_data = data[data["topic"].isin(selected_topics)].dropna(subset=["decade"]).copy()
    if heat_data.empty:
        return

    counts = (
        heat_data.groupby(["topic_display", "decade"])
        .size()
        .reset_index(name="chunks")
    )
    pivot = counts.pivot(index="topic_display", columns="decade", values="chunks").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        labels={"x": "Publication decade", "y": "Topic", "color": "Chunks"},
        title=f"Topic-by-Decade Heatmap: Top {top_n} Topics",
    )
    fig.write_html(viz_dir / "topic_decade_heatmap.html", include_plotlyjs="cdn")


def save_document_topic_heatmap(data: pd.DataFrame, viz_dir: Path, top_n: int) -> None:
    selected_topics = top_topics(data, top_n)
    doc_data = data[data["topic"].isin(selected_topics)].copy()
    if doc_data.empty:
        return

    doc_data["document"] = doc_data.apply(
        lambda row: f"{int(row['publication_year']) if pd.notna(row['publication_year']) else 'n.d.'} | {row['short_title']}",
        axis=1,
    )
    counts = (
        doc_data.groupby(["document", "topic_display"])
        .size()
        .reset_index(name="chunks")
    )
    pivot = counts.pivot(index="document", columns="topic_display", values="chunks").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={"x": "Topic", "y": "Document", "color": "Chunks"},
        title=f"Document-by-Topic Heatmap: Top {top_n} Topics",
    )
    fig.write_html(viz_dir / "document_topic_heatmap.html", include_plotlyjs="cdn")


def save_sample_timeline(data: pd.DataFrame, viz_dir: Path) -> None:
    docs = (
        data[["doc_id", "author", "title", "publication_year", "url"]]
        .drop_duplicates("doc_id")
        .dropna(subset=["publication_year"])
        .copy()
    )
    if docs.empty:
        return

    docs["short_title"] = docs["title"].map(short_title)
    docs["publication_year"] = docs["publication_year"].astype(int)
    docs = docs.sort_values("publication_year")

    fig = px.scatter(
        docs,
        x="publication_year",
        y="short_title",
        hover_data=["author", "title", "url"],
        labels={"publication_year": "Publication year", "short_title": "Document"},
        title="Sample Documents by Publication Year",
    )
    fig.write_html(viz_dir / "sample_documents_timeline.html", include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    viz_dir = output_dir / "metadata_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    data, labels = load_data(output_dir)
    save_topic_review_table(data, labels, output_dir)
    save_topic_prevalence_by_decade(data, viz_dir, args.top_n)
    save_topic_prevalence_grouped_bars(data, viz_dir, args.top_n)
    save_topic_decade_heatmap(data, viz_dir, args.top_n)
    save_document_topic_heatmap(data, viz_dir, args.top_n)
    save_sample_timeline(data, viz_dir)

    print(f"Metadata visualizations written to {viz_dir}")


if __name__ == "__main__":
    main()
