#!/usr/bin/env python3
"""Run a 10% BERTopic sample.

By default, this script uses BERTopic's default embedding behavior. Optional
OpenAI embeddings and LLM topic labels can be enabled for a later pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEXT_DIR = DATA_DIR / "texts"
TOC_PATH = DATA_DIR / "toc.csv"
OUTPUT_DIR = ROOT / "outputs" / "bertopic_sample"

CUSTOM_STOPWORDS = {
    "chapter",
    "chap",
    "contents",
    "copyright",
    "edition",
    "electronic",
    "footnote",
    "image",
    "page",
    "preface",
    "reader",
    "said",
    "shall",
    "sic",
    "sir",
    "thy",
    "unto",
    "vol",
    "would",
    "ye",
}

CURATED_TOPIC_LABELS = {
    # These keep closely related violence/control topics distinct in the
    # teaching sample, where the local labeler otherwise tends to reuse the
    # same broad label for several different clusters.
    0: (
        "Whipping and Plantation Punishment",
        "Accounts of whipping, flogging, restraint, and plantation punishment described in testimony about slavery.",
    ),
    12: (
        "Violence by Mistresses and Women Enslavers",
        "Accounts of punishment and abuse associated with mistresses, madams, women, children, and household authority.",
    ),
    14: (
        "Farm Authority and Slaveholder Control",
        "Passes, farm authority, owners, servants, and local control in plantation and farm settings.",
    ),
    41: (
        "Overseer Violence in Field Labor",
        "Overseer violence, field labor, blood, flogging, and medical treatment after punishment.",
    ),
}

IRREGULAR_LEMMAS = {
    "babies": "baby",
    "children": "child",
    "feet": "foot",
    "horses": "horse",
    "jared": "jared",
    "men": "man",
    "named": "name",
    "negroes": "negro",
    "people": "people",
    "teeth": "tooth",
    "tied": "tie",
    "women": "woman",
}

DOUBLE_CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=0.10)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-words", type=int, default=750)
    parser.add_argument("--overlap-words", type=int, default=75)
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=10,
        help="Minimum BERTopic topic size. BERTopic's package default is 10.",
    )
    parser.add_argument(
        "--clustering",
        choices=["default", "sensitive"],
        default="default",
        help="Use BERTopic defaults or a more sensitive UMAP/HDBSCAN setup.",
    )
    parser.add_argument(
        "--cleaning",
        choices=["light", "enhanced"],
        default="enhanced",
        help="Use enhanced cleaning and stopwords for more meaningful topic terms.",
    )
    parser.add_argument(
        "--no-lemmatize",
        action="store_true",
        help="Do not collapse word variants for BERTopic's topic-word representation.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["bertopic-default", "openai", "ollama"],
        default="bertopic-default",
        help="Use BERTopic defaults, OpenAI embeddings, or local Ollama embeddings.",
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--ollama-embedding-model", default="llama3.1:latest")
    parser.add_argument("--label-model", default="gpt-5.4-mini")
    parser.add_argument("--ollama-model", default="llama3.1:latest")
    parser.add_argument(
        "--representation-backend",
        choices=["ctfidf", "keybert"],
        default="keybert",
        help="Improve topic words locally with KeyBERTInspired, or use BERTopic's c-TF-IDF.",
    )
    parser.add_argument(
        "--vectorizer-min-df",
        type=int,
        default=1,
        help="Minimum document frequency for topic representation terms.",
    )
    parser.add_argument(
        "--label-backend",
        choices=["none", "openai", "ollama"],
        default="ollama",
        help="Use local Ollama labels by default; use none to skip generated labels.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    return parser.parse_args()


def normalize_doc_id(filename: str) -> str:
    return Path(filename).with_suffix("").name


def txt_name_from_xml(filename: str) -> str:
    return f"{normalize_doc_id(filename)}.txt"


def clean_year(value: object) -> int | None:
    match = re.search(r"(17|18|19)\d{2}", str(value))
    return int(match.group(0)) if match else None


def load_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(TOC_PATH)
    metadata["doc_id"] = metadata["Filename"].map(normalize_doc_id)
    metadata["text_filename"] = metadata["Filename"].map(txt_name_from_xml)
    metadata["text_path"] = metadata["text_filename"].map(lambda name: TEXT_DIR / name)
    metadata["publication_year"] = metadata["Date"].map(clean_year)
    metadata = metadata[metadata["text_path"].map(Path.exists)].copy()
    return metadata


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\[[^\]]*(?:Image|Illustration|Cover|Title Page)[^\]]*\]", " ", text, flags=re.I)
    text = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def enhanced_clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[_*~=|<>]+", " ", text)
    text = re.sub(r"\bCHAPTER\s+[IVXLCDM\d]+\.?", " ", text, flags=re.I)
    text = re.sub(r"\b(?:ADVERTISEMENT|PREFACE|CONTENTS|INTRODUCTION)\b\.?", " ", text, flags=re.I)
    text = re.sub(r"[^A-Za-z' -]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize_token(token: str) -> str:
    token = token.lower().strip("'")
    if len(token) <= 3:
        return token
    if token in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[token]
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("oes") and len(token) > 4:
        return token[:-2]
    if token.endswith("ied") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] in DOUBLE_CONSONANTS:
            stem = stem[:-1]
        return stem
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] in DOUBLE_CONSONANTS:
            stem = stem[:-1]
        elif stem.endswith("v"):
            stem = f"{stem}e"
        return stem
    if token.endswith("es") and len(token) > 4 and token[-3] in {"s", "x", "z"}:
        return token[:-2]
    if token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def lemmatize_text(text: str) -> str:
    return " ".join(lemmatize_token(token) for token in text.split())


def chunk_words(words: list[str], chunk_size: int, overlap: int) -> Iterable[tuple[int, str]]:
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("--overlap-words must be smaller than --chunk-words")
    for chunk_index, start in enumerate(range(0, len(words), step)):
        chunk = words[start : start + chunk_size]
        if len(chunk) < max(100, chunk_size // 4):
            continue
        yield chunk_index, " ".join(chunk)


def build_chunks(
    metadata: pd.DataFrame,
    chunk_size: int,
    overlap: int,
    cleaning: str,
    lemmatize: bool,
) -> pd.DataFrame:
    rows = []
    for row in metadata.itertuples(index=False):
        text = read_text(Path(row.text_path))
        if cleaning == "enhanced":
            text = enhanced_clean_text(text)
        words = text.split()
        for chunk_index, chunk in chunk_words(words, chunk_size, overlap):
            model_text = lemmatize_text(chunk) if lemmatize else chunk.lower()
            rows.append(
                {
                    "chunk_id": f"{row.doc_id}__{chunk_index:04d}",
                    "doc_id": row.doc_id,
                    "chunk_index": chunk_index,
                    "author": row.Author,
                    "title": row.Title,
                    "date_raw": row.Date,
                    "publication_year": row.publication_year,
                    "url": row.URL,
                    "text": chunk,
                    "model_text": model_text,
                }
            )
    return pd.DataFrame(rows)


def require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it, then rerun this script to "
            "generate LLM embeddings and topic labels."
        )


def embed_texts(texts: list[str], model: str, batch_size: int = 64) -> np.ndarray:
    require_openai_key()
    import openai

    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = openai.Embedding.create(model=model, input=batch)
        ordered = sorted(response["data"], key=lambda item: item["index"])
        all_embeddings.extend(item["embedding"] for item in ordered)
        print(f"Embedded {min(start + batch_size, len(texts))}/{len(texts)} chunks")
    return np.array(all_embeddings, dtype=np.float32)


def embed_texts_ollama(texts: list[str], model: str) -> np.ndarray:
    all_embeddings = []
    for index, text in enumerate(texts, start=1):
        response = requests.post(
            "http://127.0.0.1:11434/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=180,
        )
        response.raise_for_status()
        all_embeddings.append(response.json()["embedding"])
        if index % 25 == 0 or index == len(texts):
            print(f"Ollama embedded {index}/{len(texts)} chunks")
    return np.array(all_embeddings, dtype=np.float32)


def topic_prompt(topic_id: int, words: list[str], examples: list[str]) -> str:
    example_text = "\n\n".join(f"Example {idx + 1}: {ex}" for idx, ex in enumerate(examples))
    return textwrap.dedent(
        f"""
        You are helping label topics from North American slave narratives.
        Use historically careful, specific wording.
        Write a human-readable theme title, not a comma-separated list of top words.
        Collapse word variants into one idea. For example, "whip, whipped,
        whipping, overseer" should become "Whipping and Plantation Punishment."

        Topic id: {topic_id}
        Top words: {", ".join(words)}

        Representative passages:
        {example_text}

        Return compact JSON with keys: label, description.
        """
    ).strip()


def call_chat_model(prompt: str, model: str) -> dict[str, str]:
    require_openai_key()
    import openai

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    content = response["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"label": content.strip().splitlines()[0][:80], "description": content.strip()}
    return {
        "label": str(parsed.get("label", "")).strip(),
        "description": str(parsed.get("description", "")).strip(),
    }


def call_ollama_model(prompt: str, model: str) -> dict[str, str]:
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        },
        timeout=180,
    )
    response.raise_for_status()
    content = response.json()["response"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"label": content.strip().splitlines()[0][:80], "description": content.strip()}
    return normalize_label_result(parsed, content)


def normalize_label_result(parsed: dict[str, object], fallback_text: str = "") -> dict[str, str]:
    label = str(parsed.get("label", "")).strip()
    description = str(parsed.get("description", "")).strip()
    if not label:
        label = fallback_text.strip().splitlines()[0][:80] if fallback_text.strip() else "Unlabeled topic"
    if not description:
        description = label
    return {"label": label, "description": description}


def label_topics(
    topic_model,
    chunks: pd.DataFrame,
    topics: list[int],
    label_backend: str,
    label_model: str,
    ollama_model: str,
) -> pd.DataFrame:
    labels = []
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue
        topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
        representatives = chunks.loc[chunks["topic"] == topic_id, "text"].head(3)
        examples = [text[:1200] for text in representatives]
        prompt = topic_prompt(topic_id, topic_words, examples)
        if label_backend == "ollama":
            result = call_ollama_model(prompt, ollama_model)
        elif label_backend == "openai":
            result = call_chat_model(prompt, label_model)
        else:
            raise ValueError(f"Unsupported label backend: {label_backend}")
        if not result["label"] or result["label"] == "Unlabeled topic" or result["label"].startswith("{"):
            result["label"] = ", ".join(topic_words[:4]).title()
        if not result["description"] or result["description"] == "Unlabeled topic":
            result["description"] = f"Topic characterized by: {', '.join(topic_words[:8])}."
        if topic_id in CURATED_TOPIC_LABELS:
            result["label"], result["description"] = CURATED_TOPIC_LABELS[topic_id]
        labels.append({"topic": topic_id, **result, "top_words": ", ".join(topic_words)})
        print(f"Labeled topic {topic_id}: {result['label']}")
    return pd.DataFrame(labels)


def save_visualizations(topic_model, chunks: pd.DataFrame, embeddings: np.ndarray) -> None:
    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    plotters = {
        "topics.html": lambda: topic_model.visualize_topics(),
        "topic_barchart.html": lambda: topic_model.visualize_barchart(),
        "topic_hierarchy.html": lambda: topic_model.visualize_hierarchy(),
        "documents.html": lambda: topic_model.visualize_documents(chunks["text"].tolist(), embeddings=embeddings),
    }
    for filename, plotter in plotters.items():
        try:
            plotter().write_html(viz_dir / filename)
        except Exception as exc:
            print(f"Skipped {filename}: {exc}")

    time_data = chunks.dropna(subset=["publication_year"]).copy()
    time_data["decade"] = (time_data["publication_year"].astype(int) // 10) * 10
    time_counts = (
        time_data[time_data["topic"] != -1]
        .groupby(["decade", "topic"])
        .size()
        .reset_index(name="chunks")
    )
    if not time_counts.empty:
        fig = px.area(
            time_counts,
            x="decade",
            y="chunks",
            color=time_counts["topic"].astype(str),
            labels={"color": "Topic", "chunks": "Chunks", "decade": "Publication decade"},
            title="Topic Prevalence by Publication Decade",
        )
        fig.write_html(viz_dir / "topics_over_time.html")


def summarize_topics_by_document(chunks: pd.DataFrame, topic_info: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic, group in chunks.groupby("topic"):
        doc_counts = group.groupby(["doc_id", "title", "author"]).size().sort_values(ascending=False).head(5)
        years = group["publication_year"].dropna()
        topic_row = topic_info[topic_info["Topic"] == topic].iloc[0].to_dict()
        rows.append(
            {
                "topic": topic,
                "count": len(group),
                "default_name": topic_row.get("Name", ""),
                "year_min": int(years.min()) if not years.empty else "",
                "year_max": int(years.max()) if not years.empty else "",
                "top_documents": " | ".join(
                    f"{idx[1]} ({idx[2]}): {count}" for idx, count in doc_counts.items()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("topic")


def main() -> None:
    global OUTPUT_DIR
    args = parse_args()
    OUTPUT_DIR = Path(args.output_dir)

    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()
    sample_size = max(1, round(len(metadata) * args.sample_frac))
    sample = metadata.sample(n=sample_size, random_state=args.seed).sort_values("doc_id")
    chunks = build_chunks(
        sample,
        args.chunk_words,
        args.overlap_words,
        args.cleaning,
        lemmatize=not args.no_lemmatize,
    )

    sample.to_csv(OUTPUT_DIR / "sample_documents.csv", index=False)
    chunks.to_csv(OUTPUT_DIR / "sample_chunks.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Sampled {len(sample)} documents into {len(chunks)} chunks")
    if args.prepare_only:
        print(f"Prepared sample files in {OUTPUT_DIR}")
        return

    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from hdbscan import HDBSCAN
    from umap import UMAP

    vectorizer_model = None
    if args.cleaning == "enhanced":
        stopwords = sorted(set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS)
        vectorizer_model = CountVectorizer(
            stop_words=stopwords,
            ngram_range=(1, 2),
            min_df=args.vectorizer_min_df,
            max_df=0.85,
        )

    representation_model = None
    if args.representation_backend == "keybert":
        representation_model = KeyBERTInspired()

    umap_model = None
    hdbscan_model = None
    if args.clustering == "sensitive":
        umap_model = UMAP(
            n_neighbors=8,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=args.seed,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=args.min_topic_size,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

    topic_model = BERTopic(
        min_topic_size=args.min_topic_size,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
    )
    embeddings = None
    if args.embedding_backend == "openai":
        embeddings_path = OUTPUT_DIR / "embeddings_openai.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            if len(embeddings) == len(chunks):
                print(f"Loaded cached OpenAI embeddings from {embeddings_path}")
            else:
                print(f"Ignoring stale OpenAI embeddings at {embeddings_path}")
                embeddings = None
        if embeddings is None:
            embeddings = embed_texts(chunks["text"].tolist(), args.embedding_model)
            np.save(embeddings_path, embeddings)
    elif args.embedding_backend == "ollama":
        safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.ollama_embedding_model)
        embeddings_path = OUTPUT_DIR / f"embeddings_ollama_{safe_model_name}.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            if len(embeddings) == len(chunks):
                print(f"Loaded cached Ollama embeddings from {embeddings_path}")
            else:
                print(f"Ignoring stale Ollama embeddings at {embeddings_path}")
                embeddings = None
        if embeddings is None:
            embeddings = embed_texts_ollama(chunks["text"].tolist(), args.ollama_embedding_model)
            np.save(embeddings_path, embeddings)

    topic_docs = chunks["model_text"].tolist()
    topics, probabilities = topic_model.fit_transform(topic_docs, embeddings=embeddings)
    chunks["topic"] = topics

    topic_info = topic_model.get_topic_info()
    chunks.to_csv(OUTPUT_DIR / "topic_assignments.csv", index=False)
    topic_info.to_csv(OUTPUT_DIR / "topic_info.csv", index=False)
    summarize_topics_by_document(chunks, topic_info).to_csv(
        OUTPUT_DIR / "topic_summary_by_document.csv", index=False
    )
    topic_model.save(str(OUTPUT_DIR / "model"))

    if args.label_backend != "none" and not args.skip_labels:
        labels = label_topics(
            topic_model,
            chunks,
            topics,
            args.label_backend,
            args.label_model,
            args.ollama_model,
        )
        labels.to_csv(OUTPUT_DIR / "topic_labels_llm.csv", index=False)

    save_visualizations(topic_model, chunks, embeddings)
    print(f"Done. Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
