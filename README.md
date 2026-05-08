# Topic Modeling Slave Narratives

An instructional site for teaching BERTopic topic modeling using the North American Slave Narratives collection from [Documenting the American South](https://docsouth.unc.edu/neh/) (University of North Carolina). Designed for history students with no prior coding experience.

**Live site:** https://jinghanlib.github.io/topic-modeling-slave-narratives/

---

## What This Is

The site walks students through a complete topic modeling workflow applied to a 10% sample of the slave narrative collection. It covers:

- How topic modeling works and how it differs from word counting
- How preprocessing decisions (cleaning, chunking, tokenization, lemmatization) shape results
- How a local BERTopic pipeline runs — embeddings, dimensionality reduction, clustering, labeling
- How to read and interpret topic labels, CSV outputs, and nine interactive visualizations
- Where human judgment enters the process at every stage

The site has two pages: a **Simple Explanation** for reading and understanding the workflow, and a **Hands-On Exercise** for students who want to run the pipeline themselves.

---

## Repository Structure

```
.
├── _quarto.yml                      # Quarto site configuration
├── index.qmd                        # Landing page
├── instructions/
│   ├── 01_simple_explanation.qmd   # Plain-language guide to the full workflow
│   ├── 02_hands_on_exercise.qmd    # Step-by-step student exercise
│   └── assets/
│       └── site.css                # UC Riverside branded styles
├── scripts/
│   ├── run_bertopic_sample.py      # BERTopic pipeline (cleaning, chunking, embedding, clustering, labeling)
│   └── visualize_topic_metadata.py # Generates topic_review_table.csv and metadata visualizations
├── outputs/
│   └── bertopic_sample_nomic_sensitive_lemmatized/   # Reference run (do not overwrite)
│       ├── topic_review_table.csv         # Best starting point: labels, top words, years, documents
│       ├── topic_labels_llm.csv           # LLM-generated topic labels and descriptions
│       ├── topic_assignments.csv          # Which chunk was assigned to which topic
│       ├── topic_info.csv                 # BERTopic internal topic statistics
│       ├── topic_summary_by_document.csv  # Topic distribution aggregated by source document
│       ├── sample_documents.csv           # Metadata for the 29 sampled documents
│       ├── metadata_visualizations/       # Publication-year and document-based charts
│       │   ├── topic_prevalence_grouped_bars.html   # Topic shares by decade (grouped bars)
│       │   ├── topic_prevalence_by_decade.html      # Topic trends by decade (faceted lines)
│       │   ├── topic_decade_heatmap.html             # Topics × decades heatmap
│       │   ├── document_topic_heatmap.html           # Documents × topics heatmap
│       │   └── sample_documents_timeline.html        # Publication year scatter of sampled documents
│       └── visualizations/                # BERTopic built-in charts
│           ├── topics.html               # 2D topic cluster map
│           ├── topic_barchart.html       # Top words per topic
│           ├── topic_hierarchy.html      # Hierarchical topic clustering
│           └── topics_over_time.html     # Topic prevalence as stacked area chart
├── docs/                            # Rendered Quarto site served by GitHub Pages
├── requirements.txt
└── README.md
```

### What is and is not in the repository

**Included:** all source `.qmd` files, scripts, reference CSV outputs, all 9 visualization HTML files, and the rendered `docs/` site.

**Excluded from the repository:**
- `data/` — the corpus must be downloaded from Documenting the American South (see below)
- `outputs/.../embeddings_ollama_nomic-embed-text.npy` — embedding cache, regenerated automatically on re-run
- `outputs/.../model/` — serialized BERTopic model, not needed for the site
- `outputs/.../sample_chunks.csv` — large intermediate file, generated when the pipeline runs
- `outputs/.../visualizations/documents.html` — ~11 MB due to embedded raw embedding vectors
- `outputs/my_run/` — students write their own outputs here; not tracked

---

## Data

The corpus is not included in this repository. Download it from the source:

> Documenting the American South — North American Slave Narratives
> https://docsouth.unc.edu/neh/

After downloading, place the `data/` folder inside the project root. The scripts expect:

```
data/
  texts/       # 294 plain text narrative files
  xml/         # 294 XML versions
  toc.csv      # Metadata: author, title, publication year, URL
  readme.txt
```

---

## Running the Pipeline

Students write their results to `outputs/my_run/` to keep them separate from the reference outputs in `outputs/bertopic_sample_nomic_sensitive_lemmatized/`.

### Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) with two local models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows (Command Prompt)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the 10% sample

**Mac/Linux:**
```bash
python -u scripts/run_bertopic_sample.py \
  --output-dir outputs/my_run \
  --embedding-backend ollama \
  --ollama-embedding-model nomic-embed-text \
  --representation-backend ctfidf \
  --clustering sensitive \
  --label-backend ollama \
  --ollama-model llama3.1:latest
```

**Windows (Command Prompt):**
```
python -u scripts/run_bertopic_sample.py ^
  --output-dir outputs/my_run ^
  --embedding-backend ollama ^
  --ollama-embedding-model nomic-embed-text ^
  --representation-backend ctfidf ^
  --clustering sensitive ^
  --label-backend ollama ^
  --ollama-model llama3.1:latest
```

### Generate the review table

**Mac/Linux:**
```bash
python scripts/visualize_topic_metadata.py \
  --output-dir outputs/my_run \
  --top-n 15
```

**Windows (Command Prompt):**
```
python scripts/visualize_topic_metadata.py ^
  --output-dir outputs/my_run ^
  --top-n 15
```

---

## Rebuilding the Site

The site is built with [Quarto](https://quarto.org/). To rebuild after editing `.qmd` files:

```bash
quarto render
```

Output goes to `docs/`. Push to `main` to update the live site.

---

## Citation

Texts are from the North American Slave Narratives collection, Documenting the American South, University of North Carolina at Chapel Hill. Follow the collection's reuse guidance when citing or redistributing the source texts.
