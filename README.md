# Topic Modeling Slave Narratives

An instructional site for teaching BERTopic topic modeling using the North American Slave Narratives collection from [Documenting the American South](https://docsouth.unc.edu/neh/) (University of North Carolina). Designed for history students with no prior coding experience.

**Live site:** https://jinghanlib.github.io/topic-modeling-slave-narratives/

---

## What This Is

The site walks students through a complete topic modeling workflow applied to a 10% sample of the slave narrative collection. It covers:

- How topic modeling works and how it differs from word counting
- How preprocessing decisions (cleaning, chunking, lemmatization) shape results
- How a local BERTopic pipeline runs — embeddings, dimensionality reduction, clustering, labeling
- How to read and interpret topic labels, CSV outputs, and visualizations
- Where human judgment enters the process

The site includes a **Simple Explanation** page for reading and a **Hands-On Exercise** page for students who want to run the pipeline themselves.

---

## Repository Structure

```
.
├── index.qmd                        # Landing page
├── instructions/
│   ├── 01_simple_explanation.qmd   # Plain-language guide
│   └── 02_hands_on_exercise.qmd   # Step-by-step student exercise
├── scripts/
│   ├── run_bertopic_sample.py      # Main BERTopic pipeline
│   └── visualize_topic_metadata.py # Generates topic_review_table.csv and charts
├── outputs/
│   └── bertopic_sample_nomic_sensitive_lemmatized/
│       ├── *.csv                   # Reference outputs (do not overwrite)
│       └── metadata_visualizations/ # Charts embedded in the site
├── docs/                           # Rendered Quarto site (served by GitHub Pages)
├── requirements.txt
└── _quarto.yml
```

The `outputs/bertopic_sample_nomic_sensitive_lemmatized/` folder contains pre-generated reference results. Students who run the exercise write their own results to `outputs/my_run/`.

---

## Data

The corpus is not included in this repository. Download it from the source:

> Documenting the American South — North American Slave Narratives
> https://docsouth.unc.edu/neh/

After downloading, place the `data/` folder inside the project root so the scripts can find it.

---

## Running the Pipeline

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
# .venv\Scripts\activate         # Windows

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

Then generate the review table:

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

The site is built with [Quarto](https://quarto.org/). To rebuild after editing the `.qmd` files:

```bash
quarto render
```

Output goes to `docs/`. Push to `main` to update the live site.

---

## Citation

Texts are from the North American Slave Narratives collection, Documenting the American South, University of North Carolina at Chapel Hill. Follow the collection's reuse guidance when citing or redistributing the source texts.
