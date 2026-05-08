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

## For Students: Getting This Repository

### Step 1 — Open a terminal

You will type commands into a terminal window. Here is how to open one:

- **Mac:** Press `Command + Space`, type `Terminal`, and press Enter.
- **Windows:** Press `Win + R`, type `cmd`, and press Enter. Or search for **PowerShell** in the Start menu.

All commands in this README are typed into that terminal window. Press Enter after each one to run it.

### Step 2 — Download this repository

If you have Git installed, run this in the terminal:

```bash
git clone https://github.com/jinghanlib/topic-modeling-slave-narratives.git
```

If you do not have Git, go to the repository page on GitHub, click the green **Code** button, choose **Download ZIP**, then unzip the downloaded file.

### Step 3 — Open a terminal inside the project folder

The easiest way to do this is to right-click the project folder itself:

- **Mac:** Right-click the folder in Finder and choose **New Terminal at Folder**. If you do not see that option, go to System Settings → Privacy & Security → Developer Tools and enable Terminal.
- **Windows:** Hold **Shift** and right-click the folder in File Explorer, then choose **Open PowerShell window here** or **Open Command window here**.

This opens a terminal already pointed at the right location. You do not need to type any folder path.

If you close and reopen the terminal later, right-click the folder again to reopen it in the correct place before running any commands.

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
│       └── site.css                # Site styles
├── scripts/
│   ├── run_bertopic_sample.py      # BERTopic pipeline (cleaning, chunking, embedding, clustering, labeling)
│   └── visualize_topic_metadata.py # Generates topic_review_table.csv and metadata visualizations
├── outputs/
│   └── bertopic_sample_nomic_sensitive_lemmatized/   # Reference run — do not overwrite
│       ├── topic_review_table.csv         # Best starting point: labels, top words, years, documents
│       ├── topic_labels_llm.csv           # LLM-generated topic labels and descriptions
│       ├── topic_assignments.csv          # Which chunk was assigned to which topic
│       ├── topic_info.csv                 # BERTopic internal topic statistics
│       ├── topic_summary_by_document.csv  # Topic distribution aggregated by source document
│       ├── sample_documents.csv           # Metadata for the 29 sampled documents
│       ├── metadata_visualizations/       # Publication-year and document-based charts
│       │   ├── topic_prevalence_grouped_bars.html   # Topic shares by decade (grouped bars)
│       │   ├── topic_prevalence_by_decade.html      # Topic trends by decade (faceted lines)
│       │   ├── topic_decade_heatmap.html            # Topics × decades heatmap
│       │   ├── document_topic_heatmap.html          # Documents × topics heatmap
│       │   └── sample_documents_timeline.html       # Publication year scatter of sampled documents
│       └── visualizations/                # BERTopic built-in charts
│           ├── topics.html               # 2D topic cluster map
│           ├── topic_barchart.html       # Top words per topic
│           ├── topic_hierarchy.html      # Hierarchical topic clustering
│           └── topics_over_time.html     # Topic prevalence as stacked area chart
├── docs/                            # Rendered site served by GitHub Pages
├── requirements.txt
└── README.md
```

When you run the pipeline yourself, your results will be written to `outputs/my_run/`, keeping them separate from the reference outputs above.

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

Full step-by-step instructions, including installing Python, Ollama, and the required packages, are on the [Hands-On Exercise](https://jinghanlib.github.io/topic-modeling-slave-narratives/instructions/02_hands_on_exercise.html) page.

The commands below assume you have completed that setup and your terminal is inside the project folder.

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
