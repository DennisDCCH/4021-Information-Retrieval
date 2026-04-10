# 4021 Information Retrieval — MCU IMDB Reviews

End-to-end pipeline for crawling MCU movie reviews, optional classification (sentiment / sarcasm), and **Solr-based indexing and search** with a small web UI.

## Repository layout

| Path | Purpose |
|------|---------|
| [`crawling/`](crawling/) | IMDB scraping notebooks and CSV datasets (`mcu_imdb_reviews.csv`, train/test splits, manual labels) |
| [`classification/`](classification/) | Sentiment analysis, sarcasm detection, NER / preprocessing experiments |
| [`indexer/`](indexer/) | Docker Solr, schema, `ingest.py`, query demos, innovations, and web UI server/files — see [`indexer/README.md`](indexer/README.md) |

## Quick start (index + search)

1. **Solr (Docker)** — from repo root:

   ```bash
   cd indexer
   docker compose up -d
   ```

2. **Python deps** (3.11.x recommended):

   ```bash
   pip install pysolr pandas requests
   ```

3. **Schema + index** — still under `indexer/`:

   ```bash
   python ingest.py --setup-schema
   python ingest.py --csv ../crawling/data/mcu_imdb_reviews.csv
   ```

4. **Optional: web UI** — from repo root:

   ```bash
   cd indexer
   python scripts/serve_solr.py
   ```

   Open `http://127.0.0.1:8000` (proxies to `http://127.0.0.1:8983/solr/mcu_reviews`).

5. **Question 3 experiments** (innovations + JSON report):

   ```bash
   cd indexer
   python innovations/run_experiments.py
   ```

   Writes [`indexer/innovations/experiment_results.json`](indexer/innovations/experiment_results.json) and prints tables to stdout.

For Solr admin UI: `http://localhost:8983/solr/#/mcu_reviews`.

## Classification: Sentiment Analysis

The sentiment analysis evaluation script is located at [classification/sentiment_analysis/evaluate_sentiment_models.py](classification/sentiment_analysis/evaluate_sentiment_models.py#L1). It evaluates the classical models and transformer model, then writes the comparison outputs into [classification/sentiment_analysis/results/](classification/sentiment_analysis/results/).

1. Install the required Python libraries:

   ```bash
   pip install pandas matplotlib torch scikit-learn scipy transformers spacy openpyxl
   python -m spacy download en_core_web_sm
   ```

2. Ensure the preprocessed dataset path inside the script is correct before running it.

   The script currently expects the CSV path defined by `PREPROCESSED_PATH` in [evaluate_sentiment_models.py](classification/sentiment_analysis/evaluate_sentiment_models.py#L25). 
   
   Since the repository is submitted as a zip, extract the folder and update the constant to point to the extracted preprocessed dataset.

3. Run the evaluation:

   ```bash
   python classification/sentiment_analysis/evaluate_sentiment_models.py
   ```

   This generates the comparison tables, summary CSVs, the test dataset results, and the random-test predictions/summary files in the sentiment analysis results folder.
