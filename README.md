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
