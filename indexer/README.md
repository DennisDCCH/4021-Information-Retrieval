# Indexer (Solr) - MCU IMDB Reviews

This folder covers the **Indexing and ranking** part of the assignment: build a Solr index from `[crawling/data/mcu_imdb_reviews.csv](../crawling/data/mcu_imdb_reviews.csv)` and run reproducible query demos.

## 1. Start Solr (Docker)

From the repo root:

```bash
cd indexer
docker compose up -d
```

Solr creates the core `**mcu_reviews**` by default.

Check the admin UI (open in a browser):

- [http://localhost:8983/solr/#/mcu_reviews](http://localhost:8983/solr/#/mcu_reviews)

Stop (data kept in the volume):

```bash
docker compose down
```

Tear down and remove volumes:

```bash
docker compose down -v
```

## 2. Install Python dependencies

Use **Python 3.11.15** (`python --version`). Other **3.11.x** releases should work as well.

```bash
pip install pysolr pandas requests
```

## 3. Apply schema (safe to re-run)

```bash
python ingest.py --setup-schema
```

## 4. Index all data (90k+ reviews)

```bash
python ingest.py --csv ../crawling/data/mcu_imdb_reviews.csv
```

To smoke-test with the first 500 rows:

```bash
python ingest.py --csv ../crawling/data/mcu_imdb_reviews.csv --limit 500
```

## 5. Question 2: five sample queries (with timing)

```bash
python query_demo.py
```

## 6. Question 3: reproducible experiments (innovations)

After Solr is up and data is indexed, run the bundled experiment runner (from `indexer/`):

```bash
python innovations/run_experiments.py > results.txt
```

- Prints before/after tables to results.txt.
- Writes `[innovations/experiment_results.json](innovations/experiment_results.json)`

Optional: run individual demos in `[innovations/](innovations/)` (`faceted_search.py`, `helpfulness_rank.py`, `timeline_search.py`).