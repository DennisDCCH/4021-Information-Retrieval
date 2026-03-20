# Indexer (Solr) - MCU IMDB Reviews

This folder covers the **Indexing and ranking** part of the assignment: build a Solr index from `data/mcu_imdb_reviews.csv` and run reproducible query demos.

## 1. Start Solr (Docker)

From the repo root:

```bash
cd indexer
docker compose up -d
```

Solr creates the core **`mcu_reviews`** by default.

Check the admin UI (open in a browser):

- http://localhost:8983/solr/#/mcu_reviews

Stop (data kept in the volume):

```bash
docker compose down
```

Tear down and remove volumes:

```bash
docker compose down -v
```

## 2. Install Python dependencies

```bash
pip install pysolr pandas requests
```

## 3. Apply schema (safe to re-run)

```bash
python ingest.py --setup-schema
```

## 4. Index all data (90k+ reviews)

```bash
python ingest.py --csv ../data/mcu_imdb_reviews.csv
```

To smoke-test with the first 500 rows:

```bash
python ingest.py --csv ../data/mcu_imdb_reviews.csv --limit 500
```

## 5. Question 2: five sample queries (with timing)

```bash
python query_demo.py
```
