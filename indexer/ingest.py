import argparse
import json
import os
import time
import requests

from typing import Any, Dict, List

import pandas as pd
import pysolr


DEFAULT_SOLR_BASE = os.environ.get("SOLR_BASE", "http://localhost:8983/solr")
DEFAULT_CORE = os.environ.get("SOLR_CORE", "mcu_reviews")


def _safe_str(v: Any) -> str:
    # Pandas uses NaN for missing values; Solr should receive empty strings instead.
    if v is None:
        return ""
    try:
        # NaN != NaN
        if isinstance(v, float) and v != v:
            return ""
    except Exception:
        pass
    return str(v)


def setup_schema(schema_ops_path: str, solr_base: str, core: str) -> None:
    """
    Apply field definitions from `schema_ops_path` using Solr Schema API.

    The schema_ops file is a JSON array, each item being:
      { "op": "add-field"|"add-copy-field", "payload": {...} }
    """
    with open(schema_ops_path, "r", encoding="utf-8") as f:
        ops: List[Dict[str, Any]] = json.load(f)

    schema_url = f"{solr_base}/{core}/schema"
    headers = {"Content-type": "application/json"}

    session = requests.Session()
    for i, item in enumerate(ops, start=1):
        op = item["op"]
        payload = item["payload"]
        body = {op: payload}

        r = session.post(schema_url, headers=headers, data=json.dumps(body))
        if r.status_code in (200, 400):
            # Adding an already-existing field often returns 400; that's OK for reruns.
            # We'll still print a short status for debugging.
            print(f"[schema] ({i}/{len(ops)}) {op} -> HTTP {r.status_code}")
        else:
            r.raise_for_status()

        # Small delay helps avoid hammering the schema endpoint in very fast loops.
        time.sleep(0.02)


def iter_docs_from_csv(csv_path: str, limit: int | None = None):
    df = pd.read_csv(csv_path)
    n = len(df) if limit is None else min(len(df), limit)

    for row in df.head(n).itertuples(index=False):
        # Column order in the CSV:
        # movie_id, movie_title, review_id, author, author_rating,
        # upvotes, downvotes, submission_date, summary, content
        (
            movie_id,
            movie_title,
            review_id,
            author,
            author_rating,
            upvotes,
            downvotes,
            submission_date,
            summary,
            content,
        ) = row

        upvotes_i = int(upvotes) if not pd.isna(upvotes) else 0
        downvotes_i = int(downvotes) if not pd.isna(downvotes) else 0
        total = upvotes_i + downvotes_i + 1
        helpfulness = upvotes_i / total

        submission_date_s = _safe_str(submission_date)
        # Original is YYYY-MM-DD, convert to Solr pdate expected ISO-8601.
        submission_iso = submission_date_s + "T00:00:00Z"
        year = int(submission_date_s[:4]) if len(submission_date_s) >= 4 else 0

        yield {
            "id": _safe_str(review_id),
            "movie_id": _safe_str(movie_id),
            "movie_title": _safe_str(movie_title),
            "author": _safe_str(author),
            "author_rating": float(author_rating) if not pd.isna(author_rating) else 0.0,
            "upvotes": upvotes_i,
            "downvotes": downvotes_i,
            "helpfulness_score": float(helpfulness),
            "submission_date": submission_iso,
            "year": int(year),
            "summary": _safe_str(summary),
            "content": _safe_str(content),
            # Reserved for classifier output; keep empty for now.
            "sentiment": "",
        }


def index_csv(
    csv_path: str,
    solr_base: str,
    core: str,
    batch_size: int = 1000,
    limit: int | None = None,
) -> None:
    solr = pysolr.Solr(f"{solr_base}/{core}", always_commit=False, timeout=60)

    docs_batch: List[Dict[str, Any]] = []
    for doc in iter_docs_from_csv(csv_path, limit=limit):
        docs_batch.append(doc)
        if len(docs_batch) >= batch_size:
            solr.add(docs_batch)
            docs_batch = []

    if docs_batch:
        solr.add(docs_batch)

    solr.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Index MCU IMDB reviews into Solr.")
    parser.add_argument("--csv", default="../data/mcu_imdb_reviews.csv", help="Input CSV path.")
    parser.add_argument("--setup-schema", action="store_true", help="Apply schema_fields.json to Solr.")
    parser.add_argument(
        "--schema-ops",
        default="schema/schema_fields.json",
        help="Path to schema_ops JSON (relative to indexer/).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Index only first N rows (for testing).")
    parser.add_argument("--batch-size", type=int, default=1000, help="Docs per batch.")

    args = parser.parse_args()

    solr_base = DEFAULT_SOLR_BASE
    core = DEFAULT_CORE

    if args.setup_schema:
        schema_ops_path = os.path.join(os.path.dirname(__file__), args.schema_ops)
        setup_schema(schema_ops_path=schema_ops_path, solr_base=solr_base, core=core)

    csv_path = os.path.join(os.path.dirname(__file__), args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    index_csv(
        csv_path=csv_path,
        solr_base=solr_base,
        core=core,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

