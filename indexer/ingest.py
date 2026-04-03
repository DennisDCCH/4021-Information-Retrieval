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
    if v is None:
        return ""
    try:
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
    fields_resp = session.get(f"{schema_url}/fields")
    fields_resp.raise_for_status()
    existing_fields = {field["name"] for field in fields_resp.json().get("fields", [])}

    copy_resp = session.get(f"{schema_url}/copyfields")
    copy_resp.raise_for_status()
    copy_counts: Dict[tuple[str, str], int] = {}
    for item in copy_resp.json().get("copyFields", []):
        key = (item["source"], item["dest"])
        copy_counts[key] = copy_counts.get(key, 0) + 1

    for i, item in enumerate(ops, start=1):
        op = item["op"]
        payload = item["payload"]
        if op == "add-field":
            name = payload["name"]
            if name in existing_fields:
                continue
            body = {op: payload}
            r = session.post(schema_url, headers=headers, data=json.dumps(body))
            if r.status_code == 200:
                existing_fields.add(name)
            elif r.status_code != 400:
                r.raise_for_status()
            time.sleep(0.02)
            continue

        if op == "add-copy-field":
            key = (payload["source"], payload["dest"])
            duplicate_count = copy_counts.get(key, 0)

            while duplicate_count > 1:
                delete_body = {
                    "delete-copy-field": {
                        "source": payload["source"],
                        "dest": payload["dest"],
                    }
                }
                r = session.post(schema_url, headers=headers, data=json.dumps(delete_body))
                if r.status_code not in (200, 400):
                    r.raise_for_status()
                duplicate_count -= 1
                copy_counts[key] = duplicate_count
                time.sleep(0.02)

            if duplicate_count == 1:
                continue

            body = {op: payload}
            r = session.post(schema_url, headers=headers, data=json.dumps(body))
            if r.status_code == 200:
                copy_counts[key] = 1
            elif r.status_code != 400:
                r.raise_for_status()
            time.sleep(0.02)
            continue

        body = {op: payload}

        r = session.post(schema_url, headers=headers, data=json.dumps(body))
        if r.status_code not in (200, 400):
            r.raise_for_status()

        time.sleep(0.02)


def iter_docs_from_csv(csv_path: str, limit: int | None = None):
    df = pd.read_csv(csv_path)
    n = len(df) if limit is None else min(len(df), limit)

    for row in df.head(n).itertuples(index=False):
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
    parser.add_argument("--csv", default=None, help="Input CSV path.")
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

    if not args.csv:
        return

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
