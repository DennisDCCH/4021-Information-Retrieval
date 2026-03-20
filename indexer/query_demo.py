import argparse
import time
from typing import Any, Dict

import requests


DEFAULT_SOLR_BASE = "http://localhost:8983/solr"
DEFAULT_CORE = "mcu_reviews"


def run_query(session: requests.Session, solr_select_url: str, label: str, params: Dict[str, Any]) -> None:
    start = time.perf_counter()
    resp = session.get(solr_select_url, params=params, timeout=60)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    resp.raise_for_status()
    data = resp.json()

    num_found = data.get("response", {}).get("numFound", 0)
    docs = data.get("response", {}).get("docs", [])[:5]

    print(f"\n=== {label} ===")
    print(f"elapsed_ms={elapsed_ms:.2f} numFound={num_found}")

    for d in docs:
        movie_title = d.get("movie_title", "")
        review_id = d.get("id", d.get("review_id", ""))
        summary = d.get("summary", "")
        hs = d.get("helpfulness_score", "")
        submission_date = d.get("submission_date", "")
        author_rating = d.get("author_rating", "")
        print(f"- id={review_id} movie_title={movie_title} helpfulness={hs} summary={summary[:80]} author_rating={author_rating} submission_date={submission_date}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5 sample Solr queries with timing.")
    parser.add_argument("--solr-base", default=DEFAULT_SOLR_BASE, help="Base URL, e.g. http://localhost:8983/solr")
    parser.add_argument("--core", default=DEFAULT_CORE, help="Solr core name")
    parser.add_argument("--rows", type=int, default=5, help="Rows to fetch per query")
    args = parser.parse_args()

    solr_select_url = f"{args.solr_base}/{args.core}/select"
    session = requests.Session()

    # Common params: keep payload small and ensure we always get fields we print.
    common_fl = "id,movie_title,summary,helpfulness_score,submission_date"

    queries = [
        (
            "Q1-basic-keyword",
            {
                "q": 'content:"infinity war"',
                "rows": args.rows,
                "wt": "json",
                "fl": common_fl,
            },
        ),
        (
            "Q2-movie-filtered",
            {
                "q": 'content:villain',
                "fq": 'movie_title:"Thor: Ragnarok"',
                "rows": args.rows,
                "wt": "json",
                "fl": common_fl,
            },
        ),
        (
            "Q3-rating-filtered",
            {
                "q": 'content:disappointing',
                "fq": "author_rating:[1 TO 4]",
                "rows": args.rows,
                "wt": "json",
                "fl": common_fl,
            },
        ),
        (
            "Q4-timeline",
            {
                "q": "content:performance",
                "fq": 'submission_date:[2019-01-01T00:00:00Z TO 2021-12-31T00:00:00Z]',
                "rows": args.rows,
                "wt": "json",
                "fl": common_fl,
            },
        ),
        (
            "Q5-helpfulness-ranked",
            {
                "q": "content:masterpiece",
                "sort": "helpfulness_score desc",
                "rows": args.rows,
                "wt": "json",
                "fl": common_fl,
            },
        ),
    ]

    print("Running 5 demo queries for timing...")
    for label, params in queries:
        run_query(session, solr_select_url, label, params)


if __name__ == "__main__":
    main()

