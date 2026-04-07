import argparse
from typing import Any, Dict

import requests


DEFAULT_SOLR_BASE = "http://localhost:8983/solr"
DEFAULT_CORE = "mcu_reviews"


def run_facet_query(session: requests.Session, solr_select_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = session.get(solr_select_url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def facets_by_movie_title(session: requests.Session, solr_select_url: str, keyword: str, rows: int = 0) -> Dict[str, Any]:
    # rows=0 keeps payload small; we only care about facet_counts.
    params = {
        "q": f'content:"{keyword}"',
        "wt": "json",
        "rows": rows,
        "facet": "true",
        "facet.field": "movie_title",
        "facet.limit": 20,
    }
    return run_facet_query(session, solr_select_url, params)


def facets_by_author_rating_ranges(
    session: requests.Session,
    solr_select_url: str,
    keyword: str,
    start: float = 1,
    end: float = 10,
    gap: float = 3,
    rows: int = 0,
) -> Dict[str, Any]:
    params = {
        "q": f'content:"{keyword}"',
        "wt": "json",
        "rows": rows,
        "facet": "true",
        "facet.range": "author_rating",
        "facet.range.start": start,
        "facet.range.end": end,
        "facet.range.gap": gap,
        # Some Solr versions want the numeric field name to be repeated:
        # "f.author_rating.facet.mincount": 1,
    }
    return run_facet_query(session, solr_select_url, params)


def facets_by_year(session: requests.Session, solr_select_url: str, keyword: str, rows: int = 0) -> Dict[str, Any]:
    params = {
        "q": f'content:"{keyword}"',
        "wt": "json",
        "rows": rows,
        "facet": "true",
        "facet.field": "year",
        "facet.limit": 50,
    }
    return run_facet_query(session, solr_select_url, params)


def facets_by_sentiment(session: requests.Session, solr_select_url: str, keyword: str, rows: int = 0) -> Dict[str, Any]:
    # sentiment is reserved in schema; classifier will later fill it.
    params = {
        "q": f'content:"{keyword}"',
        "wt": "json",
        "rows": rows,
        "facet": "true",
        "facet.field": "sentiment",
        "facet.limit": 20,
    }
    return run_facet_query(session, solr_select_url, params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Facet search demo (multifaceted indexing/ranking).")
    parser.add_argument("--solr-base", default=DEFAULT_SOLR_BASE)
    parser.add_argument("--core", default=DEFAULT_CORE)
    parser.add_argument("--keyword", default="disappointing")
    args = parser.parse_args()

    solr_select_url = f"{args.solr_base}/{args.core}/select"
    session = requests.Session()

    print("\n[facet] By movie_title:")
    data = facets_by_movie_title(session, solr_select_url, keyword=args.keyword)
    for ft in data.get("facet_counts", {}).get("facet_fields", {}).get("movie_title", [])[:20]:
        print(ft, end=" ")
    print()

    print("\n[facet] By author_rating ranges:")
    data = facets_by_author_rating_ranges(session, solr_select_url, keyword=args.keyword, start=1, end=10, gap=3)
    print("facet ranges:", data.get("facet_counts", {}).get("facet_ranges", {}).get("author_rating", []))

    print("\n[facet] By year:")
    data = facets_by_year(session, solr_select_url, keyword=args.keyword)
    year_facets = data.get("facet_counts", {}).get("facet_fields", {}).get("year", [])
    print("top years:", year_facets[:20])

    print("\n[facet] By sentiment (reserved field):")
    data = facets_by_sentiment(session, solr_select_url, keyword=args.keyword)
    print("sentiment facets:", data.get("facet_counts", {}).get("facet_fields", {}).get("sentiment", []))


if __name__ == "__main__":
    main()

