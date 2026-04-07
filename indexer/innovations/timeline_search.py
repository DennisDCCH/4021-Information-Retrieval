import argparse
from typing import Any, Dict

import requests


DEFAULT_SOLR_BASE = "http://localhost:8983/solr"
DEFAULT_CORE = "mcu_reviews"


def build_timeline_fq(start_iso: str, end_iso: str) -> str:
    # Solr pdate range query uses ISO-8601 with timezone.
    return f"submission_date:[{start_iso} TO {end_iso}]"


def run_timeline_query(
    solr_select_url: str,
    session: requests.Session,
    keyword: str,
    start_iso: str,
    end_iso: str,
    rows: int = 10,
) -> Dict[str, Any]:
    fq = build_timeline_fq(start_iso, end_iso)
    params: Dict[str, Any] = {
        "q": f'content:"{keyword}"',
        "fq": fq,
        "rows": rows,
        "wt": "json",
        "fl": "id,movie_title,summary,submission_date",
    }
    resp = session.get(solr_select_url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Timeline search demo (date range filter).")
    parser.add_argument("--solr-base", default=DEFAULT_SOLR_BASE)
    parser.add_argument("--core", default=DEFAULT_CORE)
    parser.add_argument("--keyword", default="performance")
    parser.add_argument("--start", default="2019-01-01T00:00:00Z")
    parser.add_argument("--end", default="2021-12-31T00:00:00Z")
    parser.add_argument("--rows", type=int, default=10)
    args = parser.parse_args()

    solr_select_url = f"{args.solr_base}/{args.core}/select"
    session = requests.Session()

    data = run_timeline_query(
        solr_select_url=solr_select_url,
        session=session,
        keyword=args.keyword,
        start_iso=args.start,
        end_iso=args.end,
        rows=args.rows,
    )

    print("Timeline fq:", build_timeline_fq(args.start, args.end))
    print("numFound:", data.get("response", {}).get("numFound", 0))
    for d in data.get("response", {}).get("docs", [])[:5]:
        print("-", d.get("id"), d.get("movie_title"), d.get("submission_date"))


if __name__ == "__main__":
    main()

