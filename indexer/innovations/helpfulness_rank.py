import argparse
from typing import Any, Dict

import requests


DEFAULT_SOLR_BASE = "http://localhost:8983/solr"
DEFAULT_CORE = "mcu_reviews"


def run(session: requests.Session, solr_select_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = session.get(solr_select_url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Helpfulness-based ranking demo.")
    parser.add_argument("--solr-base", default=DEFAULT_SOLR_BASE)
    parser.add_argument("--core", default=DEFAULT_CORE)
    parser.add_argument("--keyword", default="masterpiece")
    parser.add_argument("--rows", type=int, default=10)
    args = parser.parse_args()

    solr_select_url = f"{args.solr_base}/{args.core}/select"
    session = requests.Session()

    fl = "id,movie_title,summary,helpfulness_score"

    # Baseline: default BM25 ranking
    baseline_params = {
        "q": f'content:{args.keyword}',
        "rows": args.rows,
        "wt": "json",
        "fl": fl,
    }
    data = run(session, solr_select_url, baseline_params)
    print("\n[rank] baseline(BM25):")
    for d in data.get("response", {}).get("docs", [])[:5]:
        print("-", d.get("id"), d.get("movie_title"), "helpfulness=", d.get("helpfulness_score"))

    # Strong signal: sort by helpfulness_score
    sort_params = {
        "q": f'content:{args.keyword}',
        "sort": "helpfulness_score desc",
        "rows": args.rows,
        "wt": "json",
        "fl": fl,
    }
    data = run(session, solr_select_url, sort_params)
    print("\n[rank] sort(helpfulness_score desc):")
    for d in data.get("response", {}).get("docs", [])[:5]:
        print("-", d.get("id"), d.get("movie_title"), "helpfulness=", d.get("helpfulness_score"))

    # Boost: use bf (boost functions). This depends on Solr config/version support,
    # but it's a reasonable Lucene/Solr-native way to integrate the signal.
    boost_params = {
        "q": f'content:{args.keyword}',
        "bf": "helpfulness_score",
        "rows": args.rows,
        "wt": "json",
        "fl": fl,
    }
    data = run(session, solr_select_url, boost_params)
    print("\n[rank] boosted(bf=helpfulness_score):")
    for d in data.get("response", {}).get("docs", [])[:5]:
        print("-", d.get("id"), d.get("movie_title"), "helpfulness=", d.get("helpfulness_score"))


if __name__ == "__main__":
    main()

