#!/usr/bin/env python3
"""
Run all Question 3 innovation experiments against a live Solr core.

Usage (from indexer/):
  python innovations/run_experiments.py
  python innovations/run_experiments.py --solr-base http://127.0.0.1:8983/solr

Writes: innovations/experiment_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_SOLR_BASE = os.environ.get("SOLR_BASE", "http://127.0.0.1:8983/solr")
DEFAULT_CORE = os.environ.get("SOLR_CORE", "mcu_reviews")

TOP_N = 5
FACET_MOVIE_LIMIT = 10
FACET_YEAR_LIMIT = 10


def _select_url(base: str, core: str) -> str:
    return f"{base.rstrip('/')}/{core}/select"


def _get(session: requests.Session, url: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    resp = session.get(url, params=params, timeout=120)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    return resp.json(), elapsed_ms


def _avg_helpfulness(docs: List[Dict[str, Any]]) -> Optional[float]:
    vals: List[float] = []
    for d in docs:
        h = d.get("helpfulness_score")
        if h is None:
            continue
        try:
            vals.append(float(h))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def _facet_pairs(facet_list: List[Any]) -> List[Tuple[str, int]]:
    """Solr facet_fields: [name, count, name, count, ...]."""
    out: List[Tuple[str, int]] = []
    i = 0
    while i + 1 < len(facet_list):
        name = facet_list[i]
        count = facet_list[i + 1]
        try:
            c = int(count)
        except (TypeError, ValueError):
            c = 0
        out.append((str(name), c))
        i += 2
    return out


def exp_movie_title_copy(
    session: requests.Session,
    select_url: str,
) -> Dict[str, Any]:
    """Before: exact string field. After: text_en copy-field."""
    variants = [
        {
            "label": "before_string_exact_movie_title",
            "q": 'movie_title:"infinity war"',
            "params": {
                "q": 'movie_title:"infinity war"',
                "rows": 0,
                "wt": "json",
            },
        },
        {
            "label": "after_text_en_movie_title_t",
            "q": "movie_title_t:(infinity war)",
            "params": {
                "q": "movie_title_t:(infinity war)",
                "rows": 0,
                "wt": "json",
            },
        },
    ]
    results: List[Dict[str, Any]] = []
    for v in variants:
        data, elapsed_ms = _get(session, select_url, v["params"])
        num = data.get("response", {}).get("numFound", 0)
        results.append(
            {
                "label": v["label"],
                "query": v["q"],
                "numFound": num,
                "elapsed_ms": round(elapsed_ms, 2),
            }
        )
    return {"title": "movie_title_t copy-field vs string movie_title", "variants": results}


def exp_helpfulness_ranking(
    session: requests.Session,
    select_url: str,
    keyword: str,
) -> Dict[str, Any]:
    fl = "id,movie_title,helpfulness_score"
    rows = max(TOP_N, 10)

    variants: List[Dict[str, Any]] = [
        {
            "label": "bm25_default",
            "params": {
                "q": f"content:{keyword}",
                "rows": rows,
                "wt": "json",
                "fl": fl,
            },
        },
        {
            "label": "sort_helpfulness_desc",
            "params": {
                "q": f"content:{keyword}",
                "sort": "helpfulness_score desc",
                "rows": rows,
                "wt": "json",
                "fl": fl,
            },
        },
        {
            "label": "edismax_bf_helpfulness",
            "params": {
                "defType": "edismax",
                "q": keyword,
                "qf": "content",
                "bf": "helpfulness_score",
                "rows": rows,
                "wt": "json",
                "fl": fl,
            },
        },
    ]

    out_variants: List[Dict[str, Any]] = []
    for v in variants:
        data, elapsed_ms = _get(session, select_url, v["params"])
        docs = data.get("response", {}).get("docs", [])[:TOP_N]
        top = [
            {
                "id": d.get("id"),
                "movie_title": d.get("movie_title"),
                "helpfulness_score": d.get("helpfulness_score"),
            }
            for d in docs
        ]
        avg_h = _avg_helpfulness(docs)
        out_variants.append(
            {
                "label": v["label"],
                "params": {k: v["params"][k] for k in v["params"] if k != "rows"},
                "top_n": top,
                "avg_helpfulness_top_n": round(avg_h, 6) if avg_h is not None else None,
                "elapsed_ms": round(elapsed_ms, 2),
                "numFound": data.get("response", {}).get("numFound", 0),
            }
        )

    return {"title": f"helpfulness ranking (keyword={keyword!r})", "variants": out_variants}


def exp_facets(
    session: requests.Session,
    select_url: str,
    keyword: str,
) -> Dict[str, Any]:
    base_q = f'content:"{keyword}"'
    # Solr expects repeated facet.field keys; use a list of tuples for requests.
    params: List[Tuple[str, Any]] = [
        ("q", base_q),
        ("rows", 0),
        ("wt", "json"),
        ("facet", "true"),
        ("facet.field", "movie_title"),
        ("facet.field", "year"),
        ("facet.field", "sentiment"),
        ("facet.limit", FACET_MOVIE_LIMIT),
        ("facet.mincount", 1),
    ]
    t0 = time.perf_counter()
    resp = session.get(select_url, params=params, timeout=120)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    fc = data.get("facet_counts", {}).get("facet_fields", {})
    movie_pairs = _facet_pairs(fc.get("movie_title", []))[:FACET_MOVIE_LIMIT]
    year_pairs = _facet_pairs(fc.get("year", []))[:FACET_YEAR_LIMIT]

    # Range facet on author_rating
    range_params: Dict[str, Any] = {
        "q": base_q,
        "rows": 0,
        "wt": "json",
        "facet": "true",
        "facet.range": "author_rating",
        "facet.range.start": 1,
        "facet.range.end": 10,
        "facet.range.gap": 3,
    }
    data_r, _ = _get(session, select_url, range_params)
    range_counts = data_r.get("facet_counts", {}).get("facet_ranges", {}).get("author_rating", {}).get("counts", [])

    return {
        "title": f"faceted search (keyword={keyword!r})",
        "numFound": data.get("response", {}).get("numFound", 0),
        "elapsed_ms": round(elapsed_ms, 2),
        "facet_movie_title_top": [{"movie": m, "count": c} for m, c in movie_pairs],
        "facet_year_top": [{"year": y, "count": c} for y, c in year_pairs],
        "facet_author_rating_ranges": range_counts,
        "sentiment_facet_raw": _facet_pairs(fc.get("sentiment", [])),
    }


def exp_timeline(
    session: requests.Session,
    select_url: str,
    keyword: str,
) -> Dict[str, Any]:
    base = f'content:"{keyword}"'
    windows = [
        ("unfiltered", None),
        ("post_endgame_2019_2021", "submission_date:[2019-01-01T00:00:00Z TO 2021-12-31T23:59:59Z]"),
        ("phase4_2022_2024", "submission_date:[2022-01-01T00:00:00Z TO 2024-12-31T23:59:59Z]"),
    ]
    rows: List[Dict[str, Any]] = []
    for label, fq in windows:
        params: Dict[str, Any] = {
            "q": base,
            "rows": 0,
            "wt": "json",
        }
        if fq:
            params["fq"] = fq
        data, elapsed_ms = _get(session, select_url, params)
        rows.append(
            {
                "label": label,
                "filter_query": fq,
                "numFound": data.get("response", {}).get("numFound", 0),
                "elapsed_ms": round(elapsed_ms, 2),
            }
        )
    return {"title": f"timeline / date-range (keyword={keyword!r})", "windows": rows}


def exp_sentiment_readiness(
    session: requests.Session,
    select_url: str,
    keyword: str,
) -> Dict[str, Any]:
    """Document sentiment facet state (empty until ingest joins classifier output)."""
    params = {
        "q": f'content:"{keyword}"',
        "rows": 0,
        "wt": "json",
        "facet": "true",
        "facet.field": "sentiment",
        "facet.limit": 20,
        "facet.mincount": 1,
    }
    data, elapsed_ms = _get(session, select_url, params)
    fc = data.get("facet_counts", {}).get("facet_fields", {}).get("sentiment", [])
    pairs = _facet_pairs(fc)
    return {
        "title": "sentiment facet (reserved field; populate via classifier merge in ingest)",
        "keyword": keyword,
        "numFound": data.get("response", {}).get("numFound", 0),
        "elapsed_ms": round(elapsed_ms, 2),
        "sentiment_counts_non_empty": [{"value": a, "count": b} for a, b in pairs],
        "note": "If empty: sentiment field exists in schema but ingest.py sets empty string until "
        "merged with classification/sentiment_analysis/results/test_dataset_results.csv (or full run).",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Solr innovation experiments for Question 3.")
    parser.add_argument("--solr-base", default=DEFAULT_SOLR_BASE)
    parser.add_argument("--core", default=DEFAULT_CORE)
    parser.add_argument("--out", default="", help="Output JSON path (default: innovations/experiment_results.json)")
    args = parser.parse_args()

    select_url = _select_url(args.solr_base, args.core)
    out_path = args.out
    if not out_path:
        out_path = os.path.join(os.path.dirname(__file__), "experiment_results.json")

    session = requests.Session()

    # Health check (core-level ping; /solr/admin/ping is not valid on all Solr versions)
    try:
        ping_url = f"{args.solr_base.rstrip('/')}/{args.core}/admin/ping"
        session.get(ping_url, params={"wt": "json"}, timeout=10).raise_for_status()
    except Exception as exc:
        print(f"ERROR: Cannot reach Solr at {args.solr_base}: {exc}")
        print("Start Solr: cd indexer && docker compose up -d")
        raise SystemExit(1)

    report: Dict[str, Any] = {
        "solr_base": args.solr_base,
        "core": args.core,
        "generated_at_unix": time.time(),
        "experiments": {},
    }

    print("=== Experiment 1: movie_title vs movie_title_t ===")
    e1 = exp_movie_title_copy(session, select_url)
    report["experiments"]["exp1_movie_title_copy_field"] = e1
    for v in e1["variants"]:
        print(f"  {v['label']}: numFound={v['numFound']} q={v['query']!r} ({v['elapsed_ms']} ms)")

    print("\n=== Experiment 2: helpfulness ranking (masterpiece) ===")
    e2a = exp_helpfulness_ranking(session, select_url, "masterpiece")
    report["experiments"]["exp2a_helpfulness_masterpiece"] = e2a
    for v in e2a["variants"]:
        print(f"  {v['label']}: avg_helpfulness_top5={v['avg_helpfulness_top_n']} numFound={v['numFound']}")

    print("\n=== Experiment 2b: helpfulness ranking (disappointing) ===")
    e2b = exp_helpfulness_ranking(session, select_url, "disappointing")
    report["experiments"]["exp2b_helpfulness_disappointing"] = e2b
    for v in e2b["variants"]:
        print(f"  {v['label']}: avg_helpfulness_top5={v['avg_helpfulness_top_n']} numFound={v['numFound']}")

    print("\n=== Experiment 3: faceted search (disappointing) ===")
    e3 = exp_facets(session, select_url, "disappointing")
    report["experiments"]["exp3_facets_disappointing"] = e3
    print(f"  numFound={e3['numFound']}")
    print("  top movies:", e3["facet_movie_title_top"][:5])

    print("\n=== Experiment 4: timeline (performance) ===")
    e4 = exp_timeline(session, select_url, "performance")
    report["experiments"]["exp4_timeline_performance"] = e4
    for w in e4["windows"]:
        print(f"  {w['label']}: numFound={w['numFound']} ({w['elapsed_ms']} ms)")

    print("\n=== Experiment 5: sentiment facet readiness ===")
    e5 = exp_sentiment_readiness(session, select_url, "disappointing")
    report["experiments"]["exp5_sentiment_facet"] = e5
    print(f"  non-empty sentiment buckets: {e5['sentiment_counts_non_empty']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
