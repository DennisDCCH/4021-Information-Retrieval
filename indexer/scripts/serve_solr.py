#!/usr/bin/env python3
"""Minimal UI + API server backed by the Solr indexer in indexer/."""

import argparse
import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen


def esc(value):
    text = str(value or "")
    escaped = []
    for ch in text:
        if ch in '+-&|!(){}[]^"~*?:\\/':
            escaped.append("\\" + ch)
        else:
            escaped.append(ch)
    return "".join(escaped)


def phrase_terms(value):
    return [esc(part) for part in str(value or "").split() if part.strip()]


class SolrSearchClient:
    def __init__(self, solr_base, core):
        self.select_url = f"{solr_base.rstrip('/')}/{core}/select"

    def search(self, params):
        query = (params.get("q", [""])[0] or "").strip()
        top_k = int(params.get("k", ["10"])[0])
        movie = (params.get("movie", [""])[0] or "").strip()
        min_rating = (params.get("min_rating", [""])[0] or "").strip()
        sort = (params.get("sort", ["score desc"])[0] or "score desc").strip()
        start_year = (params.get("start_year", [""])[0] or "").strip()
        end_year = (params.get("end_year", [""])[0] or "").strip()

        if not query:
            return {"query": "", "filters": {}, "time_ms": 0.0, "results": []}

        solr_params = {
            "q": f'summary:({query}) OR content:({query}) OR movie_title_t:({query})',
            "defType": "lucene",
            "rows": str(top_k),
            "wt": "json",
            "fl": (
                "id,movie_title,summary,content,author,author_rating,upvotes,"
                "downvotes,helpfulness_score,submission_date,score"
            ),
            "sort": sort,
        }

        filters = []
        if movie:
            movie_terms = phrase_terms(movie)
            if movie_terms:
                filters.append("movie_title_t:(" + " AND ".join(movie_terms) + ")")
        if min_rating:
            filters.append(f"author_rating:[{esc(min_rating)} TO *]")
        if start_year or end_year:
            start = f"{start_year}-01-01T00:00:00Z" if start_year else "*"
            end = f"{end_year}-12-31T23:59:59Z" if end_year else "*"
            filters.append(f"submission_date:[{start} TO {end}]")
        if filters:
            solr_params["fq"] = filters

        request_url = f"{self.select_url}?{urlencode(solr_params, doseq=True)}"
        t0 = time.time()
        with urlopen(request_url) as resp:
            raw = json.load(resp)
        elapsed_ms = (time.time() - t0) * 1000.0

        docs = raw.get("response", {}).get("docs", [])
        results = []
        for doc in docs:
            date = doc.get("submission_date", "")
            if date.endswith("T00:00:00Z"):
                date = date[:10]
            results.append({
                "id": doc.get("id", ""),
                "movie_title": doc.get("movie_title", ""),
                "title": doc.get("summary", ""),
                "summary": doc.get("summary", ""),
                "text": doc.get("content", ""),
                "author": doc.get("author", ""),
                "author_rating": doc.get("author_rating", ""),
                "upvotes": doc.get("upvotes", ""),
                "downvotes": doc.get("downvotes", ""),
                "helpfulness_score": doc.get("helpfulness_score", ""),
                "date": date,
                "score": round(float(doc.get("score", 0.0)), 6),
            })

        return {
            "query": query,
            "filters": {
                "movie": movie,
                "min_rating": min_rating,
                "sort": sort,
                "start_year": start_year,
                "end_year": end_year,
            },
            "time_ms": round(elapsed_ms, 2),
            "results": results,
        }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path):
        if not os.path.exists(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            data = f.read()
        content_type = "text/html; charset=utf-8"
        if path.endswith(".css"):
            content_type = "text/css; charset=utf-8"
        if path.endswith(".js"):
            content_type = "application/javascript; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/search":
            params = parse_qs(parsed.query)
            try:
                payload = self.server.search_client.search(params)
                self._send_json(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if parsed.path == "/" or parsed.path == "":
            return self._send_file(os.path.join(self.server.web_root, "index.html"))
        return self._send_file(os.path.join(self.server.web_root, parsed.path.lstrip("/")))


class AppServer(HTTPServer):
    def __init__(self, server_address, handler_cls, search_client, web_root):
        super().__init__(server_address, handler_cls)
        self.search_client = search_client
        self.web_root = web_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solr-base", default="http://127.0.0.1:8983/solr")
    parser.add_argument("--core", default="mcu_reviews")
    default_web_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web_solr")
    parser.add_argument("--web", default=default_web_root, help="Web root directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    search_client = SolrSearchClient(args.solr_base, args.core)
    server = AppServer((args.host, args.port), Handler, search_client=search_client, web_root=args.web)
    print(f"Serving Solr UI at http://{args.host}:{args.port}")
    print(f"Proxying queries to {args.solr_base.rstrip('/')}/{args.core}")
    print("Press Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
