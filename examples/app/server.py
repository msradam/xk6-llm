#!/usr/bin/env python3
"""A four-surface mock application for examples/composite.js.

Stdlib only. It stands in for the parts of an AI feature that are not the
model: a login, a document lookup the model can call as a tool, a place to
store the answer, and a page that renders it. It records every lookup so a
test can check the application's own record instead of the model's account of
itself.

    python3 examples/app/server.py            # listens on :8089

Routes:
    POST /login                -> sets a session cookie
    GET  /docs/search?q=...    -> {"hits": [...]} and counts the call
    POST /answers              -> stores {"session": ..., "answer": ...}
    GET  /                     -> renders the most recent answer
    GET  /stats                -> {"lookups": N, "answers": N}
    POST /reset                -> zero the counters
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DOCS = {
    "b-tree": "A B-tree index keeps keys sorted in a balanced tree so range scans and point lookups both cost O(log n).",
    "hash index": "A hash index maps a key to a bucket, so equality lookups are O(1) but range queries are not supported.",
    "write-ahead log": "A write-ahead log records changes before they reach the data files, so a crash can be replayed.",
}

state = {"lookups": 0, "answers": 0, "last": ""}
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002, quiet by design
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/docs/search":
            q = parse_qs(url.query).get("q", [""])[0].lower()
            hits = [{"title": k, "text": v} for k, v in DOCS.items() if k in q or q in k]
            with lock:
                state["lookups"] += 1
            return self.send_json({"hits": hits})
        if url.path == "/stats":
            with lock:
                return self.send_json({"lookups": state["lookups"], "answers": state["answers"]})
        if url.path == "/":
            with lock:
                last = state["last"]
            page = f"<!doctype html><title>Assistant</title><h1>Assistant</h1><p id=\"answer\">{last}</p>"
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/login":
            self.send_response(204)
            self.send_header("Set-Cookie", "session=demo; Path=/")
            self.end_headers()
            return
        if self.path == "/answers":
            answer = str(self.read_json().get("answer", ""))
            with lock:
                state["answers"] += 1
                state["last"] = answer.replace("<", "&lt;")
            return self.send_json({"ok": True})
        if self.path == "/reset":
            with lock:
                state.update(lookups=0, answers=0, last="")
            return self.send_json({"ok": True})
        self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8089
    print(f"mock app on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
