#!/usr/bin/env python3
"""websearch MCP server (stdio) - real web search with no API key.

opencode's built-in `websearch` tool needs a paid provider credential
(Exa / Firecrawl / Parallel / Tavily) and fails with none of them set. This
server gives the agent working web search with no credential at all, by
querying DuckDuckGo's public endpoints, and registers as a normal MCP server
so it shows up in the client's MCP list alongside github and lsp.

tools:
  search -> web results (title, url, snippet) for a query
  news   -> recent news results for a query
  fetch  -> read a URL and return it as plain text

Backends, tried in order: DuckDuckGo HTML, DuckDuckGo Lite, then the
Instant Answer API. The first that returns results wins, so a single blocked
endpoint degrades rather than fails.

stdio JSON-RPC only; no third-party deps; no credentials, no telemetry.
"""
import html
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

VERSION = "1.0.0"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TIMEOUT = float(os.environ.get("WEBSEARCH_TIMEOUT", "20"))
MAX_BYTES = int(os.environ.get("WEBSEARCH_MAX_BYTES", "2000000"))


def _get(url, data=None, timeout=None):
    """GET/POST with a browser UA. Returns decoded text or raises."""
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    if data:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as r:
        raw = r.read(MAX_BYTES)
    return raw.decode("utf-8", "replace")


class _Results(HTMLParser):
    """Pull result links + snippets out of DuckDuckGo's HTML/lite pages."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._in_link = False
        self._in_snip = False
        self._href = ""
        self._buf = []
        self._snip = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls, href = a.get("class", ""), a.get("href", "")
        # html endpoint: <a class="result__a" href="//duckduckgo.com/l/?uddg=...">
        # lite endpoint:  <a class="result-link" href="https://...">
        if tag == "a" and ("result__a" in cls or "result-link" in cls):
            self._in_link, self._href, self._buf = True, href, []
        elif "result__snippet" in cls or "result-snippet" in cls:
            self._in_snip, self._snip = True, []

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link:
            self._in_link = False
            title = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            url = self._href
            if url.startswith("//"):
                url = "https:" + url
            # unwrap DDG's /l/?uddg=<encoded> redirect
            m = re.search(r"[?&]uddg=([^&]+)", url)
            if m:
                url = urllib.parse.unquote(m.group(1))
            if title and url.startswith("http"):
                self.results.append({"title": title, "url": url, "snippet": ""})
        elif self._in_snip and tag in ("a", "td", "div"):
            self._in_snip = False
            txt = re.sub(r"\s+", " ", "".join(self._snip)).strip()
            if txt and self.results and not self.results[-1]["snippet"]:
                self.results[-1]["snippet"] = txt

    def handle_data(self, d):
        if self._in_link:
            self._buf.append(d)
        if self._in_snip:
            self._snip.append(d)


def _html_search(query, endpoint, max_results):
    doc = _get(endpoint, data={"q": query, "kl": "wt-wt"})
    p = _Results()
    p.feed(doc)
    return p.results[:max_results]


def _instant_answer(query, max_results):
    """DuckDuckGo's keyless Instant Answer API (good for entities)."""
    doc = _get("https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": "1", "no_redirect": "1"}))
    try:
        d = json.loads(doc)
    except ValueError:
        return []
    out = []
    if d.get("AbstractText"):
        out.append({"title": d.get("Heading") or query, "url": d.get("AbstractURL", ""),
                    "snippet": d["AbstractText"]})
    for t in d.get("RelatedTopics", []) or []:
        if isinstance(t, dict) and t.get("Text"):
            out.append({"title": t.get("Text", "")[:120], "url": t.get("FirstURL", ""),
                        "snippet": t.get("Text", "")})
        if len(out) >= max_results:
            break
    return out[:max_results]


def _search(query, max_results, endpoints):
    errors = []
    for name, fn in endpoints:
        try:
            res = fn()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                ValueError, TimeoutError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        if res:
            return res, None
        errors.append(f"{name}: no results")
    return [], "; ".join(errors) or "all backends failed"


def _clamp_results(n, default=8, lo=1, hi=25):
    """Clamp a caller-supplied result count.

    `int(n or default)` would turn an explicit 0 into the default, so a caller
    asking for zero results silently got eight. Treat None/empty as "unset"
    and anything non-numeric as the default.
    """
    if n is None or n == "":
        n = default
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(n, hi))


def tool_search(query, max_results=8):
    if not query or not str(query).strip():
        return {"query": query, "error": "query is required", "results": []}
    q = str(query).strip()
    n = _clamp_results(max_results)
    res, err = _search(q, n, [
        ("html", lambda: _html_search(q, "https://html.duckduckgo.com/html/", n)),
        ("lite", lambda: _html_search(q, "https://lite.duckduckgo.com/lite/", n)),
        ("instant-answer", lambda: _instant_answer(q, n)),
    ])
    out = {"query": q, "count": len(res), "results": res}
    if not res and err:
        out["error"] = err
    return out


def tool_news(query, max_results=8):
    if not query or not str(query).strip():
        return {"query": query, "error": "query is required", "results": []}
    q = str(query).strip()
    n = _clamp_results(max_results)
    # DuckDuckGo exposes no keyless news endpoint, so bias a web search toward
    # news rather than returning an error.
    res, err = _search(f"{q} news", n, [
        ("html", lambda: _html_search(f"{q} news", "https://html.duckduckgo.com/html/", n)),
        ("lite", lambda: _html_search(f"{q} news", "https://lite.duckduckgo.com/lite/", n)),
        ("instant-answer", lambda: _instant_answer(q, n)),
    ])
    out = {"query": q, "note": "keyless news falls back to web results", "count": len(res),
           "results": res}
    if not res and err:
        out["error"] = err
    return out


def tool_fetch(url, max_chars=8000):
    if not url or not str(url).startswith(("http://", "https://")):
        return {"url": url, "error": "url must be http(s)"}
    doc = _get(str(url))
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", doc, re.I | re.S)
    if m:
        title = re.sub(r"\s+", " ", html.unescape(m.group(1))).strip()
    for bad in ("script", "style", "noscript", "svg"):
        doc = re.sub(rf"<{bad}\b.*?</{bad}>", " ", doc, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", doc)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text).strip()
    cap = max(200, min(int(max_chars or 8000), 200000))
    return {"url": str(url), "title": title, "chars": len(text),
            "text": text[:cap], "truncated": len(text) > cap}


TOOLS = [
    {"name": "search", "description":
     "Search the web. No API key required. Returns title, url and snippet "
     "for each result.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "what to search for"},
         "max_results": {"type": "integer", "description": "1-25 (default 8)"}},
         "required": ["query"]}},
    {"name": "news", "description":
     "Recent news for a query. Keyless, so it falls back to web results "
     "biased toward news.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "max_results": {"type": "integer", "description": "1-25 (default 8)"}},
         "required": ["query"]}},
    {"name": "fetch", "description":
     "Fetch a URL and return it as plain text (scripts and styles removed).",
     "inputSchema": {"type": "object", "properties": {
         "url": {"type": "string"},
         "max_chars": {"type": "integer", "description": "default 8000"}},
         "required": ["url"]}},
]
HANDLERS = {"search": tool_search, "news": tool_news, "fetch": tool_fetch}


def _respond(req, result):
    sys.stdout.write(json.dumps(
        {"jsonrpc": "2.0", "id": req.get("id"), "result": result}) + "\n")
    sys.stdout.flush()


def _error(req, code, message):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req.get("id"),
                                 "error": {"code": code, "message": message}}) + "\n")
    sys.stdout.flush()


def handle(msg):
    # main() already parses the frame; accept a raw string too so this
    # function is safe to call either way. A frame that will not parse is
    # dropped rather than raised: killing the server here shows up in the
    # client only as "Connection closed".
    if isinstance(msg, (str, bytes, bytearray)):
        try:
            msg = json.loads(msg)
        except ValueError:
            return
    if not isinstance(msg, dict):
        return
    req = msg
    m = msg.get("method", "")
    if m == "initialize":
        pv = req.get("params", {}).get("protocolVersion", "2025-06-18")
        _respond(req, {"protocolVersion": pv, "capabilities": {"tools": {}},
                       "serverInfo": {"name": "websearch", "version": VERSION}})
    elif m == "ping":
        _respond(req, {})
    elif m == "tools/list":
        _respond(req, {"tools": TOOLS})
    elif m == "resources/list":
        _respond(req, {"resources": []})
    elif m == "prompts/list":
        _respond(req, {"prompts": []})
    elif m == "tools/call":
        params = req.get("params", {})
        name = params.get("name", "")
        args = params.get("arguments", {}) or {}
        if name not in HANDLERS:
            _respond(req, {"content": [{"type": "text", "text": f"unknown tool: {name}"}],
                           "isError": True})
            return
        try:
            out = HANDLERS[name](**args)
            _respond(req, {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}],
                           "isError": False})
        except Exception as e:  # noqa: BLE001 - a tool must never kill MCP
            _respond(req, {"content": [{"type": "text", "text": f"{name} error: {e}"}],
                           "isError": True})
    elif m.startswith("notifications/"):
        pass
    else:
        if req.get("id") is not None:
            _error(req, -32601, f"method not found: {m}")


def main():
    # opencode frames some messages with Content-Length headers; accept both.
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            break
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(b"content-length"):
            headers = {}
            while True:
                hdr = stdin.readline()
                if not hdr or hdr in (b"\r\n", b"\n"):
                    break
                if b":" in hdr:
                    k, _, v = hdr.partition(b":")
                    headers[k.decode().strip().lower()] = v.decode().strip()
            body = stdin.read(int(headers.get("content-length", "0")) or b"")
        else:
            body = stripped
        try:
            msg = json.loads(body.decode("utf-8", "replace"))
        except ValueError:
            continue
        if isinstance(msg, list):
            for one in msg:
                handle(one)
        else:
            handle(msg)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
