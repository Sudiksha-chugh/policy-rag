"""Local REST interface. Run with python3 app.py."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError

from rag import PolicyRag


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self.respond(200, {"status": "ok"})
        else:
            self.respond(404, {"error": "Route not found"})

    def do_POST(self):
        if self.path != "/ask":
            self.respond(404, {"error": "Route not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError
            body = json.loads(self.rfile.read(length))
            question = body["question"]
            use_graph = body.get("use_graph", True)
            if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1000:
                raise ValueError
            if not isinstance(use_graph, bool):
                raise ValueError
        except (ValueError, TypeError, KeyError):
            self.respond(400, {"error": "Provide a question of 1–1000 characters and an optional Boolean use_graph."})
            return
        try:
            result = self.server.rag.answer(question.strip(), use_graph)
        except (URLError, TimeoutError):
            self.respond(502, {"error": "The configured Ollama model is unavailable."})
            return
        self.respond(200, result)


def main():
    rag = PolicyRag(os.getenv("EMBEDDING_MODEL"), os.getenv("CHAT_MODEL"))
    port = int(os.getenv("PORT", "8000"))
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        server.rag = rag
        print(f"Policy RAG listening on http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
