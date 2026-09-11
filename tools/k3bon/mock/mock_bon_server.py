#!/usr/bin/env python3
"""Minimal streaming server for bon_replay.py, CPU only.

Its purpose is to make the identity gates FAIL on demand. A gate only ever seen
passing is untested, and the specific failure this reproduces is the real one:
vLLM's serving.py:641-645 suppresses token_ids when include_reasoning is false on
a model with a reasoning parser, so a driver that does not demand ids would score
"identical" over empty lists and report a pass.

Modes via env BON_MOCK_MODE:
  ids        normal: ids on the stream choice, deterministic per (seed, row)
  no_ids     ids suppressed -- reproduces hide_stream_metadata
  drift      ids differ from `ids` mode at one position (fakes KV contamination)
  identical  same ids regardless of policy (fakes a patch that never branches)

Also serves /metrics so bench_k3h_block5.py's snapshot/delta works.
"""
import json, os, hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer

MODE = os.environ.get("BON_MOCK_MODE", "ids")
_ACC = {"acc": 0.0, "draft": 0.0, "drafts": 0.0}


def ids_for(prompt, seed, n):
    # The hash key must NOT include MODE: drift has to be a SINGLE-position change
    # off an otherwise identical sequence, which is what real KV contamination looks
    # like. Keying the hash on the mode diverges at position 0 and would let a gate
    # that only compares sequence length or first token look like it works.
    h = hashlib.sha256(f"{prompt}|{seed}".encode()).digest()
    out = [1000 + (h[i % len(h)] + i) % 500 for i in range(n)]
    if MODE == "drift":
        out[min(3, n - 1)] = 49999
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json({"data": [{"id": "kimi-k3"}]})
        elif self.path.startswith("/metrics"):
            body = (
                f'vllm:spec_decode_num_accepted_tokens_total{{model_name="kimi-k3"}} {_ACC["acc"]}\n'
                f'vllm:spec_decode_num_draft_tokens_total{{model_name="kimi-k3"}} {_ACC["draft"]}\n'
                f'vllm:spec_decode_num_drafts_total{{model_name="kimi-k3"}} {_ACC["drafts"]}\n'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "nope"}, 404)

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        mt = int(req.get("max_tokens") or 8)
        seed = req.get("seed")
        want = bool(req.get("return_token_ids"))
        incl = bool(req.get("include_reasoning"))
        prompt = json.dumps(req.get("messages"))
        ntok = min(mt, 12)
        # The real trap: ids are withheld unless include_reasoning is also true.
        give = want and incl and MODE != "no_ids"
        seed_key = seed if MODE != "identical" else 0
        ids = ids_for(prompt, seed_key, ntok)
        _ACC["drafts"] += ntok / 5.0
        _ACC["draft"] += ntok
        _ACC["acc"] += ntok * 0.6

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(ntok):
            ch = {"index": 0, "delta": {"content": f"t{ids[i]} "},
                  "finish_reason": None}
            if give:
                ch["token_ids"] = [ids[i]]
            self._send({"choices": [ch]})
        self._send({"choices": [{"index": 0, "delta": {},
                                 "finish_reason": "length"}]})
        self._send({"choices": [], "usage": {"completion_tokens": ntok}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18930
    print(f"[mock] mode={MODE} port={port}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
