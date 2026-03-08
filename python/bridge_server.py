"""
Julia ↔ Python bridge server.

Runs as a subprocess, communicates via JSON over stdin/stdout.
Loads the Qwen base model once, then processes requests.

Protocol:
  Request:  {"cmd": "...", ...}\n
  Response: {"status": "ok", ...}\n  or  {"status": "error", "message": "..."}\n

Commands:
  get_embeddings: {"cmd": "get_embeddings", "text": "..."}
    → {"status": "ok", "embeddings": "<base64 float32>", "shape": [d, T], "tokens": [...]}

  generate: {"cmd": "generate", "prompt": "...", "max_tokens": 128, "num_candidates": 4}
    → {"status": "ok", "candidates": [...]}

  generate_conditioned: {"cmd": "generate_conditioned", "prompt": "...",
                         "conditioning": "<base64 float32>", "cond_shape": [d, M],
                         "max_tokens": 128, "num_candidates": 4}
    → {"status": "ok", "candidates": [...]}

  config: {"cmd": "config"}
    → {"status": "ok", "d_model": 896, ...}

  ping: {"cmd": "ping"} → {"status": "ok"}
  shutdown: {"cmd": "shutdown"} → process exits
"""

import sys
import json
import base64
import numpy as np
from pathlib import Path

# Ensure parent dir is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from python.base_model import BaseModel


def encode_array(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def decode_array(b64: str, shape: list[int]) -> np.ndarray:
    data = base64.b64decode(b64)
    return np.frombuffer(data, dtype=np.float32).reshape(shape)


def send(obj: dict):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    # Signal ready before loading (so Julia knows process started)
    # Load model
    print("Loading base model...", file=sys.stderr)
    model = BaseModel()
    print(f"Model loaded: d_model={model.d_model}", file=sys.stderr)

    # Signal ready
    send({"status": "ready", "d_model": model.d_model})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            send({"status": "error", "message": f"JSON parse error: {e}"})
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "ping":
                send({"status": "ok"})

            elif cmd == "shutdown":
                send({"status": "ok"})
                break

            elif cmd == "config":
                cfg = model.get_config()
                cfg["status"] = "ok"
                send(cfg)

            elif cmd == "get_embeddings":
                text = req["text"]
                embeddings, tokens = model.get_embeddings(text)
                send({
                    "status": "ok",
                    "embeddings": encode_array(embeddings),
                    "shape": list(embeddings.shape),
                    "tokens": tokens,
                })

            elif cmd == "generate":
                prompt = req["prompt"]
                max_tokens = req.get("max_tokens", 128)
                num_candidates = req.get("num_candidates", 4)
                candidates = model.generate_conditioned(
                    prompt,
                    max_new_tokens=max_tokens,
                    num_beams=num_candidates,
                    num_return_sequences=num_candidates,
                )
                send({"status": "ok", "candidates": candidates})

            elif cmd == "generate_conditioned":
                prompt = req["prompt"]
                max_tokens = req.get("max_tokens", 128)
                num_candidates = req.get("num_candidates", 4)
                conditioning = None
                if "conditioning" in req and req["conditioning"] is not None:
                    conditioning = decode_array(req["conditioning"], req["cond_shape"])
                proj_matrix = None
                if "proj_matrix" in req and req["proj_matrix"] is not None:
                    proj_matrix = decode_array(req["proj_matrix"], req["proj_shape"])
                candidates = model.generate_conditioned(
                    prompt,
                    conditioning=conditioning,
                    proj_matrix=proj_matrix,
                    max_new_tokens=max_tokens,
                    num_beams=num_candidates,
                    num_return_sequences=num_candidates,
                )
                send({"status": "ok", "candidates": candidates})

            else:
                send({"status": "error", "message": f"Unknown command: {cmd}"})

        except Exception as e:
            send({"status": "error", "message": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
