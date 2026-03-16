import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
DEFAULT_CHECKPOINT = "checkpoints/best_edit_distance.pt"
HOST = "127.0.0.1"
PORT = 7070

SYSTEM_PROMPT = (
    "You are a helpful assistant that converts natural language instructions "
    "into shell commands. Output only the shell command, nothing else."
)


def setup_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_model(checkpoint_path: str | None, device: str, dtype: torch.dtype):
    print(f"[server] Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("[server] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="sdpa" if device == "cuda" else "eager",
    )

    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"[server] Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    elif checkpoint_path:
        print(f"[server] Checkpoint not found: {checkpoint_path}. Using base model.")

    model.to(device)
    model.eval()
    print(f"[server] Model ready on {device}.")
    return model, tokenizer


def predict(
    instruction: str, model, tokenizer, device: str, max_new_tokens: int = 128
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction.strip()},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=im_end_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_model = None
_tokenizer = None
_device = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._respond(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid JSON"})
            return

        instruction = data.get("instruction", "").strip()
        if not instruction:
            self._respond(400, {"error": "missing instruction"})
            return

        max_new_tokens = int(data.get("max_new_tokens", 128))
        command = predict(instruction, _model, _tokenizer, _device, max_new_tokens)
        self._respond(200, {"command": command})

    def _respond(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    global _model, _tokenizer, _device
    device, dtype = setup_device()
    _device = device
    _model, _tokenizer = load_model(args.checkpoint, device, dtype)

    server = HTTPServer((HOST, args.port), Handler)
    print(f"[server] Listening on http://{HOST}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")


if __name__ == "__main__":
    main()
