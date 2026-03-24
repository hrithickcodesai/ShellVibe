import argparse
import platform
import time
from llama_cpp import Llama

SYSTEM_PROMPT = (
    "You are a helpful assistant that converts natural language instructions "
    "into shell commands. Output only the shell command, nothing else."
)

MODEL_PATHS = {
    "0.5b": "gguf-models/qwen2.5-0.5b-inst-q8_0.gguf",
    "1.5b": "gguf-models/qwen2.5-1.5b-inst-q8_0.gguf",
    "3b": "gguf-models/qwen2.5-3b-inst-q8_0.gguf",
}


def detect_backend() -> tuple[int, str]:
    """Auto-detect Metal on macOS, CPU otherwise."""
    if platform.system() == "Darwin":
        return -1, "metal"
    return 0, "cpu"


def load_model(model_path: str, n_gpu_layers: int) -> Llama:
    print(f"Loading {model_path}...")
    return Llama(
        model_path=model_path,
        n_ctx=512,
        n_gpu_layers=n_gpu_layers,
        n_threads=8,
        verbose=False,
    )


def predict(
    nl_instruction: str, model: Llama, max_new_tokens: int = 128
) -> tuple[str, float]:
    start = time.perf_counter()
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": nl_instruction.strip()},
        ],
        max_tokens=max_new_tokens,
        temperature=0.0,
        repeat_penalty=1.0,
        stop=["<|im_end|>"],
    )
    elapsed = time.perf_counter() - start
    text = response["choices"][0]["message"]["content"].strip()
    completion_tokens = response.get("usage", {}).get("completion_tokens", 1)
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    return text, tps


def interactive_mode(model: Llama):
    print("\nShellVibe Interactive Mode — type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            nl = input("Instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if nl.lower() in ("quit", "exit", ""):
            break
        command, tps = predict(nl, model)
        print(f"Command:     {command}")
        print(f"Speed:       {tps:.1f} tok/s\n")


def main():
    parser = argparse.ArgumentParser(
        description="ShellVibe GGUF inference: natural language → shell command"
    )
    parser.add_argument(
        "--model",
        choices=["0.5b", "1.5b", "3b"],
        default="3b",
        help="Model size to use",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )
    args = parser.parse_args()

    n_gpu_layers, backend = detect_backend()
    model_path = MODEL_PATHS[args.model]
    print(f"Backend: {backend}")
    model = load_model(model_path, n_gpu_layers)

    if args.instruction:
        command, tps = predict(args.instruction, model, args.max_new_tokens)
        print(f"\nInstruction: {args.instruction}")
        print(f"Command:     {command}")
        print(f"Speed:       {tps:.1f} tok/s")
    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()
