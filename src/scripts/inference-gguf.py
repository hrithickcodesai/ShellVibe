import argparse
from llama_cpp import Llama

SYSTEM_PROMPT = (
    "You are a helpful assistant that converts natural language instructions "
    "into shell commands. Output only the shell command, nothing else."
)


def load_model(model_path: str, n_gpu_layers: int = -1) -> Llama:
    print(f"Loading {model_path}...")
    return Llama(
        model_path=model_path,
        n_ctx=512,
        n_gpu_layers=n_gpu_layers,
        n_threads=8,
        verbose=False,
    )


def predict(nl_instruction: str, model: Llama, max_new_tokens: int = 128) -> str:
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
    return response["choices"][0]["message"]["content"].strip()


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
        command = predict(nl, model)
        print(f"Command:     {command}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ShellVibe GGUF inference: natural language → shell command"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the GGUF model file (e.g. gguf-models/best_edit_distance.gguf)",
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
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=0,
        help="-1 = full GPU offload, 0 = CPU only",
    )
    args = parser.parse_args()

    model = load_model(args.model_path, args.n_gpu_layers)

    if args.instruction:
        command = predict(args.instruction, model, args.max_new_tokens)
        print(f"\nInstruction: {args.instruction}")
        print(f"Command:     {command}")
    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()
