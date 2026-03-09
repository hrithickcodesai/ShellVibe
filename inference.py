import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
SYSTEM_PROMPT = (
    "You are a helpful assistant that converts natural language instructions "
    "into shell commands. Output only the shell command, nothing else."
)


def load_model(checkpoint_path: str | None, device: str, dtype: torch.dtype):
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading base model from {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="sdpa" if device == "cuda" else "eager",
    )

    if checkpoint_path:
        print(f"Loading fine-tuned weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("Checkpoint loaded.")

    model.to(device)
    model.eval()
    return model, tokenizer


def predict(
    nl_instruction: str, model, tokenizer, device: str, max_new_tokens: int = 128
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": nl_instruction.lower().strip()},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def setup_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        print("Using MPS (Mac Silicon)")
        return "mps", torch.float32
    else:
        print("Using CPU")
        return "cpu", torch.float32


def interactive_mode(model, tokenizer, device: str):
    print("\nShellVibe Interactive Mode — type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            nl = input("Instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if nl.lower() in ("quit", "exit", ""):
            break

        command = predict(nl, model, tokenizer, device)
        print(f"Command:     {command}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ShellVibe inference: natural language → shell command"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a fine-tuned checkpoint .pt file (e.g. checkpoints/best_loss.pt). "
        "If omitted, uses the base pretrained model.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="A single NL instruction to convert. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate (default: 128).",
    )
    args = parser.parse_args()

    device, dtype = setup_device()
    model, tokenizer = load_model(args.checkpoint, device, dtype)

    if args.instruction:
        command = predict(
            args.instruction, model, tokenizer, device, args.max_new_tokens
        )
        print(f"\nInstruction: {args.instruction}")
        print(f"Command:     {command}")
    else:
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
