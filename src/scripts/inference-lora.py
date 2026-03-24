import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a helpful assistant that converts natural language instructions "
    "into shell commands. Output only the shell command, nothing else."
)


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


def load_model(
    base_model_id: str,
    adapter_dir: str | None,
    device: str,
    dtype: torch.dtype,
):
    print(f"Loading tokenizer from {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"Loading base model from {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        attn_implementation="sdpa" if device == "cuda" else "eager",
    )

    if adapter_dir:
        print(f"Loading LoRA adapter from {adapter_dir}...")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        print("LoRA adapter loaded and merged into inference graph.")
    else:
        print("No adapter provided — running base model only.")

    model.to(device)
    model.eval()
    return model, tokenizer


def predict(
    nl_instruction: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = 128,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": nl_instruction.strip()},
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


def interactive_mode(model, tokenizer, device: str):
    print("\nShellVibe LoRA Interactive Mode — type 'quit' or 'exit' to stop.\n")
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
        description="ShellVibe LoRA inference: natural language → shell command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for the base model "
        "(e.g. Qwen/Qwen2.5-Coder-7B-Instruct).",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Path to the saved LoRA adapter directory "
        "(e.g. qwen2.5-coder-7b-lora-checkpoints/best_loss_adapter). "
        "If omitted, runs the bare base model.",
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
        help="Maximum number of tokens to generate.",
    )
    args = parser.parse_args()

    device, dtype = setup_device()
    model, tokenizer = load_model(args.base_model_id, args.adapter_dir, device, dtype)

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
