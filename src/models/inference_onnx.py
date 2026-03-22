"""
Run ShellVibe inference via an exported ONNX model using onnxruntime directly.

Usage:
    # Single instruction
    python src/models/inference_onnx.py --onnx-dir onnx_model --instruction "list all files by size"

    # Interactive mode
    python src/models/inference_onnx.py --onnx-dir onnx_model

Dependencies:
    pip install torch transformers onnxruntime  (onnxruntime-gpu for CUDA)
"""

import torch
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "You are a helpful assistant that converts natural language instructions "
    "into shell commands. Output only the shell command, nothing else."
)


def load_session(onnx_dir: str) -> ort.InferenceSession:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    print(f"ORT providers: {providers}")
    onnx_file = str(Path(onnx_dir) / "model.onnx")
    return ort.InferenceSession(onnx_file, providers=providers)


def predict(
    nl_instruction: str,
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
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
        return_tensors="np",  # get numpy arrays directly
    )

    input_ids: np.ndarray = inputs["input_ids"]  # (1, seq)
    attention_mask: np.ndarray = inputs["attention_mask"]  # (1, seq)
    im_end_id: int = tokenizer.convert_tokens_to_ids("<|im_end|>")
    prompt_len = input_ids.shape[1]

    for _ in range(max_new_tokens):
        logits = session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[
            0
        ]  # (1, seq, vocab)

        next_token = int(np.argmax(logits[0, -1, :]))
        if next_token == im_end_id or next_token == tokenizer.eos_token_id:
            break

        input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
        attention_mask = np.ones_like(input_ids)

    new_tokens = input_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def interactive_mode(session: ort.InferenceSession, tokenizer: AutoTokenizer):
    print("\nShellVibe ONNX Interactive Mode — type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            nl = input("Instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if nl.lower() in ("quit", "exit", ""):
            break

        command = predict(nl, session, tokenizer)
        print(f"Command:     {command}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ShellVibe ONNX inference: natural language → shell command"
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        required=True,
        help="Path to the exported ONNX model directory.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="A single NL instruction. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate (default: 128).",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.onnx_dir)
    session = load_session(args.onnx_dir)

    if args.instruction:
        command = predict(args.instruction, session, tokenizer, args.max_new_tokens)
        print(f"\nInstruction: {args.instruction}")
        print(f"Command:     {command}")
    else:
        interactive_mode(session, tokenizer)


if __name__ == "__main__":
    main()


SYSTEM_PROMPT = (
    "You are a helpful assistant that converts natural language instructions "
    "into shell commands. Output only the shell command, nothing else."
)


def load_model(onnx_dir: str):
    if torch.cuda.is_available():
        provider = "CUDAExecutionProvider"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        provider = "CPUExecutionProvider"
        print("Using CPU")

    print(f"Loading ONNX model from {onnx_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
    # ORT manages device placement internally — inputs stay on CPU
    model = ORTModelForCausalLM.from_pretrained(onnx_dir, provider=provider)
    return model, tokenizer


def predict(
    nl_instruction: str,
    model: ORTModelForCausalLM,
    tokenizer: AutoTokenizer,
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
    )

    input_len = inputs["input_ids"].shape[-1]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=im_end_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def interactive_mode(model: ORTModelForCausalLM, tokenizer: AutoTokenizer):
    print("\nShellVibe ONNX Interactive Mode — type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            nl = input("Instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if nl.lower() in ("quit", "exit", ""):
            break

        command = predict(nl, model, tokenizer)
        print(f"Command:     {command}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ShellVibe ONNX inference: natural language → shell command"
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        required=True,
        help="Path to the exported ONNX model directory.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="A single NL instruction. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate (default: 128).",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.onnx_dir)

    if args.instruction:
        command = predict(args.instruction, model, tokenizer, args.max_new_tokens)
        print(f"\nInstruction: {args.instruction}")
        print(f"Command:     {command}")
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
