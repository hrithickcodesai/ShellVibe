"""
Export a LoRA adapter to GGUF via two steps:
  1. Merge the adapter into the base model and save as a HuggingFace directory.
  2. Run llama.cpp's convert_hf_to_gguf.py on the merged directory.

Usage:
  python3 -m src.scripts.export-lora-gguf \
    --base_model_id Qwen/Qwen2.5-Coder-3B-Instruct \
    --adapter_dir qwen2.5-coder-3b-lora-checkpoints/best_loss_adapter \
    --output_dir exports/qwen2.5-coder-3b-merged \
    --gguf_out   exports/qwen2.5-coder-3b-q8.gguf \
    --llama_cpp_dir /path/to/llama.cpp \
    --quant_type Q8_0
"""

import argparse
import os
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


QUANT_TYPES = [
    "f32", "f16", "bf16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M", "Q5_K_S",
    "Q4_K_M", "Q4_K_S",
    "Q3_K_M", "Q3_K_S", "Q3_K_L",
    "Q2_K",
]


def merge_and_save(base_model_id: str, adapter_dir: str, output_dir: str):
    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"Loading LoRA adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged HF model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print("Merge complete. Files written:")
    for f in sorted(os.listdir(output_dir)):
        size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1e6
        print(f"  {f:50s}  {size_mb:.1f} MB")


def convert_to_gguf(
    llama_cpp_dir: str,
    merged_dir: str,
    gguf_out: str,
    quant_type: str,
):
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.isfile(convert_script):
        # older llama.cpp versions use convert.py
        convert_script = os.path.join(llama_cpp_dir, "convert.py")
    if not os.path.isfile(convert_script):
        print(
            f"ERROR: could not find convert_hf_to_gguf.py or convert.py in {llama_cpp_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(gguf_out)), exist_ok=True)

    # Step 1: convert to f16 GGUF first (always safe baseline)
    gguf_f16 = gguf_out.replace(".gguf", "_f16.gguf")
    cmd_convert = [
        sys.executable, convert_script,
        merged_dir,
        "--outfile", gguf_f16,
        "--outtype", "f16",
    ]
    print(f"\nRunning: {' '.join(cmd_convert)}")
    subprocess.run(cmd_convert, check=True)
    print(f"f16 GGUF written: {gguf_f16}")

    if quant_type.lower() in ("f16", "f32", "bf16"):
        # No quantisation step needed — just rename
        os.rename(gguf_f16, gguf_out)
        print(f"Output: {gguf_out}")
        return

    # Step 2: quantise with llama-quantize
    quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
    if not os.path.isfile(quantize_bin):
        # fallback to older binary names
        for candidate in ("quantize", "llama-quantize"):
            candidate_path = os.path.join(llama_cpp_dir, candidate)
            if os.path.isfile(candidate_path):
                quantize_bin = candidate_path
                break
        else:
            print(
                "WARNING: llama-quantize binary not found — skipping quantisation.\n"
                f"f16 GGUF is available at: {gguf_f16}",
                file=sys.stderr,
            )
            return

    cmd_quant = [quantize_bin, gguf_f16, gguf_out, quant_type]
    print(f"\nRunning: {' '.join(cmd_quant)}")
    subprocess.run(cmd_quant, check=True)

    size_mb = os.path.getsize(gguf_out) / 1e6
    print(f"\nQuantised GGUF written: {gguf_out}  ({size_mb:.0f} MB)")

    # Clean up intermediate f16 file
    os.remove(gguf_f16)
    print(f"Removed intermediate: {gguf_f16}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model and export to GGUF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_model_id", required=True,
        help="HF model ID or local path for the base model "
             "(e.g. Qwen/Qwen2.5-Coder-3B-Instruct)",
    )
    parser.add_argument(
        "--adapter_dir", required=True,
        help="Path to the saved LoRA adapter directory "
             "(e.g. qwen2.5-coder-3b-lora-checkpoints/best_loss_adapter)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Where to write the merged HuggingFace model",
    )
    parser.add_argument(
        "--gguf_out", required=True,
        help="Destination path for the final .gguf file",
    )
    parser.add_argument(
        "--llama_cpp_dir", required=True,
        help="Root directory of a compiled llama.cpp checkout",
    )
    parser.add_argument(
        "--quant_type", default="Q8_0", choices=QUANT_TYPES,
        help="Quantisation type passed to llama-quantize",
    )
    parser.add_argument(
        "--skip_merge", action="store_true",
        help="Skip the merge step if the merged HF model already exists in --output_dir",
    )
    args = parser.parse_args()

    if not args.skip_merge:
        merge_and_save(args.base_model_id, args.adapter_dir, args.output_dir)
    else:
        print(f"--skip_merge set, using existing merged model at: {args.output_dir}")

    convert_to_gguf(
        llama_cpp_dir=args.llama_cpp_dir,
        merged_dir=args.output_dir,
        gguf_out=args.gguf_out,
        quant_type=args.quant_type,
    )


if __name__ == "__main__":
    main()
