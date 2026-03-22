import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


class _LogitsWrapper(nn.Module):
    """Strips KV-cache output so torch.onnx.export sees a fixed output shape."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    parser = argparse.ArgumentParser(description="Export ShellVibe model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the fine-tuned .pt checkpoint (trained with torch.compile).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_model",
        help="Directory to save the exported ONNX model (default: onnx_model).",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Export precision. fp16 requires CUDA (default: fp32).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    args = parser.parse_args()

    use_fp16 = args.dtype == "fp16"
    if use_fp16 and not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: fp16 export requires CUDA. Use --dtype fp32 on CPU/MPS."
        )

    dtype = torch.float16 if use_fp16 else torch.float32
    device = "cuda" if use_fp16 else "cpu"

    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading model ({args.dtype})...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)

    print(f"Loading fine-tuned checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # torch.compile wraps keys with "_orig_mod." — strip before loading
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()

    wrapper = _LogitsWrapper(model)

    # Dummy inputs on the target device
    dummy = tokenizer("Export dummy input", return_tensors="pt")
    input_ids = dummy["input_ids"].to(device)
    attention_mask = dummy["attention_mask"].to(device)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_file = output_path / "model.onnx"

    print(f"Exporting to ONNX (opset={args.opset}, dtype={args.dtype})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            f=str(onnx_file),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size", 1: "sequence"},
            },
            do_constant_folding=True,
            opset_version=args.opset,
        )

    tokenizer.save_pretrained(output_path)

    print(f"\nDone. ONNX model saved to: {onnx_file.resolve()}")
    print("Run inference with:")
    print(f"  python src/models/inference_onnx.py --onnx-dir {args.output_dir}")


if __name__ == "__main__":
    main()
