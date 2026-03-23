import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import argparse
import os


def export(checkpoint_path: str, output_dir: str, model_id: str):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Building model from config...")
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_config(config).to(torch.float32)

    print(f"Loading fine-tuned weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving HF model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done. Directory contents:")
    for f in os.listdir(output_dir):
        size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1e6
        print(f"  {f:40s}  {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model_id",
        required=True,
        help="HuggingFace model ID matching the checkpoint architecture",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir, args.model_id)
