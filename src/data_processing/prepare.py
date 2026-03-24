import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
OUTPUT_DIR = "data/preprocessed"

SPLITS = {
    "train": "data/preprocessed/train.csv",
    "test": "data/preprocessed/test.csv",
}

SYSTEM_PROMPT = "You are a helpful assistant that converts natural language instructions into shell commands. Output only the shell command, nothing else."


def prepare():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Vocab size: {tokenizer.vocab_size}. Using uint32 for storage.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, csv_path in SPLITS.items():
        print(f"\nReading {split_name} data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} examples.")

        all_ids: list[int] = []
        metadata: list[list[int]] = []

        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Tokenizing {split_name}"
        ):
            nl = str(row["input_nl"]).strip()
            cm = str(row["output_command"]).strip()

            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": nl},
            ]
            full_messages = prompt_messages + [{"role": "assistant", "content": cm}]

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            full_text = tokenizer.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                tokenize=False,
            )

            p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids_enc = tokenizer.encode(full_text, add_special_tokens=False)
            c_ids = full_ids_enc[len(p_ids) :]

            full_ids = p_ids + c_ids

            all_ids.extend(full_ids)
            metadata.append([len(full_ids), len(p_ids)])

        ids_arr = np.array(all_ids, dtype=np.uint32)
        bin_path = os.path.join(OUTPUT_DIR, f"{split_name}.bin")
        ids_arr.tofile(bin_path)

        meta_arr = np.array(metadata, dtype=np.uint32)
        meta_path = os.path.join(OUTPUT_DIR, f"{split_name}_meta.bin")
        meta_arr.tofile(meta_path)

        lengths = [m[0] for m in metadata]
        lengths_arr = np.array(lengths)
        print(f"\n  [{split_name}] sequence length stats (tokens):")
        print(f"    count  : {len(lengths_arr)}")
        print(f"    min    : {lengths_arr.min()}")
        print(f"    max    : {lengths_arr.max()}")
        print(f"    mean   : {lengths_arr.mean():.1f}")
        print(f"    median : {np.median(lengths_arr):.1f}")
        print(f"    p90    : {np.percentile(lengths_arr, 90):.1f}")
        print(f"    p95    : {np.percentile(lengths_arr, 95):.1f}")
        print(f"    p99    : {np.percentile(lengths_arr, 99):.1f}")

        print(f"\nSaved {len(all_ids)} tokens to {bin_path}")
        print(f"Saved {len(metadata)} rows of metadata to {meta_path}")


if __name__ == "__main__":
    prepare()
