import os
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
RAW_NL_PATH = "data/raw/all.nl"
RAW_CM_PATH = "data/raw/all.cm"
OUTPUT_DIR = "data/preprocessed"

TRAIN_RATIO = 0.80
TEST_RATIO = 0.20

SEED = 42


def prepare():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Vocab size: {tokenizer.vocab_size}. Using uint32 for storage.")

    print(f"\nReading raw data from:\n  {RAW_NL_PATH}\n  {RAW_CM_PATH}")
    with open(RAW_NL_PATH, "r", encoding="utf-8") as f:
        nl_lines = [line.rstrip("\n") for line in f]
    with open(RAW_CM_PATH, "r", encoding="utf-8") as f:
        cm_lines = [line.rstrip("\n") for line in f]

    assert len(nl_lines) == len(cm_lines), (
        f"Line count mismatch: {len(nl_lines)} NL vs {len(cm_lines)} CM"
    )
    total = len(nl_lines)
    print(f"Loaded {total} examples.")

    indices = list(range(total))
    random.seed(SEED)
    random.shuffle(indices)

    n_train = int(total * TRAIN_RATIO)

    splits = {
        "train": indices[:n_train],
        "test": indices[n_train:],
    }
    for name, idxs in splits.items():
        print(f"  {name}: {len(idxs)} examples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SYSTEM_PROMPT = "You are a helpful assistant that converts natural language instructions into shell commands. Output only the shell command, nothing else."

    for split_name, split_indices in splits.items():
        print(f"\nProcessing split: {split_name}")

        all_ids: list[int] = []
        metadata: list[list[int]] = []

        for idx in tqdm(split_indices, desc=f"Tokenizing {split_name}"):
            nl = nl_lines[idx].strip()
            cm = cm_lines[idx].strip()

            # for Qwen2.5 the assistant turn ends with <|im_end|>
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

        print(f"Saved {len(all_ids)} tokens to {bin_path}")
        print(f"Saved {len(metadata)} rows of metadata to {meta_path}")


if __name__ == "__main__":
    prepare()
