import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_ID = "google/gemma-3-270m"
DATASET_ID = "westenfelder/NL2SH-ALFA"
OUTPUT_DIR = "data"


def prepare():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Vocab size: {tokenizer.vocab_size}. Using uint32 for storage.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    splits = ["train", "test"]
    for split_name in splits:
        print(f"\nProcessing split: {split_name}")
        curr_ds = load_dataset(DATASET_ID, split_name)
        print(f"Loaded {len(curr_ds)} examples from {split_name} split.")
        print(curr_ds)

        all_ids = []
        metadata = []

        BOS = tokenizer.bos_token
        EOS = tokenizer.eos_token

        # the dataset using train for both
        for row in tqdm(curr_ds["train"], desc=f"Tokenizing {split_name}"):
            # concatenate the prompt and completion
            prompt = f"{BOS}instruction: {row['nl'].lower().strip()}\ncommand: "
            completion = f"{row['bash'].strip()}{EOS}"

            # we do not need special token as we have added it manually
            p_ids = tokenizer.encode(prompt, add_special_tokens=False)
            c_ids = tokenizer.encode(completion, add_special_tokens=False)

            full_ids = p_ids + c_ids

            # just keep it packed, we can unpack it using metadata later
            all_ids.extend(full_ids)

            # Store metadata: [total_length, prompt_length]
            # specific for SF where we mask the prompt loss
            metadata.append([len(full_ids), len(p_ids)])

        ids_arr = np.array(all_ids, dtype=np.uint32)
        bin_path = os.path.join(OUTPUT_DIR, f"{split_name}.bin")
        ids_arr.tofile(bin_path)

        meta_arr = np.array(metadata, dtype=np.uint32)
        meta_path = os.path.join(OUTPUT_DIR, f"{split_name}_meta.bin")
        meta_arr.tofile(meta_path)

        print(f"Saved {len(all_ids)} tokens to {bin_path}")
        print(f"Saved {len(metadata)} rows of metadata to {meta_path}\n")


if __name__ == "__main__":
    prepare()
