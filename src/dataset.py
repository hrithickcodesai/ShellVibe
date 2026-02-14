import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SFTDataset(Dataset):
    def __init__(self, data_path, meta_path):
        # load flat file for tokens
        self.data = np.memmap(data_path, dtype=np.uint32, mode="r")
        # load flat file for metadata, reshape it to have [total_length, prompt_length]
        self.meta = np.memmap(meta_path, dtype=np.uint32, mode="r").reshape(-1, 2)

        # extract sample and prompt length
        self.sample_lengths = self.meta[:, 0]
        self.prompt_lengths = self.meta[:, 1]

        # compute offset array
        self.offsets = np.concatenate(([0], np.cumsum(self.sample_lengths)))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        # get the raw token indices
        start_idx = int(self.offsets[idx])
        end_idx = int(self.offsets[idx + 1])

        # converting to pytorch tensor
        tokens = torch.from_numpy(self.data[start_idx:end_idx]).long()

        # creating targets
        labels = tokens.clone()
        prompt_len = self.prompt_lengths[idx]

        # masking the prompt tokens
        labels[:prompt_len] = -100

        return {"input_ids": tokens, "labels": labels}


class SFTCollator:
    def __init__(self, pad_token_id, ignore_index=-100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch):
        # extract all the input ids and labels
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        # pad the input ids
        padded_inputs = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )

        # pad the labels
        padded_labels = pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index
        )

        # 1 for real tokens and 0 for padding tokens
        attention_mask = (padded_inputs != self.pad_token_id).long()

        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "attention_mask": attention_mask,
        }
