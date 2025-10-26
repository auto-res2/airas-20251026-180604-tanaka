"""Data loading & preprocessing pipeline."""
from typing import List, Tuple

import datasets
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Dataset wrappers -------------------------------------------------------------
# -----------------------------------------------------------------------------

class PairPreferenceDataset(torch.utils.data.Dataset):
    """Wrap HF dataset that contains (prompt, positive, negative) triples."""

    def __init__(self, hf_ds: datasets.Dataset):
        self.hf_ds = hf_ds

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        sample = self.hf_ds[idx]
        prompt = (
            sample.get("prompt")
            or sample.get("instruction")
            or sample.get("question")
            or ""
        )
        # Identify pos/neg fields --------------------------------------
        if {"completion_a", "completion_b", "choice"}.issubset(sample.keys()):
            pos, neg = (
                (sample["completion_a"], sample["completion_b"])
                if sample["choice"] == 0
                else (sample["completion_b"], sample["completion_a"])
            )
        elif {"chosen", "rejected"}.issubset(sample.keys()):
            pos, neg = sample["chosen"], sample["rejected"]
        else:
            raise KeyError("Dataset sample lacks recognised preference fields.")
        return {"prompt": prompt, "pos": pos, "neg": neg}


class PreferenceCollator:
    """Tokenises preference pairs; returns dict suitable for model input."""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos = tokenizer.eos_token or "</s>"

    def _build(self, prompts: List[str], responses: List[str]):
        joined = [p + self.eos + r + self.eos for p, r in zip(prompts, responses)]
        toks = self.tokenizer(
            joined,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # response mask: 1 for response tokens, 0 for prompt tokens -----
        mask = torch.zeros_like(toks["input_ids"], dtype=torch.bool)
        for i, p in enumerate(prompts):
            prompt_len = len(self.tokenizer(p + self.eos)["input_ids"])
            seq_len = toks["attention_mask"][i].sum().item()
            mask[i, prompt_len:seq_len] = 1
        toks["response_mask"] = mask
        return toks

    def __call__(self, batch: List[dict]):
        prompts = [b["prompt"] for b in batch]
        pos_resp = [b["pos"] for b in batch]
        neg_resp = [b["neg"] for b in batch]
        return {"pos": self._build(prompts, pos_resp), "neg": self._build(prompts, neg_resp)}

# -----------------------------------------------------------------------------
# Public API -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def prepare_datasets(cfg) -> Tuple[AutoTokenizer, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load dataset, create train/val splits, return tokenizer + datasets."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=".cache/", use_fast=True)

    raw_ds = load_dataset(cfg.dataset.name, cache_dir=".cache/")
    if "train" in raw_ds and "validation" in raw_ds:
        train_raw, val_raw = raw_ds["train"], raw_ds["validation"]
    else:
        split = raw_ds["train"].train_test_split(test_size=cfg.dataset.split.val)
        train_raw, val_raw = split["train"], split["test"]

    return tokenizer, PairPreferenceDataset(train_raw), PairPreferenceDataset(val_raw)