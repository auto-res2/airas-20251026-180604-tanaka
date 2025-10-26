"""Model loading utilities."""
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

__all__ = ["load_models", "compute_logps"]

# -----------------------------------------------------------------------------
# Loading QLoRA model & frozen reference ---------------------------------------
# -----------------------------------------------------------------------------

def load_models(cfg, tokenizer: AutoTokenizer) -> Tuple[torch.nn.Module, torch.nn.Module]:
    bnb_cfg = None
    if cfg.model.get("load_in_4bit", False):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        cache_dir=".cache/",
        torch_dtype=getattr(torch, cfg.model.torch_dtype),
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()

    if cfg.model.get("peft"):
        lora_cfg = LoraConfig(
            r=cfg.model.peft.lora_r,
            lora_alpha=cfg.model.peft.lora_alpha,
            lora_dropout=cfg.model.peft.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # Frozen reference -------------------------------------------------
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        cache_dir=".cache/",
        torch_dtype=getattr(torch, cfg.model.torch_dtype),
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    return model, ref_model

# -----------------------------------------------------------------------------
# Log-probability helpers ------------------------------------------------------
# -----------------------------------------------------------------------------

def _gather_log_probs(logits: torch.Tensor, input_ids: torch.Tensor):
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    logp = logp[:, :-1, :]
    target = input_ids[:, 1:]
    # Clamp target indices to valid range [0, vocab_size-1] to avoid index errors
    vocab_size = logp.size(-1)
    target = torch.clamp(target, 0, vocab_size - 1)
    return torch.gather(logp, 2, target.unsqueeze(-1)).squeeze(-1)


def _forward(model, part_batch, device):
    input_ids = part_batch["input_ids"].to(device)
    attention_mask = part_batch["attention_mask"].to(device)
    response_mask = part_batch["response_mask"].to(device)[:, 1:]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    token_logp = _gather_log_probs(outputs.logits, input_ids)
    # Add small epsilon to avoid division by zero
    response_sum = response_mask.sum(dim=1).clamp(min=1e-8)
    seq_logp = (token_logp * response_mask).sum(dim=1) / response_sum
    return seq_logp


def compute_logps(model, ref_model, batch, device):
    pos, neg = batch["pos"], batch["neg"]
    with torch.no_grad():
        logp_pos_ref = _forward(ref_model, pos, device)
        logp_neg_ref = _forward(ref_model, neg, device)
    logp_pos = _forward(model, pos, device)
    logp_neg = _forward(model, neg, device)
    return logp_pos, logp_neg, logp_pos_ref, logp_neg_ref