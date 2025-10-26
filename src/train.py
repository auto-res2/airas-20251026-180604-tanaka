import os
import math
import random
from typing import Dict, Any, Optional, Iterable

import hydra
import torch
import wandb
import optuna
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm

from src.preprocess import prepare_datasets, PreferenceCollator
from src.model import load_models, compute_logps

# -----------------------------------------------------------------------------
# Loss functions ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def dpo_loss(
    logp_pos: torch.Tensor,
    logp_neg: torch.Tensor,
    logp_pos_ref: torch.Tensor,
    logp_neg_ref: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Standard DPO loss."""
    preference_term = -torch.log(torch.sigmoid(logp_pos - logp_neg)).mean()
    kl = 0.5 * (((logp_pos - logp_pos_ref) ** 2) + ((logp_neg - logp_neg_ref) ** 2)).mean()
    return preference_term + beta * kl


def td_dpo_loss(
    logp_pos: torch.Tensor,
    logp_neg: torch.Tensor,
    logp_pos_ref: torch.Tensor,
    logp_neg_ref: torch.Tensor,
    beta: float,
    tau: float,
) -> torch.Tensor:
    """Temperature-Decoupled DPO loss (proposed)."""
    preference_term = -torch.log(torch.sigmoid((logp_pos - logp_neg) / tau)).mean()
    kl = 0.5 * (((logp_pos - logp_pos_ref) ** 2) + ((logp_neg - logp_neg_ref) ** 2)).mean()
    return preference_term + beta * kl

# -----------------------------------------------------------------------------
# Helper utilities -------------------------------------------------------------
# -----------------------------------------------------------------------------

def limited_iter(iterable: Iterable, max_batches: Optional[int]):
    """Yield at most `max_batches` from iterable (all if None)."""
    if max_batches is None:
        yield from iterable
    else:
        for idx, item in enumerate(iterable):
            if idx >= max_batches:
                break
            yield item


def evaluate(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Validation loop – returns dictionary of metrics."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in limited_iter(loader, max_batches):
            logp_pos, logp_neg, *_ = compute_logps(model, ref_model, batch, device)
            correct += (logp_pos > logp_neg).sum().item()
            total += logp_pos.size(0)
    model.train()
    return {"val_pairwise_accuracy": correct / total if total > 0 else 0.0}

# -----------------------------------------------------------------------------
# Core training routine --------------------------------------------------------
# -----------------------------------------------------------------------------

def _safe_seed(cfg):
    """Ensure cfg.training.seed exists – fall back gracefully if absent."""
    if hasattr(cfg.training, "seed"):
        return cfg.training.seed
    # Derive from seed_list if present otherwise fixed default
    if hasattr(cfg.training, "seed_list") and len(cfg.training.seed_list) > 0:
        seed = cfg.training.seed_list[0]
    else:
        seed = 42
    cfg.training.seed = seed
    return seed


def run_training(cfg, trial: Optional[optuna.Trial] = None) -> float:
    # ------------------------------------------------------------------
    # Reproducibility ---------------------------------------------------
    # ------------------------------------------------------------------
    seed = _safe_seed(cfg)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ------------------------------------------------------------------
    # Data  -------------------------------------------------------------
    # ------------------------------------------------------------------
    tokenizer, train_ds, val_ds = prepare_datasets(cfg)

    # Shorten datasets in trial-mode to 1-2 batches for speed
    if cfg.mode == "trial":
        subset_size = cfg.training.per_device_train_batch_size * 2
        train_ds = torch.utils.data.Subset(train_ds, range(min(subset_size, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(subset_size, len(val_ds))))

    # ------------------------------------------------------------------
    # Model -------------------------------------------------------------
    # ------------------------------------------------------------------
    model, ref_model = load_models(cfg, tokenizer)

    # ------------------------------------------------------------------
    # Dataloaders -------------------------------------------------------
    # ------------------------------------------------------------------
    collator = PreferenceCollator(tokenizer, max_length=cfg.dataset.preprocessing.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.per_device_train_batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
    )

    # ------------------------------------------------------------------
    # Optimiser & scheduler --------------------------------------------
    # ------------------------------------------------------------------
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.training.learning_rate)

    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / cfg.training.gradient_accumulation_steps
    )
    max_steps = cfg.training.epochs * num_update_steps_per_epoch

    if cfg.training.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, cfg.training.warmup_steps, max_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, cfg.training.warmup_steps, max_steps
        )

    # ------------------------------------------------------------------
    # WandB -------------------------------------------------------------
    # ------------------------------------------------------------------
    if cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        print(f"[WandB] Run URL: {wandb_run.url}")
    else:
        wandb_run = None
        os.environ["WANDB_DISABLED"] = "true"

    # ------------------------------------------------------------------
    # Device placement --------------------------------------------------
    # ------------------------------------------------------------------
    # Models are already on device via device_map in load_models()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_model.eval()

    # ------------------------------------------------------------------
    # Training loop -----------------------------------------------------
    # ------------------------------------------------------------------
    best_val_acc, global_step = 0.0, 0
    max_train_batches = 2 if cfg.mode == "trial" else None
    max_val_batches = 2 if cfg.mode == "trial" else None

    for epoch in range(cfg.training.epochs):
        model.train()
        running_correct, running_total = 0, 0
        iterable = limited_iter(train_loader, max_train_batches)
        pbar = tqdm(iterable, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}")
        for step, batch in enumerate(pbar, start=1):
            logp_pos, logp_neg, logp_pos_ref, logp_neg_ref = compute_logps(
                model, ref_model, batch, device
            )

            # Compute loss -------------------------------------------------------------
            if cfg.loss.name == "td_dpo_loss":
                loss = td_dpo_loss(
                    logp_pos,
                    logp_neg,
                    logp_pos_ref,
                    logp_neg_ref,
                    beta=float(cfg.loss.beta),
                    tau=float(cfg.loss.tau),
                )
            else:
                loss = dpo_loss(
                    logp_pos,
                    logp_neg,
                    logp_pos_ref,
                    logp_neg_ref,
                    beta=float(cfg.loss.beta),
                )
            loss = loss / cfg.training.gradient_accumulation_steps
            loss.backward()

            running_correct += (logp_pos > logp_neg).sum().item()
            running_total += logp_pos.size(0)

            if step % cfg.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # ----- logging ---------------------------------------------------------
                if wandb_run and global_step % cfg.training.logging_steps == 0:
                    wandb.log(
                        {
                            "train/loss": loss.item() * cfg.training.gradient_accumulation_steps,
                            "train/acc": running_correct / running_total,
                            "lr": scheduler.get_last_lr()[0],
                            "step": global_step,
                        },
                        step=global_step,
                    )

                # ----- evaluation -------------------------------------------------------
                if (
                    cfg.training.evaluation_strategy == "steps"
                    and global_step % cfg.training.eval_steps == 0
                ):
                    val_metrics = evaluate(
                        model, ref_model, val_loader, device, max_val_batches
                    )
                    if wandb_run:
                        wandb.log(
                            {f"val/{k}": v for k, v in val_metrics.items()},
                            step=global_step,
                        )
                    if val_metrics["val_pairwise_accuracy"] > best_val_acc:
                        best_val_acc = val_metrics["val_pairwise_accuracy"]
                        save_best(model, tokenizer, cfg)
                    model.train()

        # End-of-epoch evaluation -------------------------------------------------------
        val_metrics = evaluate(model, ref_model, val_loader, device, max_val_batches)
        if wandb_run:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
        if val_metrics["val_pairwise_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_pairwise_accuracy"]
            save_best(model, tokenizer, cfg)

    # ------------------------------------------------------------------
    # Finalise ----------------------------------------------------------
    # ------------------------------------------------------------------
    if wandb_run:
        wandb_run.summary["best_val_pairwise_accuracy"] = best_val_acc
        wandb_run.finish()

    return best_val_acc

# -----------------------------------------------------------------------------
# Misc utilities ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def save_best(model, tokenizer, cfg):
    save_path = os.path.join(cfg.results_dir, cfg.run_id, "best")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

# -----------------------------------------------------------------------------
# Optuna objective -------------------------------------------------------------
# -----------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, base_cfg):
    """Objective wrapper for Optuna hyper-parameter search."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))  # deep copy

    # Sample parameters according to search space ----------------------
    for hp_name, hp_space in cfg.optuna.search_space.items():
        if hp_space.type == "categorical":
            sampled = trial.suggest_categorical(hp_name, hp_space.choices)
        elif hp_space.type == "loguniform":
            sampled = trial.suggest_float(hp_name, hp_space.low, hp_space.high, log=True)
        else:
            raise ValueError(f"Unsupported search-space type: {hp_space.type}")

        target_field = f"loss.{hp_name}" if hp_name in ["beta", "tau"] else f"training.{hp_name}"
        OmegaConf.update(cfg, target_field, sampled, merge=False)

    cfg.wandb.mode = "disabled"  # never log individual trials
    val_acc = run_training(cfg, trial)
    return -val_acc  # Optuna minimises by default

# -----------------------------------------------------------------------------
# CLI entry-point --------------------------------------------------------------
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(os.path.join(cfg.results_dir, cfg.run_id), exist_ok=True)

    # ---------------- Mode tweaks -------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        OmegaConf.set_struct(cfg, False)
        cfg.training.epochs = 1
        OmegaConf.set_struct(cfg, True)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # ---------------- Hyper-parameter search --------------------------
    if cfg.optuna.n_trials and cfg.optuna.n_trials > 0:
        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(lambda t: optuna_objective(t, cfg), n_trials=cfg.optuna.n_trials)
        print("[Optuna] Best params:", study.best_params)
        for k, v in study.best_params.items():
            target_field = f"loss.{k}" if k in ["beta", "tau"] else f"training.{k}"
            OmegaConf.update(cfg, target_field, v, merge=False)

    # ---------------- Final training ----------------------------------
    run_training(cfg)


if __name__ == "__main__":
    main()