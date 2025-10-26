import subprocess
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    # ---------------- Mode-specific tweaks -----------------------------------
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

    # ---------------- Spawn train.py for each seed ---------------------------
    seed_list = cfg.training.get("seed_list", [cfg.training.get("seed", 42)])
    for seed in seed_list:
        run_id_seed = f"{cfg.run.run_id}-seed{seed}"
        overrides = [
            f"run.run_id={run_id_seed}",
            f"training.seed={seed}",
            f"results_dir={cfg.results_dir}",
            f"wandb.mode={cfg.wandb.mode}",
            f"mode={cfg.mode}",
        ]
        cmd = ["python", "-u", "-m", "src.train"] + overrides
        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()