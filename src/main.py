import subprocess
import sys
import os
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
    
    # Get the project root directory (where main.py's parent is)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for seed in seed_list:
        run_id_seed = f"{cfg.run.run_id}-seed{seed}"
        # Extract the run config name from the defaults or use the run_id
        run_config_name = cfg.run.run_id.rsplit('-seed', 1)[0] if 'seed' in cfg.run.run_id else cfg.run.run_id
        overrides = [
            f"run={run_config_name}",
            f"run.run_id={run_id_seed}",
            f"training.seed={seed}",
            f"results_dir={cfg.results_dir}",
            f"wandb.mode={cfg.wandb.mode}",
            f"mode={cfg.mode}",
        ]
        cmd = [sys.executable, "-u", "-m", "src.train"] + overrides
        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=project_root)

if __name__ == "__main__":
    main()