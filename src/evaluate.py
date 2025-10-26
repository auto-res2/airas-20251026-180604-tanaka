"""Independent evaluation & visualisation script.

Usage:
    uv run python -m src.evaluate \
        results_dir=/abs/path \
        run_ids='["run-1","run-2"]'
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
from scipy import stats
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Filesystem helpers -----------------------------------------------------------
# -----------------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Dict, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# -----------------------------------------------------------------------------
# Plotting helpers -------------------------------------------------------------
# -----------------------------------------------------------------------------

def plot_learning_curve(history: pd.DataFrame, run_id: str, out_dir: str) -> str:
    """Save learning-curve figure and return filepath."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["train/loss", "train/acc", "val/val_pairwise_accuracy"]:
        if col in history.columns:
            ax.plot(history["step"], history[col], label=col)
    ax.set_xlabel("Step")
    ax.set_title(f"Learning Curves – {run_id}")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_learning_curve.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrix(summary: Dict[str, float], run_id: str, out_dir: str) -> str:
    tp = summary.get("val_true_pos", summary.get("val_correct", 0))
    fp = summary.get("val_false_pos", 0)
    tn = summary.get("val_true_neg", 0)
    fn = summary.get("val_false_neg", 0)
    matrix = [[tp, fp], [fn, tn]]
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted + / -")
    ax.set_ylabel("Actual + / -")
    ax.set_title(f"Confusion Matrix – {run_id}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_confusion_matrix.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path

# -----------------------------------------------------------------------------
# Per-run processing -----------------------------------------------------------
# -----------------------------------------------------------------------------

def process_single_run(api: wandb.Api, entity: str, project: str, run_id: str, out_dir: str):
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history()  # pandas DataFrame
    summary = run.summary._json_dict
    config = dict(run.config)

    ensure_dir(out_dir)
    metrics_path = os.path.join(out_dir, "metrics.json")
    save_json({"summary": summary, "config": config}, metrics_path)

    figs: List[str] = []
    figs.append(plot_learning_curve(history, run_id, out_dir))
    figs.append(plot_confusion_matrix(summary, run_id, out_dir))
    return {
        "metrics_path": metrics_path,
        "figs": figs,
        "summary": summary,
        "config": config,
    }

# -----------------------------------------------------------------------------
# Aggregated analysis ----------------------------------------------------------
# -----------------------------------------------------------------------------

def derive_improvement_rates(baseline_runs: List[str], metric_values: Dict[str, float]):
    base_val = sum(metric_values[r] for r in baseline_runs) / len(baseline_runs)
    return {
        r: (v - base_val) / base_val if base_val != 0 else 0.0
        for r, v in metric_values.items()
        if r not in baseline_runs
    }


def group_by_seed(run_ids: List[str]):
    groups = defaultdict(list)
    for rid in run_ids:
        if "-seed" in rid:
            base = rid.split("-seed", 1)[0]
            groups[base].append(rid)
        else:
            groups[rid].append(rid)
    return groups


def aggregated_analysis(per_run: Dict[str, Dict], comp_dir: str):
    ensure_dir(comp_dir)

    metric_name = "best_val_pairwise_accuracy"
    metric_values = {rid: d["summary"].get(metric_name, 0.0) for rid, d in per_run.items()}
    save_json(metric_values, os.path.join(comp_dir, "aggregated_metrics.json"))

    # Baseline: first run as reference (could be replaced by explicit selection)
    baseline_runs = [sorted(metric_values.keys())[0]]
    improvements = derive_improvement_rates(baseline_runs, metric_values)
    save_json(
        {"baseline": baseline_runs, "improvement_rate": improvements},
        os.path.join(comp_dir, "derived_metrics.json"),
    )

    # Statistical test across seeds -----------------------------------
    grouped = group_by_seed(list(per_run.keys()))
    baseline_group = [g for g in grouped if g in baseline_runs[0]][0]
    p_values = {}
    for g, rs in grouped.items():
        if g == baseline_group:
            continue
        vals_baseline = [metric_values[r] for r in grouped[baseline_group]]
        vals_other = [metric_values[r] for r in rs]
        if len(vals_baseline) >= 2 and len(vals_other) >= 2:
            t_stat, p_val = stats.ttest_ind(vals_baseline, vals_other, equal_var=False)
            p_values[g] = p_val
    save_json(p_values, os.path.join(comp_dir, "p_values.json"))

    # Bar chart (matplotlib with error bars, avoids seaborn yerr removal)
    labels, means, stds = [], [], []
    for g, rs in grouped.items():
        vals = [metric_values[r] for r in rs]
        labels.append(g)
        means.append(sum(vals) / len(vals))
        stds.append(pd.Series(vals).std())

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.6), 5))
    colors = sns.color_palette("mako", len(labels))
    positions = range(len(labels))
    ax.bar(positions, means, yerr=stds, capsize=4, color=colors, alpha=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    for idx, m in enumerate(means):
        ax.text(idx, m + 0.002, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(metric_name)
    ax.set_title("Cross-run Comparison (mean ± s.d.)")
    fig.tight_layout()
    bar_path = os.path.join(comp_dir, "comparison_accuracy_bar_chart.pdf")
    fig.savefig(bar_path)
    plt.close(fig)

    return {
        "aggregated_metrics": os.path.join(comp_dir, "aggregated_metrics.json"),
        "derived_metrics": os.path.join(comp_dir, "derived_metrics.json"),
        "p_values": os.path.join(comp_dir, "p_values.json"),
        "figs": [bar_path],
    }

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="Directory to save outputs")
    parser.add_argument(
        "run_ids",
        type=str,
        help="JSON string list of run IDs (e.g. '[\"run-1\",\"run-2\"]')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir
    run_ids = json.loads(args.run_ids)

    # Load global WandB config to identify entity/project -------------
    import yaml

    with open(os.path.join("config", "config.yaml")) as f:
        cfg_global = yaml.safe_load(f)
    entity = cfg_global["wandb"]["entity"]
    project = cfg_global["wandb"]["project"]

    api = wandb.Api()
    per_run_outputs: Dict[str, Dict] = {}
    for rid in tqdm(run_ids, desc="Processing runs"):
        out_dir = ensure_dir(os.path.join(results_dir, rid))
        per_run_outputs[rid] = process_single_run(api, entity, project, rid, out_dir)
        print(f"Processed {rid} – outputs stored in {out_dir}")
        for fp in per_run_outputs[rid]["figs"]:
            print(fp)

    comp_dir = os.path.join(results_dir, "comparison")
    comp_outputs = aggregated_analysis(per_run_outputs, comp_dir)
    print("Aggregated analysis saved:")
    for k, v in comp_outputs.items():
        if isinstance(v, list):
            for fp in v:
                print(fp)
        else:
            print(v)


if __name__ == "__main__":
    main()