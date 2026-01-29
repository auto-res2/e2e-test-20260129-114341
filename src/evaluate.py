from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix


PRIMARY_METRIC = "accuracy"


sns.set_theme(style="whitegrid")


def parse_kv_args(argv: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed[key.lstrip("-")] = value
        elif arg.startswith("--"):
            key = arg.lstrip("-")
            if idx + 1 < len(argv) and "=" not in argv[idx + 1] and not argv[idx + 1].startswith("--"):
                parsed[key] = argv[idx + 1]
                idx += 1
            else:
                parsed[key] = "true"
        idx += 1
    return parsed


def load_wandb_config() -> Dict[str, str]:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    return {"entity": cfg.wandb.entity, "project": cfg.wandb.project}


def to_json_friendly(obj):
    if isinstance(obj, dict):
        return {k: to_json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_friendly(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_json_friendly(payload), f, indent=2)
    print(str(path))


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(sa.mean() - sb.mean())
    diffs = np.array(diffs)
    ci = np.percentile(diffs, [2.5, 97.5])
    p_value = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return float(ci[0]), float(ci[1]), p_value


def plot_learning_curve(history: pd.DataFrame, run_id: str, out_dir: Path) -> List[Path]:
    paths: List[Path] = []
    if "running_accuracy" in history.columns:
        y = history["running_accuracy"].dropna().values
    elif "correct" in history.columns:
        correct = history["correct"].dropna().astype(float).values
        if len(correct) == 0:
            return paths
        y = np.cumsum(correct) / (np.arange(len(correct)) + 1)
    else:
        return paths

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(y)), y, label="running_accuracy")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curve: {run_id}")
    ax.legend()
    if len(y) > 0:
        ax.scatter([len(y) - 1], [y[-1]], color="red")
        ax.annotate(f"{y[-1]:.3f}", (len(y) - 1, y[-1]))
    fig.tight_layout()
    out_path = out_dir / f"{run_id}_learning_curve.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(str(out_path))
    paths.append(out_path)
    return paths


def plot_confusion_matrix(history: pd.DataFrame, run_id: str, out_dir: Path) -> List[Path]:
    paths: List[Path] = []
    if "correct" not in history.columns:
        return paths
    if "final_consistent" in history.columns:
        y_pred = history["final_consistent"].fillna(0).astype(int).values
        pred_label = "Final Consistent"
    elif "faithful_pass_rate" in history.columns:
        y_pred = (history["faithful_pass_rate"].fillna(0) >= 0.5).astype(int).values
        pred_label = "Faithful Pass >= 0.5"
    else:
        return paths

    y_true = history["correct"].fillna(0).astype(int).values
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return paths
    cm = confusion_matrix(y_true[:n], y_pred[:n], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel(f"{pred_label} (Pred)")
    ax.set_ylabel("Correct (Gold)")
    ax.set_title(f"Verifier vs Correct: {run_id}")
    ax.set_xticklabels(["incorrect", "correct"])
    ax.set_yticklabels(["incorrect", "correct"], rotation=0)
    fig.tight_layout()
    out_path = out_dir / f"{run_id}_confusion_matrix_verifier_vs_correct.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(str(out_path))
    paths.append(out_path)
    return paths


def plot_parse_outcomes(history: pd.DataFrame, run_id: str, out_dir: Path) -> List[Path]:
    paths: List[Path] = []
    if "audit_parse_variant_a" not in history.columns or "audit_parse_variant_b" not in history.columns:
        return paths
    a = history["audit_parse_variant_a"].fillna(0).astype(int).values
    b = history["audit_parse_variant_b"].fillna(0).astype(int).values
    labels = []
    for aa, bb in zip(a, b):
        if aa and bb:
            labels.append("both")
        elif aa:
            labels.append("only_a")
        elif bb:
            labels.append("only_b")
        else:
            labels.append("none")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=pd.Series(labels), ax=ax)
    ax.set_title(f"Audit Parse Outcomes: {run_id}")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Count")
    fig.tight_layout()
    out_path = out_dir / f"{run_id}_audit_parse_outcomes.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(str(out_path))
    paths.append(out_path)
    return paths


def plot_distribution(history: pd.DataFrame, run_id: str, out_dir: Path, metric: str, title: str) -> List[Path]:
    paths: List[Path] = []
    if metric not in history.columns:
        return paths
    values = history[metric].dropna().values
    if len(values) == 0:
        return paths
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(values, bins=20, ax=ax)
    ax.set_title(f"{title}: {run_id}")
    ax.set_xlabel(metric)
    fig.tight_layout()
    out_path = out_dir / f"{run_id}_{metric}_hist.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(str(out_path))
    paths.append(out_path)
    return paths


def plot_scatter(history: pd.DataFrame, run_id: str, out_dir: Path) -> List[Path]:
    paths: List[Path] = []
    if "delta_answer" not in history.columns or "faithful_pass_rate" not in history.columns:
        return paths
    x = history["delta_answer"].dropna().values
    y = history["faithful_pass_rate"].dropna().values
    if len(x) == 0 or len(y) == 0:
        return paths
    n = min(len(x), len(y))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x[:n], y[:n], alpha=0.6)
    ax.set_xlabel("delta_answer")
    ax.set_ylabel("faithful_pass_rate")
    ax.set_title(f"Δ_answer vs Faithful Pass: {run_id}")
    fig.tight_layout()
    out_path = out_dir / f"{run_id}_delta_vs_faithful_scatter.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(str(out_path))
    paths.append(out_path)
    return paths


def comparison_bar(metrics: Dict[str, Dict[str, float]], metric_name: str, out_dir: Path) -> Optional[Path]:
    runs = list(metrics.get(metric_name, {}).keys())
    if not runs:
        return None
    values = [metrics[metric_name][r] for r in runs]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(runs, values)
    ax.set_title(f"Comparison: {metric_name}")
    ax.set_ylabel(metric_name)
    ax.set_xticklabels(runs, rotation=30, ha="right")
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    fig.tight_layout()
    out_path = out_dir / f"comparison_{metric_name}_bar_chart.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(str(out_path))
    return out_path


def comparison_boxplot(per_run_histories: Dict[str, pd.DataFrame], metric: str, out_dir: Path) -> Optional[Path]:
    rows = []
    for run_id, history in per_run_histories.items():
        if metric not in history.columns:
            continue
        vals = history[metric].dropna().values
        for v in vals:
            rows.append({"run_id": run_id, metric: v})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="run_id", y=metric, ax=ax)
    ax.set_title(f"Comparison Boxplot: {metric}")
    ax.set_xticklabels(df["run_id"].unique(), rotation=30, ha="right")
    fig.tight_layout()
    out_path = out_dir / f"comparison_{metric}_boxplot.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(str(out_path))
    return out_path


def comparison_table(metrics: Dict[str, Dict[str, float]], out_dir: Path) -> Optional[Path]:
    if not metrics:
        return None
    df = pd.DataFrame(metrics)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 1 + len(df.columns)), 0.6 * len(df) + 2))
    ax.axis("off")
    table = ax.table(
        cellText=df.round(4).values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.tight_layout()
    out_path = out_dir / "comparison_metrics_table.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(str(out_path))
    return out_path


def comparison_scatter(metrics: Dict[str, Dict[str, float]], x_metric: str, y_metric: str, out_dir: Path) -> Optional[Path]:
    if x_metric not in metrics or y_metric not in metrics:
        return None
    if not metrics[x_metric] or not metrics[y_metric]:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for run_id in metrics[x_metric]:
        ax.scatter(
            metrics[x_metric].get(run_id, 0.0),
            metrics[y_metric].get(run_id, 0.0),
            label=run_id,
        )
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"{y_metric} vs {x_metric}")
    ax.legend(fontsize=6)
    fig.tight_layout()
    out_path = out_dir / f"comparison_{y_metric}_vs_{x_metric}_scatter.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(str(out_path))
    return out_path


def plot_flip_matrix(
    baseline_correct: np.ndarray,
    proposed_correct: np.ndarray,
    baseline_id: str,
    proposed_id: str,
    out_dir: Path,
) -> Optional[Path]:
    n = min(len(baseline_correct), len(proposed_correct))
    if n == 0:
        return None
    matrix = np.zeros((2, 2), dtype=int)
    for b, p in zip(baseline_correct[:n], proposed_correct[:n]):
        matrix[int(b), int(p)] += 1
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Proposed Correct")
    ax.set_ylabel("Baseline Correct")
    ax.set_title(f"Flip Matrix: {baseline_id} vs {proposed_id}")
    ax.set_xticklabels(["incorrect", "correct"])
    ax.set_yticklabels(["incorrect", "correct"], rotation=0)
    fig.tight_layout()
    out_path = out_dir / f"comparison_flip_matrix_{baseline_id}_vs_{proposed_id}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(str(out_path))
    return out_path


def extract_metric_value(metric_name: str, summary: Dict, history: pd.DataFrame) -> Optional[float]:
    if summary and metric_name in summary:
        return float(summary[metric_name])
    if metric_name == "accuracy" and "correct" in history.columns:
        return float(history["correct"].dropna().mean())
    if metric_name == "audit_parse_rate" and "audit_parse_success" in history.columns:
        return float(history["audit_parse_success"].dropna().mean())
    if metric_name == "faithful_pass_rate_mean" and "faithful_pass_rate" in history.columns:
        return float(history["faithful_pass_rate"].dropna().mean())
    if metric_name == "audited_but_unfaithful_rate" and "audited_but_unfaithful" in history.columns:
        return float(history["audited_but_unfaithful"].dropna().mean())
    if metric_name == "mean_extra_generation_calls_per_question" and "extra_generation_calls" in history.columns:
        return float(history["extra_generation_calls"].dropna().mean())
    return None


def is_minimize_metric(metric_name: str) -> bool:
    lowered = metric_name.lower()
    return any(token in lowered for token in ["loss", "perplexity", "error"])


def is_trial_or_disabled(config: Dict[str, object]) -> bool:
    mode = config.get("mode")
    wandb_mode = None
    if isinstance(config.get("wandb"), dict):
        wandb_mode = config["wandb"].get("mode")
    wandb_mode = wandb_mode or config.get("wandb.mode")
    return mode == "trial" or wandb_mode == "disabled"


def fetch_history(run: wandb.apis.public.Run) -> pd.DataFrame:
    history = run.history()
    if history is None:
        return pd.DataFrame()
    if "_step" in history.columns:
        history = history.sort_values("_step")
    return history


def main() -> None:
    args = parse_kv_args(sys.argv[1:])
    if "results_dir" not in args or "run_ids" not in args:
        raise ValueError("Usage: python -m src.evaluate results_dir=... run_ids='[...]'")

    run_ids = json.loads(args["run_ids"])
    if not run_ids:
        raise ValueError("run_ids must be a non-empty JSON list.")

    wandb_cfg = load_wandb_config()
    api = wandb.Api()

    results_dir = Path(args["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    per_run_summaries: Dict[str, Dict[str, float]] = {}
    per_run_histories: Dict[str, pd.DataFrame] = {}
    per_run_correct: Dict[str, np.ndarray] = {}

    for run_id in run_ids:
        try:
            run = api.run(f"{wandb_cfg['entity']}/{wandb_cfg['project']}/{run_id}")
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch run_id={run_id} from WandB: {exc}") from exc

        config = dict(run.config)
        if is_trial_or_disabled(config):
            raise RuntimeError(
                f"Run_id={run_id} is marked as trial/disabled. Evaluation must run on full WandB runs only."
            )

        history = fetch_history(run)
        summary = run.summary._json_dict

        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        history_payload = history.astype(object).where(pd.notnull(history), None).to_dict(orient="list")
        metrics_payload = {
            "run_id": run_id,
            "summary": summary,
            "config": config,
            "history": history_payload,
        }
        save_json(run_dir / "metrics.json", metrics_payload)

        per_run_summaries[run_id] = summary
        per_run_histories[run_id] = history
        if "correct" in history.columns:
            per_run_correct[run_id] = history["correct"].dropna().astype(int).values

        plot_learning_curve(history, run_id, run_dir)
        plot_confusion_matrix(history, run_id, run_dir)
        plot_parse_outcomes(history, run_id, run_dir)
        plot_distribution(history, run_id, run_dir, "faithful_pass_rate", "Faithful Pass Rate Distribution")
        plot_distribution(history, run_id, run_dir, "cite_rate", "Cite Rate Distribution")
        plot_scatter(history, run_id, run_dir)

    metrics: Dict[str, Dict[str, float]] = {}
    metric_names = [
        "accuracy",
        "audit_parse_rate",
        "faithful_pass_rate_mean",
        "audited_but_unfaithful_rate",
        "mean_extra_generation_calls_per_question",
    ]
    for name in metric_names:
        metrics[name] = {}
        for run_id, summary in per_run_summaries.items():
            history = per_run_histories[run_id]
            value = extract_metric_value(name, summary, history)
            if value is not None:
                metrics[name][run_id] = float(value)

    best_proposed = {"run_id": None, "value": None}
    best_baseline = {"run_id": None, "value": None}

    for run_id, summary in per_run_summaries.items():
        history = per_run_histories[run_id]
        value = extract_metric_value(PRIMARY_METRIC, summary, history)
        if value is None:
            continue
        run_id_lower = run_id.lower()
        if "proposed" in run_id_lower and (best_proposed["value"] is None or value > best_proposed["value"]):
            best_proposed = {"run_id": run_id, "value": value}
        if any(tag in run_id_lower for tag in ["comparative", "baseline"]) and (
            best_baseline["value"] is None or value > best_baseline["value"]
        ):
            best_baseline = {"run_id": run_id, "value": value}

    gap = None
    if best_proposed["value"] is not None and best_baseline["value"] is not None:
        gap = (best_proposed["value"] - best_baseline["value"]) / max(best_baseline["value"], 1e-12) * 100
        if is_minimize_metric(PRIMARY_METRIC):
            gap = -gap

    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    stat_tests = {}
    if best_proposed["run_id"] and best_baseline["run_id"]:
        if best_proposed["run_id"] in per_run_correct and best_baseline["run_id"] in per_run_correct:
            a = per_run_correct[best_proposed["run_id"]]
            b = per_run_correct[best_baseline["run_id"]]
            if len(a) > 0 and len(b) > 0:
                ci_low, ci_high, p_val = bootstrap_diff(a.astype(float), b.astype(float))
                stat_tests = {
                    "best_pair": [best_proposed["run_id"], best_baseline["run_id"]],
                    "ci_95": [ci_low, ci_high],
                    "p_value": p_val,
                }

    aggregated_metrics = {
        "primary_metric": PRIMARY_METRIC,
        "metrics": metrics,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
        "stat_tests": stat_tests,
    }
    save_json(comparison_dir / "aggregated_metrics.json", aggregated_metrics)

    comparison_bar(metrics, "accuracy", comparison_dir)
    comparison_bar(metrics, "audit_parse_rate", comparison_dir)
    comparison_bar(metrics, "faithful_pass_rate_mean", comparison_dir)
    comparison_bar(metrics, "mean_extra_generation_calls_per_question", comparison_dir)
    comparison_scatter(metrics, "mean_extra_generation_calls_per_question", "accuracy", comparison_dir)

    comparison_boxplot(per_run_histories, "faithful_pass_rate", comparison_dir)
    comparison_boxplot(per_run_histories, "cite_rate", comparison_dir)
    comparison_table(metrics, comparison_dir)

    if best_proposed["run_id"] and best_baseline["run_id"]:
        if best_proposed["run_id"] in per_run_correct and best_baseline["run_id"] in per_run_correct:
            plot_flip_matrix(
                per_run_correct[best_baseline["run_id"]],
                per_run_correct[best_proposed["run_id"]],
                best_baseline["run_id"],
                best_proposed["run_id"],
                comparison_dir,
            )


if __name__ == "__main__":
    main()
