from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _rewrite_run_override() -> None:
    new_argv = []
    for arg in sys.argv:
        if arg.startswith("run="):
            new_argv.append("runs@run=" + arg.split("=", 1)[1])
        else:
            new_argv.append(arg)
    sys.argv = new_argv


_rewrite_run_override()

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import (
    AnswerRegressor,
    DeltaAnswerReranker,
    FActA2Reranker,
    answer_margin_delta,
    cot_decode_candidates,
    load_model_and_tokenizer,
)
from .preprocess import (
    build_prompt,
    compare_answers,
    extract_final_answer,
    load_math_dataset,
    parse_number,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_mode_overrides(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, False)
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.training.max_questions = 2 if cfg.training.max_questions is None else min(int(cfg.training.max_questions), 2)
        cfg.training.max_batches = 2 if cfg.training.max_batches is None else min(int(cfg.training.max_batches), 2)
        cfg.training.epochs = 1
        if hasattr(cfg, "optuna"):
            cfg.optuna.n_trials = 0
        if hasattr(cfg, "run") and hasattr(cfg.run, "optuna"):
            cfg.run.optuna.n_trials = 0
        if hasattr(cfg, "run") and hasattr(cfg.run, "training"):
            cfg.run.training.epochs = 1
            if cfg.run.training.batch_size is not None:
                cfg.run.training.batch_size = min(int(cfg.run.training.batch_size), 2)
        if hasattr(cfg, "run") and hasattr(cfg.run, "decoding"):
            if cfg.run.decoding.get("max_new_tokens") is not None:
                cfg.run.decoding.max_new_tokens = min(int(cfg.run.decoding.max_new_tokens), 64)
            if cfg.run.decoding.get("k") is not None:
                cfg.run.decoding.k = max(1, min(int(cfg.run.decoding.k), 2))
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}. Use 'trial' or 'full'.")
    return cfg


def ensure_run_optuna(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "run") or cfg.run is None:
        raise ValueError("Missing run configuration. Ensure a run config is provided via run=<id>.")
    if not hasattr(cfg.run, "optuna") or cfg.run.optuna is None:
        cfg.run.optuna = OmegaConf.create({"n_trials": 0, "search_spaces": []})
    if cfg.run.optuna.get("search_spaces") is None:
        cfg.run.optuna.search_spaces = []
    if cfg.run.optuna.get("n_trials") is None:
        cfg.run.optuna.n_trials = 0
    return cfg


def validate_decoding_cfg(cfg: DictConfig) -> None:
    if not hasattr(cfg.run, "decoding"):
        raise ValueError("Run config missing decoding section.")
    assert int(cfg.run.decoding.k) > 0, "decoding.k must be positive."
    assert int(cfg.run.decoding.max_new_tokens) > 0, "decoding.max_new_tokens must be positive."


def candidate_cache_path(run_dir: Path, k: int, max_new_tokens: int) -> Path:
    fname = f"candidates_k{k}_max{max_new_tokens}.jsonl"
    return run_dir / fname


def load_candidate_cache(path: Path) -> Dict[int, Dict[str, object]]:
    cache: Dict[int, Dict[str, object]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = int(obj["index"])
            cache[idx] = obj
    return cache


def build_reranker(cfg: DictConfig):
    rerank_cfg = cfg.run.decoding.get("rerank_score", {})
    delta_only = rerank_cfg.get("delta_answer_only", False) or "baseline" in str(cfg.run.method).lower()
    if delta_only:
        return DeltaAnswerReranker()
    return FActA2Reranker(
        m=int(cfg.run.decoding.get("m", 3)),
        alpha=float(rerank_cfg.get("alpha", 1.2)),
        beta=float(rerank_cfg.get("beta", 0.7)),
        gamma=float(rerank_cfg.get("gamma", 0.5)),
        eps=float(rerank_cfg.get("eps", 1.0e-4)),
        triage_penalty=float(rerank_cfg.get("triage_penalty", 5.0)),
        dual_extraction=cfg.run.decoding.get("audit", {}).get("dual_extraction", True),
        cite_required=cfg.run.decoding.get("audit", {}).get("cite_required", True),
        agreement_on=cfg.run.decoding.get("audit", {}).get("agreement_on", "last_val"),
        answer_extraction=cfg.run.dataset.preprocessing.get("answer_extraction", "last-number-or-boxed"),
    )


def generate_or_load_candidates(
    idx: int,
    question: str,
    model,
    tokenizer,
    cfg: DictConfig,
    cache: Dict[int, Dict[str, object]],
    cache_writer,
) -> Tuple[List[str], List[float]]:
    if idx in cache:
        entry = cache[idx]
        return entry["candidates"], entry["deltas"]

    prompt = build_prompt(question)
    candidates = cot_decode_candidates(
        model,
        tokenizer,
        prompt,
        k=cfg.run.decoding.k,
        max_new_tokens=cfg.run.decoding.max_new_tokens,
        max_length=cfg.run.dataset.preprocessing.get("max_length", 1024),
    )
    deltas = [
        answer_margin_delta(
            model,
            tokenizer,
            prompt,
            cand,
            answer_extraction=cfg.run.dataset.preprocessing.get("answer_extraction", "last-number-or-boxed"),
        )
        for cand in candidates
    ]
    entry = {"index": idx, "question": question, "candidates": candidates, "deltas": deltas}
    cache[idx] = entry
    cache_writer.write(json.dumps(entry) + "\n")
    cache_writer.flush()
    return candidates, deltas


def assert_batch_valid(question: str, gold_answer: str, tokenizer) -> None:
    assert isinstance(question, str) and question.strip(), "Question must be a non-empty string."
    assert isinstance(gold_answer, str) and gold_answer.strip(), "Gold answer must be a non-empty string."
    tokenized = tokenizer(build_prompt(question), return_tensors="pt")
    assert tokenized.input_ids.ndim == 2 and tokenized.input_ids.shape[0] == 1, "Invalid input_ids shape."


def assert_training_batch(input_ids: torch.Tensor, labels: torch.Tensor) -> None:
    assert input_ids.ndim == 2, "Training input_ids must be 2D (batch, seq)."
    assert labels.ndim == 1, "Training labels must be 1D (batch,)."
    assert input_ids.shape[0] == labels.shape[0], "Batch size mismatch between inputs and labels."


def assert_gradients(model: torch.nn.Module) -> None:
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "No gradients found before optimizer step."
    non_zero = any(torch.any(g != 0) for g in grads if g is not None)
    assert non_zero, "All gradients are zero before optimizer step."


class QARegressionDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]], max_questions: Optional[int] = None):
        if max_questions is not None:
            examples = examples[: int(max_questions)]
        self.samples: List[Tuple[str, float]] = []
        for ex in examples:
            value = parse_number(ex["answer"])
            if value is None:
                continue
            self.samples.append((ex["question"], float(value)))
        if not self.samples:
            raise ValueError("No parseable numeric answers found for supervised training.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, float]:
        return self.samples[idx]


def build_collate_fn(tokenizer, max_length: int):
    def collate(batch: List[Tuple[str, float]]):
        questions = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        tokens = tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return tokens.input_ids, tokens.attention_mask, labels

    return collate


def run_supervised_training(
    cfg: DictConfig,
    model,
    tokenizer,
    dataset: List[Dict[str, str]],
    log_wandb: bool,
) -> Dict[str, float]:
    training_cfg = cfg.run.training
    epochs = int(training_cfg.epochs)
    if epochs <= 0:
        return {"train_loss": 0.0, "epochs": 0}

    max_length = cfg.run.dataset.preprocessing.get("max_length", 1024)
    train_dataset = QARegressionDataset(dataset, max_questions=cfg.training.max_questions)
    collate_fn = build_collate_fn(tokenizer, max_length)
    loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.batch_size),
        shuffle=True,
        collate_fn=collate_fn,
    )

    regressor = AnswerRegressor(
        base_model=model,
        freeze_base=bool(getattr(training_cfg, "freeze_base", True)),
    ).to(model.device)
    trainable_params = [p for p in regressor.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for supervised training.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(training_cfg.learning_rate) if float(training_cfg.learning_rate) > 0 else 1e-4,
    )

    global_step = 0
    total_loss = 0.0
    regressor.train()

    for epoch in range(epochs):
        for step, (input_ids, attention_mask, labels) in enumerate(loader):
            if cfg.training.max_batches is not None and step >= int(cfg.training.max_batches):
                break
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            labels = labels.to(model.device)
            if epoch == 0 and step == 0:
                assert_training_batch(input_ids, labels)

            preds = regressor(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.mse_loss(preds, labels)

            grad_norm = None
            if cfg.training.log_grad_norm:
                grads = torch.autograd.grad(loss, trainable_params, create_graph=False, retain_graph=True)
                grad_norm = torch.sqrt(sum((g.pow(2).sum() for g in grads if g is not None))).item()

            loss.backward()
            assert_gradients(regressor)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            global_step += 1

            if log_wandb and global_step % cfg.training.log_every == 0:
                payload = {
                    "train_loss": loss.item(),
                    "epoch": epoch,
                    "step": global_step,
                }
                if grad_norm is not None:
                    payload["grad_norm"] = grad_norm
                wandb.log(payload, step=global_step)

    avg_loss = total_loss / max(global_step, 1)
    return {"train_loss": avg_loss, "epochs": epochs}


def run_inference(
    cfg: DictConfig,
    model,
    tokenizer,
    dataset: List[Dict[str, str]],
    run_dir: Path,
    log_wandb: bool,
    max_questions: Optional[int] = None,
) -> Dict[str, float]:
    reranker = build_reranker(cfg)
    cache_path = candidate_cache_path(run_dir, cfg.run.decoding.k, cfg.run.decoding.max_new_tokens)
    cache = load_candidate_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    parse_success = 0
    faithful_pass_sum = 0.0
    audited_unfaithful = 0
    extra_calls_total = 0

    max_q = len(dataset) if max_questions is None else min(int(max_questions), len(dataset))
    progress = tqdm(dataset[:max_q], desc="Evaluating", unit="q")

    with cache_path.open("a", encoding="utf-8") as cache_writer, torch.inference_mode():
        for idx, example in enumerate(progress):
            question = example["question"]
            gold = example["answer"]
            if idx == 0:
                assert_batch_valid(question, gold, tokenizer)

            candidates, deltas = generate_or_load_candidates(
                idx, question, model, tokenizer, cfg, cache, cache_writer
            )

            output = reranker.rerank(
                question=question,
                candidates=candidates,
                deltas=deltas,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=cfg.run.decoding.max_new_tokens,
            )

            selected_text = candidates[output.selected_index]
            pred = extract_final_answer(
                selected_text,
                strategy=cfg.run.dataset.preprocessing.get("answer_extraction", "last-number-or-boxed"),
            )
            is_correct = int(compare_answers(pred, gold))

            total += 1
            correct += is_correct
            parse_success += int(output.audit_parse_success)
            faithful_pass_sum += float(output.faithful_pass_rate)
            extra_calls_total += int(output.extra_calls)

            audited_unfaithful_flag = int(output.arith_pass_rate >= 0.8 and output.cite_rate <= 0.3)
            audited_unfaithful += audited_unfaithful_flag

            running_accuracy = correct / total

            if log_wandb and (idx % cfg.training.log_every == 0):
                wandb.log(
                    {
                        "correct": is_correct,
                        "running_accuracy": running_accuracy,
                        "delta_answer": output.delta,
                        "selected_score": output.selected_score,
                        "audit_parse_success": output.audit_parse_success,
                        "audit_parse_variant_a": output.audit_parse_variant_a,
                        "audit_parse_variant_b": output.audit_parse_variant_b,
                        "faithful_pass_rate": output.faithful_pass_rate,
                        "arith_pass_rate": output.arith_pass_rate,
                        "cite_rate": output.cite_rate,
                        "final_consistent": output.final_consistent,
                        "agreement": output.agreement,
                        "audited_but_unfaithful": audited_unfaithful_flag,
                        "extra_generation_calls": output.extra_calls,
                        "question_index": idx,
                    },
                    step=total,
                )

            progress.set_postfix({"acc": f"{running_accuracy:.3f}"})

    accuracy = correct / max(1, total)
    audit_parse_rate = parse_success / max(1, total)
    faithful_pass_rate_mean = faithful_pass_sum / max(1, total)
    audited_but_unfaithful_rate = audited_unfaithful / max(1, total)
    mean_extra_generation_calls = extra_calls_total / max(1, total)

    metrics = {
        "accuracy": accuracy,
        "audit_parse_rate": audit_parse_rate,
        "faithful_pass_rate_mean": faithful_pass_rate_mean,
        "audited_but_unfaithful_rate": audited_but_unfaithful_rate,
        "mean_extra_generation_calls_per_question": mean_extra_generation_calls,
        "num_questions": total,
    }
    return metrics


def suggest_from_space(trial: optuna.Trial, space: Dict[str, object]):
    name = space["param_name"]
    dist = space["distribution_type"]
    if dist == "categorical":
        return trial.suggest_categorical(name, space["choices"])
    if dist == "uniform":
        return trial.suggest_float(name, float(space["low"]), float(space["high"]))
    if dist == "loguniform":
        return trial.suggest_float(name, float(space["low"]), float(space["high"]), log=True)
    raise ValueError(f"Unsupported distribution type: {dist}")


def apply_optuna_params(cfg: DictConfig, params: Dict[str, object]) -> None:
    OmegaConf.set_struct(cfg, False)
    for key, val in params.items():
        if key == "m":
            cfg.run.decoding.m = val
        elif key in {"alpha", "beta", "gamma", "eps", "triage_penalty"}:
            cfg.run.decoding.rerank_score[key] = val
        elif key == "max_new_tokens":
            cfg.run.decoding.max_new_tokens = val
        else:
            cfg.run.decoding[key] = val


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_mode_overrides(cfg)
    cfg = ensure_run_optuna(cfg)
    set_seed(int(cfg.seed))
    validate_decoding_cfg(cfg)

    run_id = cfg.run.run_id
    results_dir = Path(cfg.results_dir)
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_math_dataset(cfg.run.dataset, cache_dir=cfg.cache_dir)
    if not dataset:
        raise RuntimeError("Loaded dataset is empty. Check dataset split and preprocessing.")
    model, tokenizer = load_model_and_tokenizer(cfg.run.model, cache_dir=cfg.cache_dir)

    # Post-init assertions
    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set."
    out_emb = model.get_output_embeddings()
    assert out_emb is not None and out_emb.weight.shape[0] == len(tokenizer), (
        "Model output embeddings do not match tokenizer vocab size."
    )

    optuna_cfg = cfg.run.optuna
    n_trials = int(optuna_cfg.get("n_trials", 0))

    best_params: Dict[str, object] = {}
    if n_trials > 0:
        search_spaces = optuna_cfg.get("search_spaces") or []
        study = optuna.create_study(direction=cfg.optuna.direction)
        val_size = int(cfg.optuna.validation_size)

        def objective(trial: optuna.Trial) -> float:
            params = {s["param_name"]: suggest_from_space(trial, s) for s in search_spaces}
            trial_cfg = copy.deepcopy(cfg)
            OmegaConf.set_struct(trial_cfg, False)
            trial_cfg.wandb.mode = "disabled"
            apply_optuna_params(trial_cfg, params)
            metrics = run_inference(
                trial_cfg,
                model,
                tokenizer,
                dataset,
                run_dir=run_dir,
                log_wandb=False,
                max_questions=min(val_size, len(dataset)),
            )
            return float(metrics["accuracy"])

        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        apply_optuna_params(cfg, best_params)

    log_wandb = cfg.wandb.mode != "disabled"
    if log_wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )

    if str(cfg.run.training.task).lower() == "inference_only":
        metrics = run_inference(
            cfg,
            model,
            tokenizer,
            dataset,
            run_dir=run_dir,
            log_wandb=log_wandb,
            max_questions=cfg.training.max_questions,
        )
    else:
        metrics = run_supervised_training(
            cfg,
            model,
            tokenizer,
            dataset,
            log_wandb=log_wandb,
        )

    if log_wandb:
        for key, value in metrics.items():
            wandb.summary[key] = value
        if best_params:
            wandb.summary["best_params"] = best_params
        print(f"WandB run URL: {wandb.run.get_url()}")
        wandb.finish()


if __name__ == "__main__":
    main()
