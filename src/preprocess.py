from __future__ import annotations

import re
from typing import Dict, List, Optional

from datasets import load_dataset


def build_prompt(question: str) -> str:
    return f"Q: {question.strip()}\nA:"


def normalize_numeric_str(text: str) -> str:
    return text.strip().replace(",", "")


def parse_number(text: str) -> Optional[float]:
    if text is None:
        return None
    text = normalize_numeric_str(str(text))
    text = re.sub(r"[^0-9\-\./]", "", text)
    if not text:
        return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?/-?\d+(?:\.\d+)?", text):
        num, den = text.split("/")
        try:
            return float(num) / float(den)
        except ZeroDivisionError:
            return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return float(text)
    return None


def extract_final_answer(text: str, strategy: str = "last-number-or-boxed") -> str:
    if not text:
        return ""
    if strategy == "last-number-or-boxed":
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            return boxed[-1].strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/-?\d+(?:\.\d+)?)?", text.replace(",", ""))
    return numbers[-1].strip() if numbers else ""


def compare_answers(pred: str, gold: str, tol: float = 1e-6) -> bool:
    pred_val = parse_number(pred)
    gold_val = parse_number(gold)
    if pred_val is None or gold_val is None:
        return False
    return abs(pred_val - gold_val) <= tol * max(1.0, abs(gold_val))


def extract_gsm8k_gold(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return extract_final_answer(answer_text)


def _safe_load_dataset(*args, **kwargs):
    try:
        return load_dataset(*args, **kwargs)
    except Exception as exc:
        name = args[0] if args else kwargs.get("path", "unknown")
        split = kwargs.get("split", "unknown")
        cache_dir = kwargs.get("cache_dir", "unknown")
        msg = (
            f"Failed to load dataset '{name}' with split='{split}'. "
            f"Check dataset availability and cache_dir='{cache_dir}'."
        )
        raise RuntimeError(msg) from exc


def load_math_dataset(dataset_cfg, cache_dir: str = ".cache/") -> List[Dict[str, str]]:
    name = str(dataset_cfg.name).lower()
    split = dataset_cfg.split

    if name == "gsm8k":
        ds = _safe_load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
        examples: List[Dict[str, str]] = []
        for ex in ds:
            question = str(ex["question"]).strip()
            answer = extract_gsm8k_gold(str(ex["answer"]))
            examples.append({"question": question, "answer": answer})
        return examples

    if name == "svamp":
        ds = _safe_load_dataset("svamp", split=split, cache_dir=cache_dir)
        examples = []
        for ex in ds:
            body = str(ex.get("Body", ""))
            question = str(ex.get("Question", ""))
            full_question = f"{body} {question}".strip()
            answer = str(ex.get("Answer", "")).strip()
            examples.append({"question": full_question, "answer": answer})
        return examples

    raise ValueError(f"Unsupported dataset name: {dataset_cfg.name}")
