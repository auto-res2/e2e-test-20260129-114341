from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .preprocess import extract_final_answer


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


class UnsafeExpr(Exception):
    pass


def _check_ast(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _check_ast(node.body)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise UnsafeExpr
        _check_ast(node.left)
        _check_ast(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise UnsafeExpr
        _check_ast(node.operand)
    elif isinstance(node, (ast.Constant, ast.Num)):
        val = node.n if hasattr(node, "n") else node.value
        if not isinstance(val, (int, float)):
            raise UnsafeExpr
    else:
        raise UnsafeExpr


def safe_eval_arith(expr: str) -> float:
    expr = expr.strip()
    if not re.fullmatch(r"[0-9\s\+\-\*/\(\)\.\*]+", expr):
        raise UnsafeExpr
    tree = ast.parse(expr, mode="eval")
    _check_ast(tree)
    return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {}))


def parse_first_json_array(text: str) -> Optional[List[Dict[str, object]]]:
    match = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def cite_tokens_ok(expr: str, cite: str) -> bool:
    expr_tokens = re.findall(r"\d+\.?\d*|\*\*|[\+\-\*/\(\)]", expr)
    cite_flat = cite.replace(" ", "")
    return all(token in cite_flat for token in expr_tokens)


def audit_trace(
    trace: Optional[List[Dict[str, object]]],
    candidate_text: str,
    final_answer_str: str,
    tol: float = 1e-6,
    cite_required: bool = True,
) -> Dict[str, object]:
    if not isinstance(trace, list) or len(trace) == 0:
        return {
            "arith_pass_rate": 0.0,
            "cite_rate": 0.0,
            "faithful_pass_rate": 0.0,
            "final_consistent": 0,
            "last_val": None,
            "values": [],
            "parseable": False,
        }

    arith_pass = 0
    cite_pass = 0
    values: List[float] = []
    last_val = None
    n = 0

    for step in trace:
        if not isinstance(step, dict):
            continue
        if not all(k in step for k in ["expr", "val", "cite"]):
            continue
        n += 1
        expr = str(step["expr"]).strip()
        cite = str(step["cite"])
        cite_ok = True
        if cite_required:
            cite_ok = (cite in candidate_text) and cite_tokens_ok(expr, cite)
        if cite_ok:
            cite_pass += 1
        try:
            claimed = float(step["val"])
            computed = safe_eval_arith(expr)
            arith_ok = math.isfinite(computed) and abs(computed - claimed) <= tol * max(1.0, abs(claimed))
        except Exception:
            arith_ok = False
        if arith_ok:
            arith_pass += 1
            values.append(claimed)
            if cite_ok:
                last_val = claimed

    if n == 0:
        return {
            "arith_pass_rate": 0.0,
            "cite_rate": 0.0,
            "faithful_pass_rate": 0.0,
            "final_consistent": 0,
            "last_val": None,
            "values": [],
            "parseable": False,
        }

    arith_rate = arith_pass / n
    cite_rate = cite_pass / n if cite_required else 1.0
    faithful_rate = arith_rate * cite_rate

    final_consistent = 0
    if final_answer_str and last_val is not None:
        try:
            final_consistent = int(
                abs(float(final_answer_str) - float(last_val)) <= tol * max(1.0, abs(float(last_val)))
            )
        except Exception:
            final_consistent = 0

    return {
        "arith_pass_rate": arith_rate,
        "cite_rate": cite_rate,
        "faithful_pass_rate": faithful_rate,
        "final_consistent": final_consistent,
        "last_val": last_val,
        "values": values,
        "parseable": True,
    }


def agreement_score(
    audit_a: Dict[str, object], audit_b: Dict[str, object], eps: float = 1e-4, mode: str = "values"
) -> float:
    if mode == "last_val":
        va, vb = audit_a.get("last_val"), audit_b.get("last_val")
        if va is None or vb is None:
            return eps
        return 1.0 if abs(float(va) - float(vb)) <= 1e-6 * max(1.0, abs(float(va))) else eps

    vals_a = audit_a.get("values", [])
    vals_b = audit_b.get("values", [])
    if not vals_a or not vals_b:
        return eps
    n = min(len(vals_a), len(vals_b))
    if n == 0:
        return eps
    matches = 0
    for va, vb in zip(vals_a[:n], vals_b[:n]):
        if abs(float(va) - float(vb)) <= 1e-6 * max(1.0, abs(float(va))):
            matches += 1
    return max(matches / n, eps)


@torch.inference_mode()
def extract_trace(
    model,
    tokenizer,
    question: str,
    candidate: str,
    variant: int = 0,
    max_new_tokens: int = 256,
) -> Optional[List[Dict[str, object]]]:
    if variant == 0:
        header = (
            "Extract arithmetic steps from the candidate as JSON only.\n"
            "Output JSON array of objects with keys: expr (string), val (number), cite (verbatim substring from candidate).\n"
            "Constraints: expr uses only digits,+,-,*,/,(), optional **. cite must appear EXACTLY in the candidate.\n"
            "No extra text.\n"
        )
    else:
        header = (
            "Return ONLY a JSON array. Each element: {\"cite\":...,\"expr\":...,\"val\":...}.\n"
            "cite must be copied exactly from the candidate solution text.\n"
            "expr must be pure arithmetic with digits and + - * / ( ) and optional **.\n"
            "No commentary.\n"
        )
    prompt = f"{header}\nQuestion: {question}\nCandidate: {candidate}\nJSON:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return parse_first_json_array(gen)


def load_model_and_tokenizer(model_cfg, cache_dir: str = ".cache/"):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(str(model_cfg.dtype), torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map=model_cfg.device_map,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def cot_decode_candidates(
    model,
    tokenizer,
    prompt: str,
    k: int = 10,
    max_new_tokens: int = 256,
    max_length: int = 1024,
) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    logits = model(**inputs).logits[:, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    topk = torch.topk(log_probs, k)
    candidates: List[str] = []
    for token_id in topk.indices[0].tolist():
        first = torch.tensor([[token_id]], device=model.device)
        input_ids = torch.cat([inputs.input_ids, first], dim=1)
        attention_mask = torch.cat(
            [inputs.attention_mask, torch.ones_like(first, device=model.device)], dim=1
        )
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0, inputs.input_ids.shape[1] :]
        candidates.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return candidates


@torch.inference_mode()
def answer_margin_delta(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    answer_extraction: str = "last-number-or-boxed",
) -> float:
    ans = extract_final_answer(completion, answer_extraction)
    if not ans:
        return -1e9
    full = prompt + completion
    ids = tokenizer(full, return_tensors="pt").input_ids.to(model.device)
    out = model(ids)
    log_probs = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
    ans_ids = tokenizer(ans, add_special_tokens=False).input_ids
    if len(ans_ids) == 0 or ids.shape[1] < len(ans_ids) + 1:
        return -1e9
    margins = []
    for i in range(len(ans_ids)):
        pos = -len(ans_ids) + i - 1
        lp = log_probs[0, pos]
        top2 = torch.topk(lp, 2).values
        margins.append((top2[0] - top2[1]).item())
    return float(np.mean(margins)) if margins else -1e9


@dataclass
class RerankOutput:
    selected_index: int
    selected_score: float
    scores: List[float]
    delta: float
    audit_parse_success: int
    audit_parse_variant_a: int
    audit_parse_variant_b: int
    faithful_pass_rate: float
    arith_pass_rate: float
    cite_rate: float
    final_consistent: int
    agreement: float
    extra_calls: int


class DeltaAnswerReranker:
    def rerank(
        self,
        question: str,
        candidates: List[str],
        deltas: List[float],
        model,
        tokenizer,
        max_new_tokens: int = 256,
    ) -> RerankOutput:
        scores = deltas
        selected_index = int(np.argmax(scores))
        return RerankOutput(
            selected_index=selected_index,
            selected_score=float(scores[selected_index]),
            scores=list(scores),
            delta=float(deltas[selected_index]),
            audit_parse_success=0,
            audit_parse_variant_a=0,
            audit_parse_variant_b=0,
            faithful_pass_rate=0.0,
            arith_pass_rate=0.0,
            cite_rate=0.0,
            final_consistent=0,
            agreement=0.0,
            extra_calls=0,
        )


class FActA2Reranker:
    def __init__(
        self,
        m: int,
        alpha: float,
        beta: float,
        gamma: float,
        eps: float,
        triage_penalty: float,
        dual_extraction: bool,
        cite_required: bool,
        agreement_on: str,
        answer_extraction: str,
    ) -> None:
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.triage_penalty = triage_penalty
        self.dual_extraction = dual_extraction
        self.cite_required = cite_required
        self.agreement_on = agreement_on
        self.answer_extraction = answer_extraction

    def rerank(
        self,
        question: str,
        candidates: List[str],
        deltas: List[float],
        model,
        tokenizer,
        max_new_tokens: int = 256,
    ) -> RerankOutput:
        top_idx = sorted(range(len(candidates)), key=lambda i: deltas[i], reverse=True)[: self.m]
        scores: List[float] = []
        infos: List[Dict[str, object]] = []
        extra_calls = 0

        for i, cand in enumerate(candidates):
            delta = float(deltas[i])
            if i not in top_idx:
                score = delta - self.triage_penalty
                info = {
                    "score": score,
                    "delta": delta,
                    "audit_parse_success": 0,
                    "audit_parse_variant_a": 0,
                    "audit_parse_variant_b": 0,
                    "faithful_pass_rate": 0.0,
                    "arith_pass_rate": 0.0,
                    "cite_rate": 0.0,
                    "final_consistent": 0,
                    "agreement": 0.0,
                }
            else:
                ans_str = extract_final_answer(cand, self.answer_extraction)
                trace_a = extract_trace(
                    model,
                    tokenizer,
                    question,
                    cand,
                    variant=0,
                    max_new_tokens=max_new_tokens,
                )
                extra_calls += 1
                audit_a = audit_trace(
                    trace_a,
                    cand,
                    ans_str,
                    cite_required=self.cite_required,
                )
                parseable_a = int(audit_a.get("parseable", False))

                if self.dual_extraction:
                    trace_b = extract_trace(
                        model,
                        tokenizer,
                        question,
                        cand,
                        variant=1,
                        max_new_tokens=max_new_tokens,
                    )
                    extra_calls += 1
                    audit_b = audit_trace(
                        trace_b,
                        cand,
                        ans_str,
                        cite_required=self.cite_required,
                    )
                    parseable_b = int(audit_b.get("parseable", False))
                else:
                    audit_b = {
                        "faithful_pass_rate": 0.0,
                        "final_consistent": 0,
                        "values": [],
                        "parseable": False,
                        "arith_pass_rate": 0.0,
                        "cite_rate": 0.0,
                        "last_val": None,
                    }
                    parseable_b = 0

                best_audit = audit_a if audit_a["faithful_pass_rate"] >= audit_b["faithful_pass_rate"] else audit_b
                faithful_rate = float(best_audit["faithful_pass_rate"])
                final_consistent = int(best_audit["final_consistent"])
                agreement = (
                    agreement_score(audit_a, audit_b, eps=self.eps, mode=self.agreement_on)
                    if self.dual_extraction
                    else 1.0
                )
                score = delta + self.alpha * math.log(faithful_rate + self.eps) + self.beta * final_consistent
                score += self.gamma * math.log(agreement + self.eps)

                info = {
                    "score": score,
                    "delta": delta,
                    "audit_parse_success": int(parseable_a or parseable_b),
                    "audit_parse_variant_a": parseable_a,
                    "audit_parse_variant_b": parseable_b,
                    "faithful_pass_rate": faithful_rate,
                    "arith_pass_rate": float(best_audit.get("arith_pass_rate", 0.0)),
                    "cite_rate": float(best_audit.get("cite_rate", 0.0)),
                    "final_consistent": final_consistent,
                    "agreement": agreement,
                }

            scores.append(float(info["score"]))
            infos.append(info)

        selected_index = int(np.argmax(scores))
        selected_info = infos[selected_index]
        return RerankOutput(
            selected_index=selected_index,
            selected_score=float(scores[selected_index]),
            scores=scores,
            delta=float(selected_info["delta"]),
            audit_parse_success=int(selected_info["audit_parse_success"]),
            audit_parse_variant_a=int(selected_info["audit_parse_variant_a"]),
            audit_parse_variant_b=int(selected_info["audit_parse_variant_b"]),
            faithful_pass_rate=float(selected_info["faithful_pass_rate"]),
            arith_pass_rate=float(selected_info["arith_pass_rate"]),
            cite_rate=float(selected_info["cite_rate"]),
            final_consistent=int(selected_info["final_consistent"]),
            agreement=float(selected_info["agreement"]),
            extra_calls=extra_calls,
        )


class AnswerRegressor(nn.Module):
    def __init__(self, base_model: nn.Module, freeze_base: bool = True) -> None:
        super().__init__()
        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Base model config must include hidden_size for regression head.")
        self.base_model = base_model
        self.freeze_base = freeze_base
        self.regressor = nn.Linear(hidden_size, 1)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_base:
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden = outputs.hidden_states[-1].detach()
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.regressor(pooled).squeeze(-1)
