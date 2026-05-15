"""
Mechanistic metrics for the full-realism contamination experiment.

The metrics here are designed to be computed on a frozen calibration batch
of real OpenWebText so that all numbers are comparable across (ratio, seed,
checkpoint). They target the kinds of internal signals that we expect to
exhibit a phase transition before behavioral perplexity collapses:

- Layerwise effective rank of hidden states (PCA-based)
- LoRA / adapter weight norm drift (training-data invariant signal)
- Attention pattern entropy across heads / layers
- Feature density via PCA on a sample of activations
- Cosine similarity between gradients across recent steps (gradient topology)

All functions are implemented to work on any GPT-2 class HuggingFace model.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _maybe_unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap PEFT / DDP wrappers to get to the bare HF model."""
    inner = model
    for attr in ("module", "base_model"):
        if hasattr(inner, attr):
            cand = getattr(inner, attr)
            if isinstance(cand, torch.nn.Module):
                inner = cand
    if hasattr(inner, "model"):
        cand = inner.model
        if isinstance(cand, torch.nn.Module) and hasattr(cand, "transformer"):
            inner = cand
    return inner


def _transformer(model: torch.nn.Module):
    """Return the GPT-2 backbone (`.transformer`)."""
    inner = _maybe_unwrap(model)
    if hasattr(inner, "transformer"):
        return inner.transformer
    if hasattr(inner, "h"):  # already the GPT2Model
        return inner
    raise ValueError("Could not locate GPT-2 transformer backbone on model.")


def _to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# 1) Held-out perplexity
# ---------------------------------------------------------------------------


@torch.no_grad()
def held_out_perplexity(
    model: torch.nn.Module,
    eval_batches: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    """Per-token perplexity on a held-out batch of real text."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in eval_batches:
        batch = _to_device(batch, device)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        if attention_mask is not None:
            tgt_mask = attention_mask[:, 1:].reshape(-1).float()
            loss = (per_tok * tgt_mask).sum().item()
            n_tok = int(tgt_mask.sum().item())
        else:
            loss = per_tok.sum().item()
            n_tok = targets.numel()
        total_loss += loss
        total_tokens += n_tok
    if total_tokens == 0:
        return float("inf")
    return float(math.exp(total_loss / total_tokens))


# ---------------------------------------------------------------------------
# 2) Layerwise effective rank of hidden states
# ---------------------------------------------------------------------------


@torch.no_grad()
def _layerwise_hidden_states(
    model: torch.nn.Module,
    calibration_batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    """Return list of per-layer hidden states, each shape (M, H)
    where M is the number of attended tokens in the calibration batch.
    """
    model.eval()
    batch = _to_device(calibration_batch, device)
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        output_hidden_states=True,
        return_dict=True,
    )
    hs_layers = out.hidden_states  # tuple of (B, T, H)
    mask = batch.get("attention_mask")
    flats = []
    for hs in hs_layers:
        if mask is not None:
            m = mask.bool().reshape(-1)
            flat = hs.reshape(-1, hs.size(-1))[m]
        else:
            flat = hs.reshape(-1, hs.size(-1))
        flats.append(flat.float())
    return flats


def _effective_rank(matrix: torch.Tensor, n_components: int = 256) -> float:
    """Effective rank: exp(entropy of normalized squared singular values)."""
    if matrix.numel() == 0:
        return 0.0
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    k = int(min(n_components, centered.shape[0], centered.shape[1]))
    if k < 2:
        return 0.0
    s = torch.linalg.svdvals(centered)[:k]
    s2 = s.pow(2)
    total = s2.sum()
    if total <= 0:
        return 0.0
    p = (s2 / total).clamp_min(1e-12)
    H = -(p * p.log()).sum()
    return float(torch.exp(H).item())


@torch.no_grad()
def representation_rank_layerwise(
    model: torch.nn.Module,
    calibration_batch: Dict[str, torch.Tensor],
    device: torch.device,
    n_components: int = 256,
) -> Dict[str, float]:
    """Effective rank per layer."""
    flats = _layerwise_hidden_states(model, calibration_batch, device)
    out: Dict[str, float] = {}
    ranks = []
    for i, flat in enumerate(flats):
        r = _effective_rank(flat, n_components=n_components)
        out[f"repr_rank_layer_{i}"] = r
        ranks.append(r)
    if ranks:
        out["repr_rank_mean"] = float(np.mean(ranks))
        out["repr_rank_last"] = float(ranks[-1])
        out["repr_rank_min"] = float(min(ranks))
    return out


# ---------------------------------------------------------------------------
# 3) Weight norm drift (LoRA-aware)
# ---------------------------------------------------------------------------


def lora_weight_norms(
    model: torch.nn.Module,
    init_norms: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """L2 norm of LoRA A and B matrices (if any), and overall trainable norm.

    If `init_norms` is provided, also reports drift = current - init.
    """
    out: Dict[str, float] = {}
    a_sq, b_sq = 0.0, 0.0
    trainable_sq = 0.0
    n_a, n_b, n_trainable = 0, 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n2 = float(p.detach().pow(2).sum().item())
        trainable_sq += n2
        n_trainable += 1
        if "lora_A" in name:
            a_sq += n2
            n_a += 1
        elif "lora_B" in name:
            b_sq += n2
            n_b += 1
    out["lora_A_norm"] = math.sqrt(a_sq) if n_a else 0.0
    out["lora_B_norm"] = math.sqrt(b_sq) if n_b else 0.0
    out["trainable_norm"] = math.sqrt(trainable_sq)
    if init_norms is not None:
        for k in ("lora_A_norm", "lora_B_norm", "trainable_norm"):
            out[f"{k}_drift"] = out[k] - init_norms.get(k, 0.0)
    return out


def snapshot_norms(model: torch.nn.Module) -> Dict[str, float]:
    """Capture initial trainable-parameter norms for later drift tracking."""
    return {
        k: v for k, v in lora_weight_norms(model).items() if not k.endswith("_drift")
    }


# ---------------------------------------------------------------------------
# 4) Attention entropy across heads/layers
# ---------------------------------------------------------------------------


@torch.no_grad()
def attention_pattern_entropy(
    model: torch.nn.Module,
    calibration_batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    """Mean Shannon entropy of attention distributions per layer.

    A collapsed attention pattern (e.g. always attend to <bos>) has very low
    entropy; a healthy attention has entropy growing with sequence length.
    """
    model.eval()
    batch = _to_device(calibration_batch, device)
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        output_attentions=True,
        return_dict=True,
    )
    attns = out.attentions  # tuple of (B, n_heads, T, T)
    if not attns:
        return {}
    mask = batch.get("attention_mask")
    layer_H = []
    for li, A in enumerate(attns):
        # Each row of A sums to 1 over keys
        # H_row = -sum p log p
        Af = A.float().clamp_min(1e-12)
        H = -(Af * Af.log()).sum(dim=-1)  # (B, n_heads, T)
        if mask is not None:
            m = mask.bool().unsqueeze(1).expand_as(H)
            vals = H[m].mean()
        else:
            vals = H.mean()
        layer_H.append(float(vals.item()))
    out_d = {f"attn_entropy_layer_{i}": v for i, v in enumerate(layer_H)}
    out_d["attn_entropy_mean"] = float(np.mean(layer_H))
    out_d["attn_entropy_last"] = float(layer_H[-1])
    return out_d


# ---------------------------------------------------------------------------
# 5) Feature density via PCA
# ---------------------------------------------------------------------------


@torch.no_grad()
def feature_density_pca(
    model: torch.nn.Module,
    calibration_batch: Dict[str, torch.Tensor],
    device: torch.device,
    var_threshold: float = 0.95,
    n_components: int = 256,
) -> Dict[str, float]:
    """Number of PCA components needed to explain `var_threshold` of variance
    in the last-layer hidden state. A sharp drop indicates representation
    collapse.
    """
    flats = _layerwise_hidden_states(model, calibration_batch, device)
    if not flats:
        return {}
    last = flats[-1]
    if last.numel() == 0:
        return {"feat_density": 0.0}
    centered = last - last.mean(dim=0, keepdim=True)
    k = int(min(n_components, centered.shape[0], centered.shape[1]))
    s = torch.linalg.svdvals(centered)[:k]
    var = s.pow(2)
    total = var.sum()
    if total <= 0:
        return {"feat_density": 0.0}
    cum = torch.cumsum(var / total, dim=0)
    n_needed = int(((cum < var_threshold).sum() + 1).item())
    return {
        "feat_density": float(n_needed),
        "feat_density_frac": float(n_needed / k),
    }


# ---------------------------------------------------------------------------
# 6) Gradient topology
# ---------------------------------------------------------------------------


class GradientTopologyTracker:
    """Tracks cosine similarity between gradient snapshots over time.

    Calling `update` flattens the gradients of all trainable parameters, stores
    them in a small ring buffer, and computes the mean pairwise cosine
    similarity. Higher values mean gradients are more correlated step-to-step
    (typically late training); a sudden spike in cosine alignment can indicate
    representation collapse.
    """

    def __init__(self, window: int = 8):
        self.window = window
        self.buffer: List[torch.Tensor] = []

    @torch.no_grad()
    def _flat_grads(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        parts = []
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                parts.append(p.grad.detach().reshape(-1).float().cpu())
        if not parts:
            return None
        return torch.cat(parts)

    def update(self, model: torch.nn.Module) -> Dict[str, float]:
        v = self._flat_grads(model)
        if v is None:
            return {}
        norm = torch.linalg.vector_norm(v)
        if norm <= 0:
            return {}
        v = v / norm
        self.buffer.append(v)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        if len(self.buffer) < 2:
            return {"grad_norm": float(norm.item())}
        sims = []
        for i in range(len(self.buffer) - 1):
            sims.append(float((self.buffer[i] @ self.buffer[-1]).item()))
        return {
            "grad_norm": float(norm.item()),
            "grad_cos_recent_mean": float(np.mean(sims)),
            "grad_cos_recent_max": float(np.max(sims)),
        }


# ---------------------------------------------------------------------------
# 7) N-gram diversity (for behavioral cross-check)
# ---------------------------------------------------------------------------


def _distinct_ngram_ratio(token_lists: List[List[int]], n: int) -> float:
    total = 0
    seen: Counter = Counter()
    for toks in token_lists:
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            seen[tuple(toks[i : i + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    return len(seen) / total


def _repetition_rate(token_lists: List[List[int]]) -> float:
    """Fraction of tokens that are an immediate repeat of the previous token."""
    rep, tot = 0, 0
    for toks in token_lists:
        for i in range(1, len(toks)):
            tot += 1
            if toks[i] == toks[i - 1]:
                rep += 1
    return rep / tot if tot else 0.0


@torch.no_grad()
def ngram_diversity_and_repetition(
    model: torch.nn.Module,
    prompt_input_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 96,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> Dict[str, float]:
    """Generate continuations from fixed prompts and measure distinct-n &
    immediate-repetition rates."""
    model.eval()
    prompt_input_ids = prompt_input_ids.to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id

    base = _maybe_unwrap(model)
    target = base if hasattr(base, "generate") else model
    out = target.generate(input_ids=prompt_input_ids, **gen_kwargs)

    prompt_len = prompt_input_ids.shape[1]
    generated = out[:, prompt_len:].cpu().tolist()
    res = {f"distinct_{n}": _distinct_ngram_ratio(generated, n) for n in (2, 3, 4)}
    res["repetition_rate"] = _repetition_rate(generated)
    return res


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_all_metrics(
    model: torch.nn.Module,
    calibration_batch: Dict[str, torch.Tensor],
    eval_batches: Iterable[Dict[str, torch.Tensor]],
    prompt_input_ids: torch.Tensor,
    device: torch.device,
    init_norms: Optional[Dict[str, float]] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    n_pca_components: int = 256,
    max_new_tokens: int = 96,
) -> Dict[str, float]:
    """Compute every mechanistic + behavioral metric for one checkpoint."""
    was_training = model.training
    model.eval()
    metrics: Dict[str, float] = {}

    # 1) perplexity
    metrics["perplexity"] = held_out_perplexity(model, eval_batches, device)
    # 2) layerwise effective rank
    metrics.update(
        representation_rank_layerwise(
            model,
            calibration_batch,
            device,
            n_components=n_pca_components,
        )
    )
    # 3) weight-norm drift
    metrics.update(lora_weight_norms(model, init_norms=init_norms))
    # 4) attention entropy
    try:
        metrics.update(attention_pattern_entropy(model, calibration_batch, device))
    except Exception as e:
        print(f"[metrics] attention entropy failed: {e}", flush=True)
    # 5) feature density
    try:
        metrics.update(
            feature_density_pca(
                model,
                calibration_batch,
                device,
                n_components=n_pca_components,
            )
        )
    except Exception as e:
        print(f"[metrics] feature density failed: {e}", flush=True)
    # 6) ngram diversity / repetition
    try:
        metrics.update(
            ngram_diversity_and_repetition(
                model,
                prompt_input_ids,
                device,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        )
    except Exception as e:
        print(f"[metrics] ngram diversity failed: {e}", flush=True)

    if was_training:
        model.train()
    return metrics
