"""
Mechanistic metrics for the Data Contamination Gradient experiment.

Each metric is computed against the same fixed calibration batch across all
runs so that values are comparable across (ratio, seed, step). Metrics are
designed to add <10% wall-clock overhead per logging step.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_attention_weight(model: torch.nn.Module) -> torch.Tensor:
    """
    Return the last attention layer's projection weight (c_attn for GPT-2).
    Falls back to scanning named_parameters for the highest-indexed `attn` weight.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(model, "transformer"):
        n_layer = cfg.n_layer
        attn = model.transformer.h[n_layer - 1].attn
        if hasattr(attn, "c_attn"):
            return attn.c_attn.weight.detach()
        if hasattr(attn, "q_proj"):
            return torch.cat(
                [attn.q_proj.weight.detach(),
                 attn.k_proj.weight.detach(),
                 attn.v_proj.weight.detach()], dim=0)
    last_w = None
    last_idx = -1
    for name, p in model.named_parameters():
        if "attn" in name and p.dim() == 2:
            try:
                idx = int([t for t in name.split(".") if t.isdigit()][0])
            except (IndexError, ValueError):
                idx = 0
            if idx >= last_idx:
                last_idx = idx
                last_w = p.detach()
    if last_w is None:
        raise ValueError("Could not locate an attention weight matrix in model.")
    return last_w


def _hidden_states(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Return last-layer hidden states, shape (B*T, H), with attention mask applied."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    kwargs = {"output_hidden_states": True, "return_dict": True}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(device)
    out = model(input_ids=input_ids, **kwargs)
    hs = out.hidden_states[-1]  # (B, T, H)
    if attention_mask is not None:
        mask = attention_mask.to(device).bool().reshape(-1)
        flat = hs.reshape(-1, hs.size(-1))
        flat = flat[mask]
    else:
        flat = hs.reshape(-1, hs.size(-1))
    return flat


# ---------------------------------------------------------------------------
# 1) Held-out perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def perplexity(
    model: torch.nn.Module,
    eval_batches: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    """Token-level perplexity on a held-out batch of real text."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in eval_batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(device)
        out = model(input_ids=input_ids, **kwargs)
        logits = out.logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        n_tok = targets.numel()
        if attention_mask is not None:
            tgt_mask = attention_mask[:, 1:].to(device).reshape(-1).float()
            # recompute with masked sum
            per_tok = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            )
            loss = (per_tok * tgt_mask).sum()
            n_tok = int(tgt_mask.sum().item())
        total_loss += loss.item()
        total_tokens += n_tok
    if total_tokens == 0:
        return float("inf")
    return float(math.exp(total_loss / total_tokens))


# ---------------------------------------------------------------------------
# 2) Effective rank of last attention weight
# ---------------------------------------------------------------------------

@torch.no_grad()
def attention_weight_rank(model: torch.nn.Module) -> float:
    """
    Effective rank = sum(s_i^2) / max(s_i^2) of the last attention weight.
    Equals 1 if rank-1, scales up as energy spreads across more directions.
    """
    W = _last_attention_weight(model).float()
    if W.numel() == 0 or torch.isnan(W).any() or torch.isinf(W).any():
        return 0.0
    s = torch.linalg.svdvals(W)
    s2 = s.pow(2)
    if s2.numel() == 0 or s2.max() <= 1e-12 or torch.isnan(s2).any():
        return 0.0
    return float((s2.sum() / s2.max()).item())


# ---------------------------------------------------------------------------
# 3) Representation entropy (PCA explained-variance entropy)
# ---------------------------------------------------------------------------

@torch.no_grad()
def representation_entropy(
    model: torch.nn.Module,
    calibration_batch: Dict[str, torch.Tensor],
    device: torch.device,
    n_components: int = 64,
) -> float:
    """
    Shannon entropy (nats) of the explained-variance ratios of PCA on
    last-layer hidden states. Higher = more isotropic representation.
    """
    flat = _hidden_states(model, calibration_batch, device).float()
    flat = flat - flat.mean(dim=0, keepdim=True)
    n_samples = flat.shape[0]
    if n_samples < 2:
        return 0.0
    k = int(min(n_components, flat.shape[1], n_samples - 1))
    s = torch.linalg.svdvals(flat)[:k]
    var = s.pow(2)
    total = var.sum()
    if total <= 0:
        return 0.0
    p = (var / total).clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())


# ---------------------------------------------------------------------------
# 4) Directional concentration: cosine similarity of random hidden pairs
# ---------------------------------------------------------------------------

@torch.no_grad()
def directional_concentration(
    model: torch.nn.Module,
    calibration_batch: Dict[str, torch.Tensor],
    device: torch.device,
    n_pairs: int = 4096,
    rng_seed: int = 0,
) -> Tuple[float, float]:
    """
    Mean and std of cosine similarity between random pairs of last-layer
    hidden state vectors. High mean = collapsed/anisotropic representation.
    """
    flat = _hidden_states(model, calibration_batch, device).float()
    n = flat.shape[0]
    if n < 2:
        return 0.0, 0.0
    g = torch.Generator(device="cpu").manual_seed(rng_seed)
    idx_a = torch.randint(0, n, (n_pairs,), generator=g)
    idx_b = torch.randint(0, n, (n_pairs,), generator=g)
    same = idx_a == idx_b
    if same.any():
        idx_b[same] = (idx_b[same] + 1) % n
    a = flat[idx_a]
    b = flat[idx_b]

    # Handle zero norms which would cause NaNs in cosine similarity
    a_norm = a.norm(dim=-1, keepdim=True)
    b_norm = b.norm(dim=-1, keepdim=True)
    a = torch.where(a_norm == 0, torch.zeros_like(a), a)
    b = torch.where(b_norm == 0, torch.zeros_like(b), b)

    cos = F.cosine_similarity(a, b, dim=-1)

    # Filter out NaNs if any persist
    valid = ~torch.isnan(cos) & ~torch.isinf(cos)
    if not valid.any():
        return 0.0, 0.0

    cos = cos[valid]
    if cos.numel() == 1:
        return float(cos.mean().item()), 0.0

    return float(cos.mean().item()), float(cos.std(unbiased=False).item())


# ---------------------------------------------------------------------------
# 5) N-gram diversity on generated samples
# ---------------------------------------------------------------------------

def _distinct_ngram_ratio(token_lists: List[List[int]], n: int) -> float:
    total = 0
    seen: Counter = Counter()
    for toks in token_lists:
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            seen[tuple(toks[i:i + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    return len(seen) / total


@torch.no_grad()
def ngram_diversity(
    model: torch.nn.Module,
    prompt_input_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 64,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> Dict[str, float]:
    """
    Generate continuations from fixed prompts and measure distinct-n ratios
    for n in {2,3,4}. Lower ratio = more repetition / collapse.
    """
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
    out = model.generate(input_ids=prompt_input_ids, **gen_kwargs)
    prompt_len = prompt_input_ids.shape[1]
    generated = out[:, prompt_len:].cpu().tolist()
    return {
        f"distinct_{n}": _distinct_ngram_ratio(generated, n) for n in (2, 3, 4)
    }


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
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    n_pca_components: int = 64,
    n_cosine_pairs: int = 4096,
    max_new_tokens: int = 64,
) -> Dict[str, float]:
    """Compute every metric and return a flat dict of metric_name -> float."""
    was_training = model.training
    model.eval()
    out: Dict[str, float] = {}

    # 1) perplexity
    out["perplexity"] = perplexity(model, eval_batches, device)
    # 2) attention weight rank
    out["attn_effective_rank"] = attention_weight_rank(model)
    # 3) representation entropy
    out["repr_entropy"] = representation_entropy(
        model, calibration_batch, device, n_components=n_pca_components
    )
    # 4) directional concentration
    cos_mean, cos_std = directional_concentration(
        model, calibration_batch, device, n_pairs=n_cosine_pairs
    )
    out["cos_sim_mean"] = cos_mean
    out["cos_sim_std"] = cos_std
    # 5) n-gram diversity
    diversity = ngram_diversity(
        model, prompt_input_ids, device,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id, eos_token_id=eos_token_id,
    )
    out.update(diversity)

    if was_training:
        model.train()
    return out
