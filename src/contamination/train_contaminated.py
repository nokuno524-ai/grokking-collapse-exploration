"""
Train GPT-2 small from scratch on a (ratio, seed) contaminated mixture and log
mechanistic metrics every 500 steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
)

try:
    from .metrics import compute_all_metrics
except ImportError:
    from src.contamination.metrics import compute_all_metrics  # type: ignore


DEFAULT_DATA_ROOT = "/scratch/qzp4ta/grokking-collapse/data/contaminated"
DEFAULT_OUTPUT_DIR = "/scratch/qzp4ta/grokking-collapse/results/contamination"
LOG_EVERY = 500
EVAL_BATCHES = 32
CALIB_BATCH_SIZE = 16
PROMPT_BATCH_SIZE = 8
PROMPT_LEN = 16


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_loaders(
    train_path: Path,
    test_path: Path,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, Dict[str, torch.Tensor], torch.Tensor]:
    train_ds = Dataset.load_from_disk(str(train_path))
    test_ds = Dataset.load_from_disk(str(test_path))

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=g,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )

    # Fixed calibration batch (same indices across all runs) drawn from test
    calib_indices = list(range(min(CALIB_BATCH_SIZE, len(test_ds))))
    calib_rows = test_ds.select(calib_indices)
    calibration_batch = collate([calib_rows[i] for i in range(len(calib_rows))])

    # Fixed prompts for n-gram diversity, taken from test set
    prompt_rows = test_ds.select(list(range(min(PROMPT_BATCH_SIZE, len(test_ds)))))
    prompt_input_ids = torch.tensor(
        [row["input_ids"][:PROMPT_LEN] for row in prompt_rows],
        dtype=torch.long,
    )
    return train_loader, test_loader, calibration_batch, prompt_input_ids


def take_eval_batches(test_loader: DataLoader, n_batches: int) -> List[Dict[str, torch.Tensor]]:
    batches = []
    for i, b in enumerate(test_loader):
        if i >= n_batches:
            break
        batches.append(b)
    return batches


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(
    ratio_pct: int,
    seed: int,
    data_root: Path,
    output_dir: Path,
    max_steps: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    weight_decay: float,
    grad_clip: float,
    log_every: int,
):
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] ratio={ratio_pct}% seed={seed} device={device}", flush=True)

    train_path = data_root / f"ratio_{ratio_pct}" / f"seed_{seed}"
    if not train_path.exists():
        # ratio 0 has only one mixture (the clean train), reuse it
        train_path = data_root / "clean_train"
    test_path = data_root / "test"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, test_loader, calib_batch, prompt_input_ids = build_loaders(
        train_path, test_path, batch_size=batch_size, seed=seed,
    )
    eval_batches_cache = take_eval_batches(test_loader, EVAL_BATCHES)

    # GPT-2 small from scratch
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )

    history: List[Dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"ratio_{ratio_pct}_seed_{seed}.json"

    step = 0
    start = time.time()
    train_iter = iter(train_loader)
    model.train()

    # Initial metrics at step 0
    init_metrics = compute_all_metrics(
        model, calib_batch, eval_batches_cache, prompt_input_ids, device,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    init_metrics["step"] = 0
    init_metrics["lr"] = scheduler.get_last_lr()[0]
    history.append(init_metrics)
    print(f"[train] step=0 metrics={init_metrics}", flush=True)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1

        if step % log_every == 0 or step == max_steps:
            metrics = compute_all_metrics(
                model, calib_batch, eval_batches_cache, prompt_input_ids, device,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            metrics["step"] = step
            metrics["train_loss"] = float(loss.item())
            metrics["lr"] = scheduler.get_last_lr()[0]
            metrics["elapsed_s"] = time.time() - start
            history.append(metrics)
            print(f"[train] step={step} train_loss={loss.item():.4f} "
                  f"ppl={metrics['perplexity']:.2f} "
                  f"rank={metrics['attn_effective_rank']:.2f} "
                  f"H={metrics['repr_entropy']:.3f} "
                  f"cos={metrics['cos_sim_mean']:.3f}", flush=True)
            with open(results_path, "w") as f:
                json.dump({
                    "ratio_pct": ratio_pct,
                    "seed": seed,
                    "max_steps": max_steps,
                    "batch_size": batch_size,
                    "lr": lr,
                    "warmup_steps": warmup_steps,
                    "weight_decay": weight_decay,
                    "history": history,
                }, f, indent=2)

    print(f"[train] done. results -> {results_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, required=True,
                        help="Contamination ratio (0-1 or 0-100; >1 treated as percent)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args()

    ratio_pct = int(round(args.ratio if args.ratio > 1 else args.ratio * 100))
    train_one(
        ratio_pct=ratio_pct,
        seed=args.seed,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
