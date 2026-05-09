"""
Continue pretraining of GPT-2 medium on a contaminated mixture using LoRA.

The model is loaded with the pretrained GPT-2 medium weights and a LoRA
adapter (rank 16-32) is attached to the attention projections so that the
total number of trainable parameters fits in 24 GB VRAM. Training proceeds
for `--max-steps` steps with AdamW + cosine schedule. Checkpoints are saved
every `--ckpt-every` steps and a comprehensive set of mechanistic metrics is
logged every `--log-every` steps.

Run as
------
python -m src.contamination_real.train_real \
    --ratio 30 --seed 0 --max-steps 50000 \
    --data-root /scratch/qzp4ta/grokking-collapse/data/contaminated_real \
    --output-dir /scratch/qzp4ta/grokking-collapse/results/contamination_real
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
    HAS_PEFT = True
except ImportError:  # peft not installed yet
    HAS_PEFT = False

try:
    from .mechanistic_metrics import (
        GradientTopologyTracker,
        compute_all_metrics,
        snapshot_norms,
    )
except ImportError:  # script invoked outside package
    from src.contamination_real.mechanistic_metrics import (  # type: ignore
        GradientTopologyTracker,
        compute_all_metrics,
        snapshot_norms,
    )


DEFAULT_DATA_ROOT = "/scratch/qzp4ta/grokking-collapse/data/contaminated_real"
DEFAULT_OUTPUT_DIR = "/scratch/qzp4ta/grokking-collapse/results/contamination_real"
LOG_EVERY = 500
CKPT_EVERY = 10000
EVAL_BATCHES = 32
CALIB_BATCH_SIZE = 8
PROMPT_BATCH_SIZE = 8
PROMPT_LEN = 32


# ---------------------------------------------------------------------------
# Reproducibility / housekeeping
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
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


def resolve_train_path(data_root: Path, ratio_pct: int, seed: int, mode: str) -> Path:
    """Locate the dataset directory for a (ratio, seed, mode) tuple.

    Falls back to the clean-train split if the ratio is 0 and no per-seed
    mixture exists.
    """
    candidates = []
    if mode != "ai":
        candidates.append(data_root / f"mode_{mode}" / f"ratio_{ratio_pct}" / f"seed_{seed}")
    candidates.append(data_root / f"ratio_{ratio_pct}" / f"seed_{seed}")
    if ratio_pct == 0:
        candidates.append(data_root / "clean_train")
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    raise FileNotFoundError(
        f"No mixture found for ratio={ratio_pct}% seed={seed} mode={mode}; "
        f"searched: {[str(c) for c in candidates]}"
    )


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
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    calib_indices = list(range(min(CALIB_BATCH_SIZE, len(test_ds))))
    calib_rows = test_ds.select(calib_indices)
    calibration_batch = collate([calib_rows[i] for i in range(len(calib_rows))])

    prompt_rows = test_ds.select(list(range(min(PROMPT_BATCH_SIZE, len(test_ds)))))
    prompt_input_ids = torch.tensor(
        [list(row["input_ids"])[:PROMPT_LEN] for row in prompt_rows],
        dtype=torch.long,
    )
    return train_loader, test_loader, calibration_batch, prompt_input_ids


def take_eval_batches(
    test_loader: DataLoader, n_batches: int
) -> List[Dict[str, torch.Tensor]]:
    out = []
    for i, b in enumerate(test_loader):
        if i >= n_batches:
            break
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    base_model: str = "gpt2-medium"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_lora: bool = True


def build_model(cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    model.config.use_cache = False
    if cfg.use_lora:
        if not HAS_PEFT:
            raise RuntimeError("peft is required for LoRA but is not installed")
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["c_attn", "c_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    model.to(device)
    return model


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
    grad_accum: int,
    lr: float,
    warmup_steps: int,
    weight_decay: float,
    grad_clip: float,
    log_every: int,
    ckpt_every: int,
    base_model: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    mode: str,
    use_amp: bool,
):
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_real] ratio={ratio_pct}% seed={seed} device={device} "
          f"mode={mode}", flush=True)

    train_path = resolve_train_path(data_root, ratio_pct, seed, mode)
    test_path = data_root / "test"
    print(f"[train_real] train_path={train_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, test_loader, calib_batch, prompt_input_ids = build_loaders(
        train_path, test_path, batch_size=batch_size, seed=seed,
    )
    eval_batches_cache = take_eval_batches(test_loader, EVAL_BATCHES)

    cfg = TrainConfig(
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        use_lora=use_lora,
    )
    model = build_model(cfg, device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_real] model params: total={n_total:,} trainable={n_train:,}",
          flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    init_norms = snapshot_norms(model)
    grad_tracker = GradientTopologyTracker(window=8)

    history: List[Dict[str, float]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"ratio_{ratio_pct}_seed_{seed}"
    if mode != "ai":
        run_tag = f"mode-{mode}_{run_tag}"
    results_path = output_dir / f"{run_tag}.json"
    ckpt_dir = output_dir / "ckpt" / run_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def log_metrics(step: int, train_loss: float) -> Dict[str, float]:
        metrics = compute_all_metrics(
            model,
            calibration_batch=calib_batch,
            eval_batches=eval_batches_cache,
            prompt_input_ids=prompt_input_ids,
            device=device,
            init_norms=init_norms,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        metrics["step"] = step
        metrics["train_loss"] = float(train_loss)
        metrics["lr"] = scheduler.get_last_lr()[0]
        return metrics

    def save_ckpt(step: int) -> None:
        sub = ckpt_dir / f"step_{step}"
        sub.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(str(sub))
            tokenizer.save_pretrained(str(sub))
            print(f"[train_real] saved checkpoint -> {sub}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[train_real] checkpoint save failed: {e}", flush=True)

    # Step 0 metrics
    init_metrics = log_metrics(step=0, train_loss=float("nan"))
    history.append(init_metrics)
    print(f"[train_real] step=0 ppl={init_metrics.get('perplexity', float('nan')):.3f} "
          f"rank_last={init_metrics.get('repr_rank_last', 0):.2f}", flush=True)

    step = 0
    micro_step = 0
    start = time.time()
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    model.train()

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        if input_ids.numel() == 0:
            continue

        try:
            with torch.cuda.amp.autocast(
                enabled=use_amp and device.type == "cuda", dtype=torch.float16
            ):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss / grad_accum
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[train_real] non-finite loss at micro_step={micro_step}, skipping",
                      flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            accum_loss += float(loss.item()) * grad_accum
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[train_real] OOM at micro_step={micro_step}; skipping batch",
                  flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue

        micro_step += 1
        if micro_step % grad_accum != 0:
            continue

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        grad_log = grad_tracker.update(model)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        running_loss = accum_loss / grad_accum
        accum_loss = 0.0

        if step % log_every == 0 or step == max_steps:
            metrics = log_metrics(step=step, train_loss=running_loss)
            metrics.update(grad_log)
            metrics["elapsed_s"] = time.time() - start
            history.append(metrics)
            print(
                f"[train_real] step={step} loss={running_loss:.4f} "
                f"ppl={metrics.get('perplexity', float('nan')):.2f} "
                f"rank_last={metrics.get('repr_rank_last', 0):.2f} "
                f"attn_H={metrics.get('attn_entropy_mean', 0):.3f} "
                f"feat_d={metrics.get('feat_density', 0):.0f} "
                f"lora_dB={metrics.get('lora_B_norm_drift', 0):.3f}",
                flush=True,
            )
            with open(results_path, "w") as f:
                json.dump({
                    "ratio_pct": ratio_pct,
                    "seed": seed,
                    "mode": mode,
                    "max_steps": max_steps,
                    "batch_size": batch_size,
                    "grad_accum": grad_accum,
                    "lr": lr,
                    "warmup_steps": warmup_steps,
                    "weight_decay": weight_decay,
                    "use_lora": use_lora,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "base_model": base_model,
                    "history": history,
                }, f, indent=2)

        if ckpt_every > 0 and (step % ckpt_every == 0 or step == max_steps):
            save_ckpt(step)

    print(f"[train_real] done. results -> {results_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=float, required=True,
                        help="Contamination ratio (0-1 or 0-100; >1 treated as percent)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", type=str, default="ai",
                        choices=["ai", "noise", "scarcity", "external", "self"])
    parser.add_argument("--base-model", type=str, default="gpt2-medium")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    parser.add_argument("--no-lora", action="store_true",
                        help="Disable LoRA (full fine-tune; needs more VRAM)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    args = parser.parse_args()

    ratio_pct = int(round(args.ratio if args.ratio > 1 else args.ratio * 100))
    train_one(
        ratio_pct=ratio_pct,
        seed=args.seed,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        base_model=args.base_model,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        mode=args.mode,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
