"""
Prepare real-world contamination mixtures for the full-realism experiment.

Pipeline
--------
1. Stream OpenWebText from HuggingFace, take N documents, split 95/5 into
   train / clean held-out test, tokenize with the GPT-2 tokenizer.
2. Generate AI continuations from a generator model (default: GPT-2 XL) for
   the documents that will be replaced. Continuations are produced from the
   first PROMPT_LEN tokens of the corresponding clean document so length and
   topic distributions are matched.
3. For each contamination ratio in {0, 5, 15, 30, 50, 80} and seed in
   {0, 1, 2}, write an HF Dataset to disk under
   data_root/ratio_{pct}/seed_{s}/.
4. Also support the auxiliary baselines:
   - random-token noise (--mode noise)        (replaces with uniform tokens)
   - data scarcity (--mode scarcity)          (drops contaminated rows)
   - self-contamination (--mode self)         (gen model = trainee model)

Run as
------
python -m src.contamination_real.prepare_real_data \
    --ratios 0 5 15 30 50 80 \
    --seeds 0 1 2 \
    --gen-model gpt2-xl \
    --n-docs 200000

Or to use an existing AI dataset (HC3 / RAID etc.):

python -m src.contamination_real.prepare_real_data \
    --ratios 0 5 15 30 50 80 --seeds 0 1 2 \
    --ai-dataset Hello-SimpleAI/HC3 --ai-text-field chatgpt_answers
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_DATA_ROOT = Path("/scratch/qzp4ta/grokking-collapse/data/contaminated_real")
N_DOCS = 200_000
TRAIN_FRAC = 0.95
MAX_SEQ_LEN = 1024  # may be overridden by --max-seq-len CLI flag
PROMPT_LEN = 64
GEN_LEN = 960
GEN_BATCH_SIZE = 4
DEFAULT_RATIOS = (0, 5, 15, 30, 50, 80)
DEFAULT_SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokenize_fn(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors=None,
    )


def _safe_select(ds: Dataset, indices: List[int]) -> Dataset:
    """`Dataset.select` with bound checking."""
    bounded = [i for i in indices if 0 <= i < len(ds)]
    return ds.select(bounded)


# ---------------------------------------------------------------------------
# Clean splits
# ---------------------------------------------------------------------------


def build_clean_splits(
    tokenizer,
    n_docs: int,
    train_frac: float,
    shuffle_seed: int,
    streaming: bool = True,
) -> tuple[Dataset, Dataset]:
    """Build a deterministic OpenWebText slice and tokenize it.

    Streaming = True is much friendlier on disk space; we just iterate the
    first n_docs documents and materialize them into a single Dataset.
    """
    if streaming:
        raw_iter = load_dataset(
            "openwebtext", split="train", streaming=True, trust_remote_code=True
        )
        texts: List[str] = []
        for i, row in enumerate(raw_iter):
            if i >= n_docs:
                break
            txt = row.get("text") or ""
            if txt:
                texts.append(txt)
        rng = np.random.default_rng(shuffle_seed)
        order = rng.permutation(len(texts))
        texts = [texts[i] for i in order]
        raw = Dataset.from_dict({"text": texts})
    else:
        raw = load_dataset("openwebtext", split="train", trust_remote_code=True)
        raw = raw.shuffle(seed=shuffle_seed)
        raw = raw.select(range(min(n_docs, len(raw))))

    n_train = int(round(train_frac * len(raw)))
    train_raw = raw.select(range(n_train))
    test_raw = raw.select(range(n_train, len(raw)))

    train_tok = train_raw.map(
        lambda b: _tokenize_fn(b, tokenizer),
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Tokenizing train",
    )
    test_tok = test_raw.map(
        lambda b: _tokenize_fn(b, tokenizer),
        batched=True,
        remove_columns=test_raw.column_names,
        desc="Tokenizing test",
    )
    return train_tok, test_tok


# ---------------------------------------------------------------------------
# AI generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_ai_docs(
    train_tok: Dataset,
    indices: List[int],
    tokenizer,
    gen_model,
    device: torch.device,
    seed: int,
    batch_size: Optional[int] = None,
    prompt_len: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> Dataset:
    """Generate AI continuations for the rows referenced by `indices`."""
    # Resolve dynamic defaults so callers pick up CLI overrides.
    if batch_size is None:
        batch_size = GEN_BATCH_SIZE
    if prompt_len is None:
        prompt_len = PROMPT_LEN
    if max_new_tokens is None:
        max_new_tokens = GEN_LEN
    gen_model.eval()
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = eos_id

    out_rows = {"input_ids": [], "attention_mask": []}

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if not batch_idx:
            continue
        prompt_ids = []
        for i in batch_idx:
            ids = list(train_tok[i]["input_ids"])[:prompt_len]
            if len(ids) < prompt_len:
                ids = ids + [pad_id] * (prompt_len - len(ids))
            prompt_ids.append(ids)

        prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        attn = (prompt_t != pad_id).long()
        attn[:, 0] = 1  # ensure at least one attended position per row

        torch.manual_seed(seed * 1_000_003 + start)

        try:
            out = gen_model.generate(
                input_ids=prompt_t,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            half = max(1, len(batch_idx) // 2)
            print(
                f"[prepare_real_data] OOM at batch {start}; "
                f"retrying with batch={half}",
                flush=True,
            )
            return _generate_recursively(
                train_tok,
                indices,
                tokenizer,
                gen_model,
                device,
                seed,
                batch_size=half,
                prompt_len=prompt_len,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        for row in out.detach().cpu().tolist():
            row = row[:MAX_SEQ_LEN]
            mask = [1] * len(row)
            if len(row) < MAX_SEQ_LEN:
                row = row + [pad_id] * (MAX_SEQ_LEN - len(row))
                mask = mask + [0] * (MAX_SEQ_LEN - len(mask))
            out_rows["input_ids"].append(row)
            out_rows["attention_mask"].append(mask)

    return Dataset.from_dict(out_rows)


def _generate_recursively(*args, **kwargs) -> Dataset:
    """Used after OOM to retry with a smaller batch size."""
    return generate_ai_docs(*args, **kwargs)


# ---------------------------------------------------------------------------
# Other contamination modes (random-token noise, scarcity, self-contam)
# ---------------------------------------------------------------------------


def make_noise_replacements(
    n: int,
    tokenizer,
    seed: int,
) -> Dataset:
    """Replace contaminated rows with uniformly random token sequences.

    Forms the matched 'random noise' baseline: same fraction of the train set
    is corrupted, but the corruption has no language structure at all.
    """
    rng = np.random.default_rng(seed * 7919 + 11)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    vocab = tokenizer.vocab_size
    rows = []
    masks = []
    for _ in range(n):
        ids = rng.integers(0, vocab, size=MAX_SEQ_LEN, dtype=np.int64).tolist()
        rows.append(ids)
        masks.append([1] * MAX_SEQ_LEN)
    _ = pad_id  # silence linter
    return Dataset.from_dict({"input_ids": rows, "attention_mask": masks})


def make_external_replacements(
    n: int,
    ai_dataset: str,
    ai_text_field: str,
    tokenizer,
    seed: int,
) -> Dataset:
    """Take n samples from an external AI-text dataset (e.g. HC3, RAID)."""
    ds = load_dataset(ai_dataset, split="train", trust_remote_code=True)
    if ai_text_field not in ds.column_names:
        raise ValueError(
            f"Field '{ai_text_field}' not found in dataset {ai_dataset}; "
            f"available: {ds.column_names}"
        )
    ds = ds.shuffle(seed=seed)

    texts: List[str] = []
    for row in ds:
        v = row[ai_text_field]
        if isinstance(v, list):
            v = " ".join(str(x) for x in v if x)
        if isinstance(v, str) and v.strip():
            texts.append(v)
        if len(texts) >= n:
            break
    if len(texts) < n:
        # Repeat to fill
        while len(texts) < n:
            texts += texts[: max(1, n - len(texts))]
        texts = texts[:n]

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors=None,
    )
    return Dataset.from_dict(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
    )


# ---------------------------------------------------------------------------
# Mixture builder
# ---------------------------------------------------------------------------


def make_mixture(
    train_tok: Dataset,
    ratio: float,
    seed: int,
    tokenizer,
    gen_model,
    device: torch.device,
    mode: str = "ai",
    ai_dataset: Optional[str] = None,
    ai_text_field: Optional[str] = None,
) -> Dataset:
    """Build a contamination mixture for a given ratio and seed.

    mode = "ai": generate continuations with `gen_model`.
    mode = "noise": replace with uniform random tokens.
    mode = "scarcity": just drop contaminated rows (training set shrinks).
    mode = "external": draw from `ai_dataset[ai_text_field]`.
    mode = "self": same as "ai" but `gen_model` is the trainee model.
    """
    if ratio <= 0:
        return train_tok
    n = len(train_tok)
    n_replace = int(round(ratio * n))
    g = torch.Generator().manual_seed(seed * 100003 + int(ratio * 10_000))
    perm = torch.randperm(n, generator=g).tolist()
    replace_idx = sorted(perm[:n_replace])
    keep_idx = sorted(perm[n_replace:])

    kept = train_tok.select(keep_idx)
    kept = kept.remove_columns(
        [c for c in kept.column_names if c not in ("input_ids", "attention_mask")]
    )

    if mode == "scarcity":
        return kept.shuffle(seed=seed)

    if mode == "noise":
        repl = make_noise_replacements(n_replace, tokenizer, seed)
    elif mode == "external":
        if not ai_dataset or not ai_text_field:
            raise ValueError(
                "--ai-dataset and --ai-text-field required for mode=external"
            )
        repl = make_external_replacements(
            n_replace, ai_dataset, ai_text_field, tokenizer, seed
        )
    else:
        if gen_model is None:
            raise ValueError("Generator model required for AI / self contamination")
        repl = generate_ai_docs(
            train_tok, replace_idx, tokenizer, gen_model, device, seed
        )

    mixture = Dataset.from_dict(
        {
            "input_ids": list(kept["input_ids"]) + list(repl["input_ids"]),
            "attention_mask": list(kept["attention_mask"])
            + list(repl["attention_mask"]),
        }
    )
    return mixture.shuffle(seed=seed)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    # Declared up-front so the parser defaults (which reference these globals)
    # are valid before the override block reassigns them.
    global MAX_SEQ_LEN, PROMPT_LEN, GEN_LEN, GEN_BATCH_SIZE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_RATIOS),
        help="Contamination ratios in percent (0-100)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--n-docs", type=int, default=N_DOCS)
    parser.add_argument(
        "--gen-model",
        type=str,
        default="gpt2-xl",
        help="HF model used to generate AI text",
    )
    parser.add_argument("--tokenizer", type=str, default="gpt2-medium")
    parser.add_argument(
        "--mode",
        type=str,
        default="ai",
        choices=["ai", "noise", "scarcity", "external", "self"],
    )
    parser.add_argument(
        "--ai-dataset",
        type=str,
        default=None,
        help="HF dataset for mode=external (e.g. Hello-SimpleAI/HC3)",
    )
    parser.add_argument("--ai-text-field", type=str, default=None)
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not stream OpenWebText (load full split)",
    )
    parser.add_argument("--gen-batch-size", type=int, default=GEN_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=GEN_LEN)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        help="Max sequence length for tokenization & generation outputs",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=PROMPT_LEN,
        help="Prompt length used to seed AI continuations",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    set_seed(0)

    # Override module-level constants so downstream helpers see the new sizes.
    MAX_SEQ_LEN = int(args.max_seq_len)
    PROMPT_LEN = min(int(args.prompt_len), max(1, MAX_SEQ_LEN // 4))
    GEN_LEN = max(1, min(int(args.max_new_tokens), MAX_SEQ_LEN - PROMPT_LEN))
    GEN_BATCH_SIZE = int(args.gen_batch_size)

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prepare_real_data] device={device} mode={args.mode}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    clean_train_path = data_root / "clean_train"
    clean_test_path = data_root / "test"
    if clean_train_path.exists() and clean_test_path.exists():
        print(
            f"[prepare_real_data] reusing cached clean splits at {data_root}",
            flush=True,
        )
        train_tok = Dataset.load_from_disk(str(clean_train_path))
        test_tok = Dataset.load_from_disk(str(clean_test_path))
    else:
        train_tok, test_tok = build_clean_splits(
            tokenizer,
            n_docs=args.n_docs,
            train_frac=TRAIN_FRAC,
            shuffle_seed=1234,
            streaming=not args.no_stream,
        )
        train_tok.save_to_disk(str(clean_train_path))
        test_tok.save_to_disk(str(clean_test_path))
        print(
            f"[prepare_real_data] saved clean train ({len(train_tok)}) "
            f"test ({len(test_tok)}) -> {data_root}",
            flush=True,
        )

    gen_model = None
    if args.mode in ("ai", "self"):
        print(f"[prepare_real_data] loading generator: {
                args.gen_model}", flush=True)
        # fp16 only on CUDA; fp16 on CPU silently upcasts and is slow.
        gen_dtype = torch.float16 if device.type == "cuda" else torch.float32
        gen_model = AutoModelForCausalLM.from_pretrained(
            args.gen_model, torch_dtype=gen_dtype
        ).to(device)
        gen_model.eval()

    config_record = {
        "mode": args.mode,
        "ratios_pct": args.ratios,
        "seeds": args.seeds,
        "gen_model": args.gen_model if args.mode in ("ai", "self") else None,
        "ai_dataset": args.ai_dataset,
        "ai_text_field": args.ai_text_field,
        "n_docs": args.n_docs,
        "max_seq_len": MAX_SEQ_LEN,
        "tokenizer": args.tokenizer,
    }
    (data_root / "config.json").write_text(json.dumps(config_record, indent=2))

    for ratio_pct in args.ratios:
        ratio = float(ratio_pct) / 100.0
        for seed in args.seeds:
            sub = f"ratio_{int(ratio_pct)}/seed_{seed}"
            if args.mode != "ai":
                sub = f"mode_{args.mode}/" + sub
            out_dir = data_root / sub
            if out_dir.exists() and any(out_dir.iterdir()):
                print(f"[prepare_real_data] skip existing {out_dir}", flush=True)
                continue
            print(
                f"[prepare_real_data] building ratio={ratio_pct}% seed={seed} " f"mode={
                    args.mode}", flush=True
            )
            mixture = make_mixture(
                train_tok,
                ratio,
                seed,
                tokenizer,
                gen_model,
                device,
                mode=args.mode,
                ai_dataset=args.ai_dataset,
                ai_text_field=args.ai_text_field,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            mixture.save_to_disk(str(out_dir))
            print(f"[prepare_real_data] wrote {
                    len(mixture)} rows -> {out_dir}", flush=True)

    print("[prepare_real_data] done.", flush=True)


if __name__ == "__main__":
    main()
