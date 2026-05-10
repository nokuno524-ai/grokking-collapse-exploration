"""
Prepare clean and AI-contaminated training mixtures.

Pipeline
--------
1. Load OpenWebText, take the first 100K documents, shuffle deterministically,
   split 90/10 into train/test.
2. Tokenize with the GPT-2 tokenizer; clip / pad to MAX_SEQ_LEN.
3. Cache the clean train/test splits once on disk.
4. For each (ratio, seed) combination, replace `ratio` of the train set with
   GPT-2-medium completions of the original document's first 50 tokens, write
   the mixture to data/contaminated/ratio_{pct}/seed_{s}/.
5. Save the clean test set once at data/contaminated/test/.

Run as:  python -m src.contamination.prepare_data \
            --ratios 0 10 30 50 80 100 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

DEFAULT_DATA_ROOT = Path("/scratch/qzp4ta/grokking-collapse/data/contaminated")
N_DOCS = 100_000
TRAIN_FRAC = 0.9
MAX_SEQ_LEN = 512
PROMPT_LEN = 50
GEN_LEN = 512
GEN_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Clean corpus
# ---------------------------------------------------------------------------

def _tokenize_fn(examples, tokenizer):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors=None,
    )
    return enc


def build_clean_splits(
    tokenizer,
    n_docs: int = N_DOCS,
    train_frac: float = TRAIN_FRAC,
    shuffle_seed: int = 1234,
) -> tuple[Dataset, Dataset]:
    """Load OpenWebText, shuffle, slice, tokenize. Returns (train_ds, test_ds)."""
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
# Contamination via GPT-2-medium
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_contaminated_docs(
    train_tok: Dataset,
    n_to_replace: int,
    indices: List[int],
    tokenizer,
    gen_model: GPT2LMHeadModel,
    device: torch.device,
    seed: int,
) -> Dataset:
    """
    For the given `indices` of train_tok, take the first PROMPT_LEN tokens as
    prompt, generate GEN_LEN tokens with GPT-2-medium, and return a Dataset
    of {input_ids, attention_mask} replacements aligned with `indices`.
    """
    gen_model.eval()
    g = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    g.manual_seed(seed)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    replacements = {"input_ids": [], "attention_mask": []}

    for start in range(0, n_to_replace, GEN_BATCH_SIZE):
        batch_idx = indices[start:start + GEN_BATCH_SIZE]
        prompt_ids = []
        for i in batch_idx:
            ids = train_tok[i]["input_ids"][:PROMPT_LEN]
            prompt_ids.append(ids)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        attn = (prompt_tensor != pad_id).long()

        torch.manual_seed(seed + start)
        out = gen_model.generate(
            input_ids=prompt_tensor,
            attention_mask=attn,
            max_new_tokens=GEN_LEN,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        # Truncate / pad to MAX_SEQ_LEN
        for row in out.cpu().tolist():
            row = row[:MAX_SEQ_LEN]
            mask = [1] * len(row)
            if len(row) < MAX_SEQ_LEN:
                row = row + [pad_id] * (MAX_SEQ_LEN - len(row))
                mask = mask + [0] * (MAX_SEQ_LEN - len(mask))
            replacements["input_ids"].append(row)
            replacements["attention_mask"].append(mask)

    return Dataset.from_dict(replacements)


def make_mixture(
    train_tok: Dataset,
    ratio: float,
    seed: int,
    tokenizer,
    gen_model: GPT2LMHeadModel,
    device: torch.device,
) -> Dataset:
    """
    Return a Dataset where `ratio` fraction of train_tok has been replaced
    with GPT-2-medium continuations. Choice of which docs to replace is
    deterministic in (seed, ratio).
    """
    if ratio <= 0:
        return train_tok
    n = len(train_tok)
    n_replace = int(round(ratio * n))
    g = torch.Generator().manual_seed(seed * 100003 + int(ratio * 1000))
    perm = torch.randperm(n, generator=g).tolist()
    replace_idx = sorted(perm[:n_replace])
    keep_idx = sorted(perm[n_replace:])

    replacements = generate_contaminated_docs(
        train_tok, n_replace, replace_idx, tokenizer, gen_model, device, seed
    )
    kept = train_tok.select(keep_idx)
    kept = kept.remove_columns([c for c in kept.column_names
                                if c not in ("input_ids", "attention_mask")])
    mixture = Dataset.from_dict({
        "input_ids": list(kept["input_ids"]) + list(replacements["input_ids"]),
        "attention_mask": list(kept["attention_mask"]) + list(replacements["attention_mask"]),
    })
    # Shuffle the mixture so contaminated rows aren't all at the end.
    mixture = mixture.shuffle(seed=seed)
    return mixture


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[0, 10, 30, 50, 80, 100],
                        help="Contamination ratios in percent (0-100)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--n-docs", type=int, default=N_DOCS)
    parser.add_argument("--gen-model", type=str, default="gpt2-medium")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prepare_data] device={device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    clean_train_path = data_root / "clean_train"
    clean_test_path = data_root / "test"
    if clean_train_path.exists() and clean_test_path.exists():
        print(f"[prepare_data] reusing cached clean splits at {data_root}")
        train_tok = Dataset.load_from_disk(str(clean_train_path))
        test_tok = Dataset.load_from_disk(str(clean_test_path))
    else:
        train_tok, test_tok = build_clean_splits(
            tokenizer, n_docs=args.n_docs, train_frac=TRAIN_FRAC, shuffle_seed=1234
        )
        train_tok.save_to_disk(str(clean_train_path))
        test_tok.save_to_disk(str(clean_test_path))
        print(f"[prepare_data] saved clean train ({len(train_tok)}) and "
              f"test ({len(test_tok)}) to {data_root}")

    # Load generator model
    gen_model = GPT2LMHeadModel.from_pretrained(args.gen_model).to(device)
    gen_model.eval()

    # Build each (ratio, seed) mixture
    for ratio_pct in args.ratios:
        ratio = float(ratio_pct) / 100.0
        for seed in args.seeds:
            out_dir = data_root / f"ratio_{int(ratio_pct)}" / f"seed_{seed}"
            if out_dir.exists() and any(out_dir.iterdir()):
                print(f"[prepare_data] skip existing {out_dir}")
                continue
            print(f"[prepare_data] building ratio={ratio_pct}% seed={seed}")
            mixture = make_mixture(train_tok, ratio, seed, tokenizer, gen_model, device)
            out_dir.mkdir(parents=True, exist_ok=True)
            mixture.save_to_disk(str(out_dir))
            print(f"[prepare_data] wrote {len(mixture)} rows -> {out_dir}")

    print("[prepare_data] done.")


if __name__ == "__main__":
    main()
