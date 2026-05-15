"""
Run lm-eval-harness on saved checkpoints from train_real.py.

For each (ratio, seed) checkpoint subdirectory in --output-dir/ckpt/, this
script reattaches the LoRA adapter to the base GPT-2 medium model, materializes
a merged HF model in a tmp dir, and invokes lm-eval-harness via its Python API
on a fixed task list (HellaSwag, ARC-Easy, PIQA, WinoGrande by default).

Results are written next to the per-step checkpoint directory as
`eval_results.json` and aggregated into `output-dir/downstream_summary.json`.

Run as
------
python -m src.contamination_real.eval_downstream \
    --output-dir /scratch/qzp4ta/grokking-collapse/results/contamination_real \
    --tasks hellaswag arc_easy piqa winogrande \
    --batch-size 8 --limit 1000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch

DEFAULT_OUTPUT_DIR = "/scratch/qzp4ta/grokking-collapse/results/contamination_real"
DEFAULT_TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_checkpoints(output_dir: Path) -> List[Path]:
    """Find all per-step checkpoint dirs under `output_dir/ckpt/*/step_*`."""
    ckpt_root = output_dir / "ckpt"
    if not ckpt_root.exists():
        return []
    out = []
    for run_dir in sorted(ckpt_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for step_dir in sorted(run_dir.iterdir()):
            if step_dir.is_dir() and step_dir.name.startswith("step_"):
                out.append(step_dir)
    return out


def merge_lora_to_tmp(
    ckpt_dir: Path,
    base_model: str,
    tmp_root: Path,
) -> Path:
    """If ckpt_dir is a LoRA adapter, merge it back onto `base_model` and
    return a path to the merged model. Otherwise return ckpt_dir unchanged.
    """
    is_lora = (ckpt_dir / "adapter_config.json").exists()
    if not is_lora:
        return ckpt_dir

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[eval] merging LoRA adapter {ckpt_dir} into {base_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(ckpt_dir))
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tmp_root.mkdir(parents=True, exist_ok=True)
    out = tmp_root / f"merged_{ckpt_dir.parent.name}_{ckpt_dir.name}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    return out


def run_lm_eval(
    model_dir: Path,
    tasks: List[str],
    batch_size: int,
    limit: Optional[int],
    device: str,
) -> Dict:
    """Invoke lm-eval-harness's Python API on the merged model directory."""
    from lm_eval import evaluator

    # `lm_eval` expects model_args formatted like 'pretrained=...,dtype=...'
    model_args = f"pretrained={str(model_dir)},dtype=float16"
    print(
        f"[eval] running lm-eval on {model_dir} tasks={tasks} "
        f"limit={limit} batch_size={batch_size}",
        flush=True,
    )
    res = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=0,
        batch_size=batch_size,
        device=device,
        limit=limit,
    )
    return res


def summarize_lm_eval(res: Dict) -> Dict[str, float]:
    """Pull the headline metric per task out of an lm-eval result dict."""
    if not res or "results" not in res:
        return {}
    out = {}
    for task, vals in res["results"].items():
        for k, v in vals.items():
            if isinstance(v, (int, float)) and not k.endswith("_stderr"):
                out[f"{task}/{k}"] = float(v)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", type=str, default="gpt2-medium")
    parser.add_argument("--tasks", type=str, nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit examples per task (for fast smoke-testing)",
    )
    parser.add_argument(
        "--ckpt-glob",
        type=str,
        default=None,
        help="Optional glob to filter checkpoint dirs (e.g. 'ratio_30_*')",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Only evaluate the final checkpoint of each run",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--tmp-root",
        type=str,
        default=str(Path(tempfile.gettempdir()) / "grokking_eval_tmp"),
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    output_dir = Path(args.output_dir)
    tmp_root = Path(args.tmp_root)

    ckpts = discover_checkpoints(output_dir)
    if args.ckpt_glob:
        from fnmatch import fnmatch

        ckpts = [c for c in ckpts if fnmatch(c.parent.name, args.ckpt_glob)]
    if args.final_only:
        # Take the highest-step checkpoint per run
        latest: Dict[str, Path] = {}
        for c in ckpts:
            run = c.parent.name
            try:
                s = int(c.name.split("_")[-1])
            except ValueError:
                s = -1
            cur = latest.get(run)
            if cur is None or int(cur.name.split("_")[-1]) < s:
                latest[run] = c
        ckpts = list(latest.values())

    if not ckpts:
        print(f"[eval] no checkpoints found under {output_dir}/ckpt/", flush=True)
        return

    print(f"[eval] evaluating {len(ckpts)} checkpoints", flush=True)

    summary: Dict[str, Dict] = {}
    for c in ckpts:
        run = c.parent.name
        step = c.name
        out_path = c / "eval_results.json"
        if out_path.exists():
            print(f"[eval] reusing existing results -> {out_path}", flush=True)
            try:
                summary[f"{run}/{step}"] = json.loads(out_path.read_text())
                continue
            except Exception:
                pass

        try:
            merged = merge_lora_to_tmp(c, args.base_model, tmp_root)
            res = run_lm_eval(
                merged, args.tasks, args.batch_size, args.limit, args.device
            )
            metrics = summarize_lm_eval(res)
            payload = {
                "run": run,
                "step": step,
                "tasks": args.tasks,
                "limit": args.limit,
                "metrics": metrics,
            }
            out_path.write_text(json.dumps(payload, indent=2))
            summary[f"{run}/{step}"] = payload
            print(f"[eval] {run}/{step}: {metrics}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[eval] FAILED on {c}: {e}", flush=True)
            summary[f"{run}/{step}"] = {"error": str(e)}

    summary_path = output_dir / "downstream_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[eval] wrote summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
