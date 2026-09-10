import pytest
from pathlib import Path
import json
import pandas as pd
from src.sweep.difficulty_sweep import generate_manifest
from src.analysis.difficulty_analysis import analyze_interaction, collect_results

def test_generate_manifest(tmp_path):
    manifest_path, tasks = generate_manifest(tmp_path)

    # Check total length
    # Grid 1: 3 primes * 2 operands * 4 levels * 2 seeds = 48
    # Grid 2: 3 d_models * 4 levels * 2 seeds = 24
    # Total = 72
    assert len(tasks) == 72

    # Check no duplicates
    t_tupled = [tuple(sorted(t.items())) for t in tasks]
    assert len(t_tupled) == len(set(t_tupled))

    # Check manifest file creation
    assert manifest_path.exists()

    with open(manifest_path) as f:
        data = json.load(f)
    assert len(data) == 72


def test_interaction_analysis(tmp_path):
    # Create mock df directly representing results
    df = pd.DataFrame([
        # Baseline (difficulty 0, collapse 0) -> step 1000
        {"type": "difficulty", "prime": 1, "num_operands": 1, "difficulty_score": 1, "collapse_level": 0.0, "seed": 1, "grok_step": 1000},
        {"type": "difficulty", "prime": 1, "num_operands": 1, "difficulty_score": 1, "collapse_level": 0.0, "seed": 2, "grok_step": 1000},
        # High difficulty (difficulty 100, collapse 0) -> step 2000
        {"type": "difficulty", "prime": 10, "num_operands": 2, "difficulty_score": 100, "collapse_level": 0.0, "seed": 1, "grok_step": 2000},
        {"type": "difficulty", "prime": 10, "num_operands": 2, "difficulty_score": 100, "collapse_level": 0.0, "seed": 2, "grok_step": 2000},
        # High collapse (difficulty 0, collapse 0.5) -> step 3000
        {"type": "difficulty", "prime": 1, "num_operands": 1, "difficulty_score": 1, "collapse_level": 0.5, "seed": 1, "grok_step": 3000},
        {"type": "difficulty", "prime": 1, "num_operands": 1, "difficulty_score": 1, "collapse_level": 0.5, "seed": 2, "grok_step": 3000},
        # Both (difficulty 100, collapse 0.5) -> step 10000 (planted huge interaction)
        {"type": "difficulty", "prime": 10, "num_operands": 2, "difficulty_score": 100, "collapse_level": 0.5, "seed": 1, "grok_step": 10000},
        {"type": "difficulty", "prime": 10, "num_operands": 2, "difficulty_score": 100, "collapse_level": 0.5, "seed": 2, "grok_step": 10000},
    ])

    # We just want to see it run without errors and produce a summary
    model = analyze_interaction(df, tmp_path)
    assert model is not None
    assert (tmp_path / "interaction_summary.txt").exists()

    # Since we planted a huge interaction, the coefficient should be positive
    # It's an OLS model, so check the param
    assert "interaction" in model.params
    assert model.params["interaction"] > 0

def test_censoring_logic(tmp_path):
    # Ensure collect_results correctly imputes max_step when grokked is False
    res_dir = tmp_path / "test_res" / "cond1"
    res_dir.mkdir(parents=True)
    with open(res_dir / "results.json", "w") as f:
        json.dump({
            "config": {
                "condition_name": "cond1",
                "prime": 5, "num_operands": 2, "collapse_level": 0.0, "seed": 42, "d_model": 128,
                "max_steps": 10000
            },
            "grokked": False,
            "grokking_step": None
        }, f)

    res_dir2 = tmp_path / "test_res" / "cond2"
    res_dir2.mkdir(parents=True)
    with open(res_dir2 / "results.json", "w") as f:
        json.dump({
            "config": {
                "condition_name": "cond2",
                "prime": 5, "num_operands": 2, "collapse_level": 0.0, "seed": 42, "d_model": 128,
                "max_steps": 10000
            },
            "grokked": True,
            "grokking_step": 1234
        }, f)

    df = collect_results(tmp_path / "test_res")
    assert len(df) == 2

    censored_row = df[df["condition"] == "cond1"].iloc[0]
    assert censored_row["grokked"] == False
    assert censored_row["grok_step"] == 10000

    grokked_row = df[df["condition"] == "cond2"].iloc[0]
    assert grokked_row["grokked"] == True
    assert grokked_row["grok_step"] == 1234
