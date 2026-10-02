import os
import json
import tempfile
import argparse
from pathlib import Path
from unittest.mock import patch

from src.orchestrator import cmd_run, cmd_report, get_config_hash, update_registry, get_registry_hash


def test_orchestrator_smoke_and_report():
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)

        args = argparse.Namespace(
            smoke=True,
            max_steps=2,
            output_dir=str(out_dir)
        )

        # 1. Test smoke run
        cmd_run(args)

        # Verify state is updated
        state_path = out_dir / "orchestrator_state.json"
        assert state_path.exists()
        with open(state_path) as f:
            state = json.load(f)
            assert len(state["completed_tasks"]) == 1
            assert state["completed_tasks"][0] == "level0_sev0.3_seed42"

        # Verify registry
        registry_path = out_dir / "registry.json"
        assert registry_path.exists()

        # 2. Test report generation
        args_report = argparse.Namespace(
            output_dir=str(out_dir),
            force=False
        )
        cmd_report(args_report)


        assert (out_dir / "report" / "manifest.json").exists()


def test_crash_resume():
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)
        args = argparse.Namespace(
            smoke=True,
            max_steps=2,
            output_dir=str(out_dir)
        )

        # First run partially
        # We can fake a partial run by writing a checkpoint
        pair_dir = out_dir / "level0_sev0.3" / "seed_42"
        pair_dir.mkdir(parents=True, exist_ok=True)
        # touch checkpoint
        import torch
        from src.train import TrainConfig
        from src.model import ModularArithmeticTransformer
        config = TrainConfig(prime=7)
        model = ModularArithmeticTransformer(prime=7, d_model=128, n_heads=4, d_ff=512, n_layers=1)
        optimizer = torch.optim.AdamW(model.parameters())
        torch.save({"step": 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": {}, "state": {}}, pair_dir / "checkpoint_1.pt")


        # Run it
        cmd_run(args)

        # Verify it created results
        assert (pair_dir / "results.json").exists()


def test_staleness_detection(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)
        registry_path = out_dir / "registry.json"
        registry_path.write_text('{"hash1": {"config": {}, "metrics": {"grokked": false, "grokking_step": null}}}')

        args = argparse.Namespace(
            output_dir=str(out_dir),
            force=False
        )

        # First report should generate
        cmd_report(args)
        captured = capsys.readouterr()
        assert "Generating reports..." in captured.out

        # Second report should skip
        cmd_report(args)
        captured = capsys.readouterr()
        assert "Reports are up-to-date. Skipping regeneration." in captured.out

        # Modify registry
        registry_path.write_text('{"hash1": {"config": {}, "metrics": {"grokked": false, "grokking_step": null}}, "hash2": {"config": {}, "metrics": {"grokked": true, "grokking_step": 100}}}')

        # Third report should regenerate
        cmd_report(args)
        captured = capsys.readouterr()
        assert "Staleness detected: registry has changed. Regenerating reports..." in captured.out

        # Run with force
        cmd_report(argparse.Namespace(output_dir=str(out_dir), force=True))
        captured = capsys.readouterr()
        assert "Staleness detected" in captured.out or "Regenerating reports" in captured.out
