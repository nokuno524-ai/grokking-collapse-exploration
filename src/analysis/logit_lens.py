import torch
import torch.nn.functional as F
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic

def logit_lens_analysis(model: ModularArithmeticTransformer, dataloader, device):
    """
    Perform logit lens analysis by decoding intermediate representations.
    For a 1-layer model, we decode:
    - After embedding addition (tok + pos)
    - After attention (but before MLP)
    - After MLP (final output)
    """
    model.eval()

    # Store predictions at each stage
    # shape: (n_stages, total_samples)
    stages = ["embeddings", "attention_out", "mlp_out"]
    accuracies = {stage: 0.0 for stage in stages}
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # Step 1: Embeddings
            tok = model.token_embed(inputs)
            positions = torch.arange(2, device=device).unsqueeze(0).expand(batch_size, -1)
            pos = model.pos_embed(positions)
            h_0 = tok + pos

            # Decode h_0
            h_0_pooled = model.ln(h_0).mean(dim=1)
            logits_0 = model.output_head(h_0_pooled)
            preds_0 = logits_0.argmax(dim=-1)
            accuracies["embeddings"] += (preds_0 == targets).sum().item()

            # Step 2: Attention
            layer = model.transformer.layers[0]
            # PyTorch's TransformerEncoderLayer doesn't easily expose intermediate attention.
            # We can manually do it or use forward with hooks.
            # Easiest is to manually apply attention.
            # Alternatively, since it's just 1 layer, h_1 = layer(h_0) is the MLP out.
            # Let's use a hook to get the attention output.
            attn_out = None
            def hook(module, input, output):
                nonlocal attn_out
                # output is typically the output of the module
                attn_out = output[0] if isinstance(output, tuple) else output

            # We hook the first part of the transformer layer (the attention + dropout/add/norm)
            # But nn.TransformerEncoderLayer combines them.
            # We can just look at intermediate by manually doing:
            # h = h + self_attn(h, h, h)
            # h = norm1(h)
            h_attn = layer.self_attn(h_0, h_0, h_0, need_weights=False)[0]
            h_1 = h_0 + layer.dropout1(h_attn)
            h_1_norm = layer.norm1(h_1)

            # Decode h_1_norm
            h_1_pooled = model.ln(h_1_norm).mean(dim=1)
            logits_1 = model.output_head(h_1_pooled)
            preds_1 = logits_1.argmax(dim=-1)
            accuracies["attention_out"] += (preds_1 == targets).sum().item()

            # Step 3: MLP (Final out)
            # continue the forward pass
            h_2 = h_1_norm + layer.dropout2(layer.linear2(F.gelu(layer.linear1(h_1_norm))))
            h_2_norm = layer.norm2(h_2) # This is usually not present in pre-LN?
            # Wait, PyTorch default is post-LN, but model.forward has its own LN.
            # Let's just use the full layer forward.
            h_full = model.transformer(h_0)
            h_full_pooled = model.ln(h_full).mean(dim=1)
            logits_full = model.output_head(h_full_pooled)
            preds_full = logits_full.argmax(dim=-1)
            accuracies["mlp_out"] += (preds_full == targets).sum().item()

            total += batch_size

    for stage in stages:
        accuracies[stage] /= total

    return accuracies

def track_logit_lens(run_dir: str, prime: int = 59):
    """
    Tracks logit lens accuracies over training.
    """
    run_dir = Path(run_dir)
    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda x: int(x.stem.split('_')[1]))

    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = DatasetConfig(prime=prime)
    _, _, test_in, test_tgt = generate_modular_arithmetic(config)
    from torch.utils.data import TensorDataset, DataLoader
    test_dataset = TensorDataset(test_in, test_tgt)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    results = []

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_config = ckpt.get("config", {})

        model = ModularArithmeticTransformer(
            prime=model_config.get("prime", prime),
            d_model=model_config.get("d_model", 128),
            n_heads=model_config.get("n_heads", 4),
            d_ff=model_config.get("d_ff", 512),
            n_layers=model_config.get("n_layers", 1),
        ).to(device)

        model_state = ckpt["model_state"]
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
        model.load_state_dict(model_state)

        step = ckpt["step"]

        accs = logit_lens_analysis(model, test_loader, device)

        results.append({
            "step": step,
            **accs
        })

        print(f"Step {step}: Embed: {accs['embeddings']:.4f}, Attn: {accs['attention_out']:.4f}, MLP: {accs['mlp_out']:.4f}")

    out_path = run_dir / "logit_lens_tracking.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved logit lens tracking to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Path to run directory with checkpoints")
    args = parser.parse_args()

    if args.run_dir:
        track_logit_lens(args.run_dir)
    else:
        print("No run dir provided, doing a dry run...")
        model = ModularArithmeticTransformer()
        device = torch.device("cpu")
        config = DatasetConfig()
        _, _, test_in, test_tgt = generate_modular_arithmetic(config)
        from torch.utils.data import TensorDataset, DataLoader
        test_dataset = TensorDataset(test_in, test_tgt)
        test_loader = DataLoader(test_dataset, batch_size=32)

        accs = logit_lens_analysis(model, test_loader, device)
        print("Dry run logit lens:", accs)
