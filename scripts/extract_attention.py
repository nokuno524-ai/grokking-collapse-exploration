"""
Script to extract attention weights and metrics from checkpoints across different collapse severities.
"""

import os
import argparse
import torch
from pathlib import Path
from src.model import ModularArithmeticTransformer

def get_all_layers_attention_weights(model: ModularArithmeticTransformer, x: torch.Tensor) -> list[torch.Tensor]:
    """
    Extracts attention weights for all transformer layers using forward hooks.
    Returns a list of attention weights tensors, one for each layer.
    Shape of each tensor: (B, n_heads, T, T)
    """
    all_attn_weights = []
    hooks = []

    # Temporarily patch MultiheadAttention.forward to return per-head weights
    def custom_forward(self, query, key, value, *args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False
        return self._original_forward(query, key, value, *args, **kwargs)

    for layer in model.transformer.layers:
        layer.self_attn._original_forward = layer.self_attn.forward
        layer.self_attn.forward = custom_forward.__get__(layer.self_attn)

        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                all_attn_weights.append(output[1].clone())
        hooks.append(layer.self_attn.register_forward_hook(hook))

    # Run forward pass
    model(x)

    # Remove hooks and restore forward methods
    for h in hooks:
        h.remove()

    for layer in model.transformer.layers:
        layer.self_attn.forward = layer.self_attn._original_forward
        del layer.self_attn._original_forward

    return all_attn_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="analysis/attention")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
    results_dir = Path("results")

    # Probe sequence
    torch.manual_seed(42)
    probe_x = torch.randint(0, 59, (32, 2))

    attention_data = {}

    for condition in conditions:
        cond_dir = results_dir / condition
        if not cond_dir.exists():
            print(f"Skipping {condition}, directory not found.")
            continue

        print(f"Processing condition: {condition}")
        attention_data[condition] = {}

        # Sort checkpoints by step
        checkpoints = list(cond_dir.glob("checkpoint_*.pt"))
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))

        for ckpt_path in checkpoints:
            step = int(ckpt_path.stem.split("_")[1])

            # Load model
            model = ModularArithmeticTransformer()
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "model_state" in state_dict:
                state_dict = state_dict["model_state"]

            # Remove 'module.' prefix if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()

            with torch.no_grad():
                attn_weights_list = get_all_layers_attention_weights(model, probe_x)

            attention_data[condition][step] = attn_weights_list

    # Save the extracted data
    torch.save(attention_data, output_dir / "extracted_attention.pt")
    print(f"Saved extracted attention to {output_dir / 'extracted_attention.pt'}")

if __name__ == "__main__":
    main()
