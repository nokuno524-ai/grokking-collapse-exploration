import os
import argparse
import glob
import torch
import numpy as np
from typing import List, Tuple, Dict
from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic

def patch_self_attn(model: torch.nn.Module) -> List[Tuple[int, torch.Tensor]]:
    """
    Patches the multi-head attention module to capture and store attention weights.
    Returns a list to which tuples of (layer_idx, attention_weights) will be appended during forward pass.
    """
    extracted_weights = []

    # In ModularArithmeticTransformer, the transformer encoder layers are accessed via model.transformer.layers
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'layers'):
        raise ValueError("Model does not have expected transformer layers structure.")

    for i, layer in enumerate(model.transformer.layers):
        original_forward = layer.self_attn.forward

        def new_forward(self, *args, layer_idx=i, orig_fwd=original_forward, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            # Force eager fallback for SDPA to ensure attention weights are returned
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                attn_output, attn_weights = orig_fwd(*args, **kwargs)
                # Store the weights from this layer. Detach and move to CPU immediately to save GPU memory.
                extracted_weights.append((layer_idx, attn_weights.detach().cpu()))
                return attn_output, attn_weights

        import types
        layer.self_attn.forward = types.MethodType(new_forward, layer.self_attn)

    return extracted_weights

def extract_attention(checkpoint_path: str, dataset_config: DatasetConfig, batch_size: int = 512, device: str = "cpu") -> Dict[str, np.ndarray]:
    """
    Loads a model checkpoint, patches attention, runs it over the test set, and extracts attention weights.
    Chunks processing to prevent memory hazards.
    """
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Initialize model using config from checkpoint if available, or default
    model = ModularArithmeticTransformer()
    if 'model_state_dict' in checkpoint:
        # strict=False can be risky, but we should use strict=True per memory
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.to(device)
    model.eval()

    # Patch attention
    weights_list = patch_self_attn(model)

    # Get test data
    _, _, test_inputs, _ = generate_modular_arithmetic(dataset_config)

    # Create dataloader for chunked evaluation
    dataset = torch.utils.data.TensorDataset(test_inputs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # We will accumulate attention weights across batches
    # Structure: dict[layer_idx] -> list of batches
    layer_weights = {}

    print(f"Running evaluation on {device} to extract attention...")
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            # Clear previous weights
            weights_list.clear()

            # Forward pass will populate weights_list
            _ = model(inputs)

            # Accumulate
            for layer_idx, w in weights_list:
                if layer_idx not in layer_weights:
                    layer_weights[layer_idx] = []
                layer_weights[layer_idx].append(w)

    # Concatenate across batches
    final_weights = {}
    for layer_idx, w_list in layer_weights.items():
        # Shape per batch: (batch_size, n_heads, seq_len, seq_len)
        cat_w = torch.cat(w_list, dim=0).numpy()
        final_weights[f"layer_{layer_idx}"] = cat_w

    return final_weights

def main():
    parser = argparse.ArgumentParser(description="Extract attention matrices from checkpoints.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing checkpoint_*.pt files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save extracted .npz files.")
    parser.add_argument("--prime", type=int, default=59, help="Modulus for the dataset.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for extraction.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup dataset config (we just need test set, so collapse params don't matter much here, but we use defaults)
    config = DatasetConfig(prime=args.prime)

    checkpoints = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pt"))
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    for ckpt_path in checkpoints:
        filename = os.path.basename(ckpt_path)
        step = filename.replace("checkpoint_", "").replace(".pt", "")
        out_path = os.path.join(args.output_dir, f"attn_weights_step_{step}.npz")

        if os.path.exists(out_path):
            print(f"Skipping {ckpt_path}, output already exists.")
            continue

        try:
            attn_matrices = extract_attention(ckpt_path, config, batch_size=args.batch_size, device=args.device)
            np.savez(out_path, **attn_matrices)
            print(f"Saved attention matrices to {out_path}")
        except Exception as e:
            print(f"Error extracting from {ckpt_path}: {e}")

if __name__ == "__main__":
    main()
