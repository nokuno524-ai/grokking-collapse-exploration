import os
import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.model import ModularArithmeticTransformer

def ablate_head_hook(head_idx, n_heads):
    def hook(module, input, output):
        # The output of self_attn is (batch_size, seq_len, d_model)
        # We need to zero out the contribution of head_idx.
        # But out_proj is applied after concat of heads.
        # In PyTorch's MultiheadAttention, output = linear(concat(heads)).
        # We can't easily intercept the pre-out_proj activations with standard hooks,
        # but we can monkey-patch the out_proj weight for the duration of the pass.
        pass # implemented differently below
    return hook

def evaluate_ablation(model, head_idx, eval_data, eval_labels):
    model.eval()

    # Store original weights
    orig_weight = model.transformer.layers[0].self_attn.out_proj.weight.data.clone()

    # Zero out the head
    d_model = model.d_model
    n_heads = model.n_heads
    head_dim = d_model // n_heads

    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim

    with torch.no_grad():
        model.transformer.layers[0].self_attn.out_proj.weight.data[:, start_idx:end_idx] = 0

        logits = model(eval_data)
        preds = logits.argmax(dim=-1)
        acc = (preds == eval_labels).float().mean().item()

        # Restore weights
        model.transformer.layers[0].self_attn.out_proj.weight.data = orig_weight

    return acc

def track_circuit_formation(condition):
    result_dir = f"results/{condition}"

    if not os.path.exists(result_dir):
        return None

    with open(os.path.join(result_dir, "results.json"), "r") as f:
        res = json.load(f)
        config = res["config"]

    checkpoints = []
    for f in os.listdir(result_dir):
        if f.startswith("checkpoint_") and f.endswith(".pt"):
            step = int(f.split("_")[1].split(".")[0])
            checkpoints.append((step, os.path.join(result_dir, f)))
    checkpoints.sort()

    model = ModularArithmeticTransformer(
        prime=config.get("prime", 59),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 1)
    )

    # Generate eval data
    torch.manual_seed(42)
    prime = config.get("prime", 59)
    eval_data = torch.randint(0, prime, (1024, 2))
    eval_labels = (eval_data[:, 0] + eval_data[:, 1]) % prime

    results = {"step": [], "clean_acc": []}
    for h in range(config.get("n_heads", 4)):
        results[f"head_{h}_ablated_acc"] = []
        results[f"head_{h}_effect"] = []

    for step, ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        with torch.no_grad():
            clean_logits = model(eval_data)
            clean_preds = clean_logits.argmax(dim=-1)
            clean_acc = (clean_preds == eval_labels).float().mean().item()

        results["step"].append(step)
        results["clean_acc"].append(clean_acc)

        for h in range(config.get("n_heads", 4)):
            ablated_acc = evaluate_ablation(model, h, eval_data, eval_labels)
            results[f"head_{h}_ablated_acc"].append(ablated_acc)
            results[f"head_{h}_effect"].append(clean_acc - ablated_acc)

    return results

def run_analysis():
    conditions = ["pure", "medium_collapse"]
    all_results = {}

    for cond in conditions:
        print(f"Tracking circuit formation for {cond}...")
        res = track_circuit_formation(cond)
        if res is not None:
            all_results[cond] = res

            # Save JSON
            with open(f"results/circuit_formation_{cond}.json", "w") as f:
                json.dump(res, f, indent=2)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(res["step"], res["clean_acc"], 'k--', label="Clean Acc", linewidth=2)

            for h in range(4): # assuming 4 heads
                plt.plot(res["step"], res[f"head_{h}_effect"], label=f"Head {h} Importance")

            plt.title(f"Attention Head Importance Over Time ({cond})")
            plt.xlabel("Step")
            plt.ylabel("Accuracy Drop (Clean - Ablated)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"results/circuit_formation_{cond}.png", dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    run_analysis()
