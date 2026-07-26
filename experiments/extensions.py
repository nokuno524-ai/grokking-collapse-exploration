import torch
import numpy as np
from src.data import DatasetConfig, generate_modular_arithmetic
from src.model import ModularArithmeticTransformer

def setup_partial_collapse_experiment(prime=59, pure_fraction=0.5, synthetic_fraction=0.5):
    """
    Setup an experiment where only *some* of the training data is synthetic,
    while the rest is pure.
    """
    # 1. Generate full pure dataset
    config = DatasetConfig(prime=prime, train_fraction=0.3, collapse_level=0.0)
    train_in_pure, train_tgt_pure, test_in, test_tgt = generate_modular_arithmetic(config)

    # 2. Generate full collapsed dataset
    config_collapse = DatasetConfig(prime=prime, train_fraction=0.3, collapse_level=1.0)
    train_in_syn, train_tgt_syn, _, _ = generate_modular_arithmetic(config_collapse)

    # Mix them based on fractions
    num_train = len(train_in_pure)
    num_pure = int(num_train * pure_fraction)
    num_syn = int(num_train * synthetic_fraction)

    mixed_in = torch.cat([train_in_pure[:num_pure], train_in_syn[:num_syn]])
    mixed_tgt = torch.cat([train_tgt_pure[:num_pure], train_tgt_syn[:num_syn]])

    # Shuffle
    indices = torch.randperm(len(mixed_in))
    mixed_in = mixed_in[indices]
    mixed_tgt = mixed_tgt[indices]

    return mixed_in, mixed_tgt, test_in, test_tgt

def setup_recovery_experiment(checkpoint_path, new_pure_samples):
    """
    Can a collapsed model grokk with additional clean data?
    Loads a collapsed checkpoint and prepares a new dataloader with clean data.
    """
    model = ModularArithmeticTransformer()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict["model_state"] if "model_state" in state_dict else state_dict)

    # Generate pure dataset
    config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    # Return the model to resume training and the clean data
    return model, train_in, train_tgt, test_in, test_tgt

def setup_transfer_experiment(checkpoint_path, new_prime=61):
    """
    Do grokked features transfer to new tasks?
    E.g., transfer from mod 59 to mod 61.
    """
    model = ModularArithmeticTransformer(prime=59)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict["model_state"] if "model_state" in state_dict else state_dict)

    # Replace output head and token embed for new prime
    old_d_model = model.d_model
    model.prime = new_prime
    model.token_embed = torch.nn.Embedding(new_prime, old_d_model)
    model.output_head = torch.nn.Linear(old_d_model, new_prime)
    # init new weights
    torch.nn.init.normal_(model.token_embed.weight, std=0.02)
    torch.nn.init.normal_(model.output_head.weight, std=0.02)
    torch.nn.init.zeros_(model.output_head.bias)

    # Generate data for new prime
    config = DatasetConfig(prime=new_prime, train_fraction=0.3, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    return model, train_in, train_tgt, test_in, test_tgt

def setup_scale_experiment(d_model, n_layers, n_heads):
    """
    Does collapse-grokking interaction change with model width/depth?
    """
    model = ModularArithmeticTransformer(
        prime=59,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4
    )

    # Run data with some standard collapse level
    config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.5)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    return model, train_in, train_tgt, test_in, test_tgt

def main():
    print("Testing partial collapse setup...")
    train_in, train_tgt, test_in, test_tgt = setup_partial_collapse_experiment(pure_fraction=0.5, synthetic_fraction=0.5)
    print(f"Partial collapse dataset size: {len(train_in)} train, {len(test_in)} test")

    print("Testing recovery experiment setup...")
    try:
        model, tr_i, tr_t, te_i, te_t = setup_recovery_experiment("results/high_collapse/checkpoint_50000.pt", new_pure_samples=1000)
        print("Recovery model loaded and dataset created.")
    except Exception as e:
        print(f"Skipped due to missing checkpoint: {e}")

    print("Testing transfer experiment setup...")
    try:
        model, tr_i, tr_t, te_i, te_t = setup_transfer_experiment("results/pure/checkpoint_50000.pt", new_prime=61)
        print(f"Transfer model adapted for prime {model.prime}, new head size {model.output_head.weight.shape}")
    except Exception as e:
        print(f"Skipped due to missing checkpoint: {e}")

    print("Testing scale experiment setup...")
    model, tr_i, tr_t, te_i, te_t = setup_scale_experiment(d_model=256, n_layers=2, n_heads=8)
    print(f"Scale model created: {model.d_model} d_model, {model.transformer.num_layers} layers")

if __name__ == "__main__":
    main()
