import os
import argparse
import yaml
import torch
import torch.distributed as dist
import numpy as np
import random
import logging
import json
import time
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_distributed():
    """Sets up distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "is_distributed": True,
            "device": torch.device(f"cuda:{local_rank}")
        }
    else:
        return {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "is_distributed": False,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }


def init_logging(config, log_dir, use_wandb=False, wandb_project="grokking-collapse"):
    """Initializes TensorBoard and WandB logging."""
    writer = SummaryWriter(log_dir=log_dir)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project=wandb_project, config=config, dir=log_dir)

    return writer


def compute_fourier_concentration(model, top_k: int = 5) -> float:
    """
    Measure how concentrated the Fourier spectrum is on the top-k frequencies.
    High concentration → grokking has occurred (or is occurring).
    """
    # Using underlying model if wrapped in DDP
    base_model = model.module if hasattr(model, "module") else model
    spectrum = base_model.get_embedding_fourier_spectrum()  # (prime, d_model)
    # Average across embedding dimensions
    avg_spectrum = spectrum.mean(dim=1)  # (prime,)
    # Exclude DC component
    avg_spectrum = avg_spectrum[1:]
    total_energy = avg_spectrum.sum()
    if total_energy < 1e-10:
        return 0.0
    top_energy = avg_spectrum.topk(min(top_k, len(avg_spectrum))).values.sum()
    return (top_energy / total_energy).item()


def evaluate(model, dataloader, device):
    """Evaluate model, return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Use autocast for precision issues if needed as noted in memory
            with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=device.type=='cuda'):
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)

            # Accumulate using float64 as noted in memory for stability
            total_loss += float(loss.item()) * inputs.shape[0]
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += inputs.shape[0]

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default="results/run", help="Output directory")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="grokking-collapse", help="WandB project name")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup distributed
    dist_env = setup_distributed()
    rank = dist_env["rank"]
    device = dist_env["device"]

    # Set seed based on config and rank
    seed = config["training"].get("seed", 42) + rank
    set_seed(seed)

    # Only log on main process
    writer = None
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = init_logging(config, args.output_dir, args.use_wandb, args.wandb_project)
        logger.info(f"Loaded config: {config}")
        logger.info(f"Using device: {device}")

    from src.model import ModularArithmeticTransformer
    from src.data import generate_modular_arithmetic, DatasetConfig

    # Setup dataset config
    data_config = DatasetConfig(
        prime=config["data"].get("vocabulary_size", 59),
        train_fraction=config["data"].get("train_fraction", 0.3),
        collapse_level=config["data"].get("collapse_ratio", 0.0),
        collapse_severity=config["data"].get("collapse_severity", 0.0),
        noise_fraction=config["data"].get("noise_fraction", 0.0),
        seed=seed,
    )

    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(data_config)

    train_dataset = TensorDataset(train_in, train_tgt)
    test_dataset = TensorDataset(test_in, test_tgt)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if dist_env["is_distributed"] else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 512),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        generator=loader_generator if train_sampler is None else None,
    )
    test_loader = DataLoader(test_dataset, batch_size=config["training"].get("batch_size", 512), shuffle=False)

    model = ModularArithmeticTransformer(
        prime=config["data"].get("vocabulary_size", 59),
        d_model=config["model"].get("dim", 128),
        n_heads=config["model"].get("heads", 4),
        d_ff=config["model"].get("d_ff", 512),
        n_layers=config["model"].get("layers", 1),
    ).to(device)

    if dist_env["is_distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist_env["local_rank"]])

    optimizer_name = config["training"].get("optimizer", "AdamW")
    lr = float(config["training"].get("lr", 1e-3))
    weight_decay = float(config["training"].get("weight_decay", 1.0))

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_steps = config["training"].get("steps", 50000)
    eval_every = config["evaluation"].get("eval_every", 100)
    save_every = config["evaluation"].get("checkpoints", 5000)
    metrics_to_log = config["evaluation"].get("metrics", [])

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    dataloader_iter = iter(train_loader)

    history = []
    grokked = False
    grokking_step = None
    grokking_threshold = 0.95
    start_time = time.time()

    loss_history = []
    early_stop_divergence_threshold = 10.0 # If loss is > 10x the minimum seen so far
    min_loss = float('inf')

    for step in range(1, max_steps + 1):
        model.train()

        try:
            inputs, targets = next(dataloader_iter)
        except StopIteration:
            if dist_env["is_distributed"]:
                train_sampler.set_epoch(step)
            dataloader_iter = iter(train_loader)
            inputs, targets = next(dataloader_iter)

        inputs, targets = inputs.to(device), targets.to(device)

        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=device.type=='cuda'):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())

        # Divergence check
        min_loss = min(min_loss, loss_val)
        if step > 100 and loss_val > min_loss * early_stop_divergence_threshold and loss_val > 5.0:
            logger.warning(f"Loss diverged at step {step}. Loss: {loss_val:.4f}, Min Loss: {min_loss:.4f}. Stopping early.")
            break

        if step % eval_every == 0:
            train_loss, train_acc = evaluate(model, train_loader, device)
            test_loss, test_acc = evaluate(model, test_loader, device)

            base_model = model.module if hasattr(model, "module") else model

            weight_norm = base_model.get_weight_norm()
            embedding_rank = base_model.get_embedding_rank()
            fourier_concentration = compute_fourier_concentration(base_model)

            if test_acc >= grokking_threshold and not grokked:
                grokked = True
                grokking_step = step
                if rank == 0:
                    logger.info(f"🎉 GROKKING at step {step}! Test acc: {test_acc:.4f}")

            entry = {
                "step": step,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "weight_norm": weight_norm,
                "embedding_rank": embedding_rank,
                "fourier_concentration": fourier_concentration,
            }
            history.append(entry)

            if rank == 0:
                writer.add_scalar("Loss/Train", train_loss, step)
                writer.add_scalar("Loss/Test", test_loss, step)
                writer.add_scalar("Accuracy/Train", train_acc, step)
                writer.add_scalar("Accuracy/Test", test_acc, step)
                writer.add_scalar("Metrics/WeightNorm", weight_norm, step)
                writer.add_scalar("Metrics/EmbeddingRank", embedding_rank, step)
                writer.add_scalar("Metrics/FourierConcentration", fourier_concentration, step)

                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log(entry)

                elapsed = time.time() - start_time
                logger.info(
                    f"Step {step:5d} | "
                    f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} | "
                    f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} | "
                    f"‖W‖={weight_norm:.2f} rank={embedding_rank:.1f} "
                    f"fourier={fourier_concentration:.3f} | "
                    f"time={elapsed:.1f}s"
                )

        if step % save_every == 0 and rank == 0:
            ckpt_path = output_dir / f"checkpoint_{step}.pt"
            base_model = model.module if hasattr(model, "module") else model
            torch.save({
                "step": step,
                "model_state": base_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)

    if rank == 0:
        results = {
            "config": config,
            "grokked": grokked,
            "grokking_step": grokking_step,
            "final_train_acc": history[-1]["train_acc"] if history else 0.0,
            "final_test_acc": history[-1]["test_acc"] if history else 0.0,
            "final_weight_norm": history[-1]["weight_norm"] if history else 0.0,
            "final_embedding_rank": history[-1]["embedding_rank"] if history else 0.0,
            "final_fourier_concentration": history[-1]["fourier_concentration"] if history else 0.0,
            "history": history,
        }

        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_path}")
        logger.info(f"Grokked: {grokked} at step {grokking_step}")

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        writer.close()

    # Clean up distributed
    if dist_env["is_distributed"]:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
