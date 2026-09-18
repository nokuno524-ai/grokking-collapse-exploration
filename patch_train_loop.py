import re

with open("src/train.py", "r") as f:
    content = f.read()

train_data_setup = """
    # Set seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    def create_loader(collapse_level):
        data_config = DatasetConfig(
            prime=config.prime,
            train_fraction=config.train_fraction,
            collapse_level=collapse_level,
            collapse_severity=config.collapse_severity,
            noise_fraction=config.noise_fraction,
            seed=config.seed,
        )
        train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(data_config)
        train_dataset = TensorDataset(train_in, train_tgt)
        test_dataset = TensorDataset(test_in, test_tgt)

        loader_generator = torch.Generator()
        loader_generator.manual_seed(config.seed)

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            generator=loader_generator,
        )
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        return train_loader, test_loader

    # Loaders for phase A and B, depending on curriculum mode
    if config.curriculum_mode == "none":
        train_loader, test_loader = create_loader(config.collapse_level)
        train_loader_A = train_loader_B = train_loader
    elif config.curriculum_mode == "clean_to_collapse":
        train_loader_A, test_loader = create_loader(0.0)
        train_loader_B, _ = create_loader(config.collapse_level)
    elif config.curriculum_mode == "collapse_to_clean":
        train_loader_A, test_loader = create_loader(config.collapse_level)
        train_loader_B, _ = create_loader(0.0)
    elif config.curriculum_mode == "interleaved":
        train_loader_A, test_loader = create_loader(0.0)
        train_loader_B, _ = create_loader(config.collapse_level)
    else:
        raise ValueError(f"Unknown curriculum mode: {config.curriculum_mode}")

    # Use test_loader from phase A (they share the same test set because seed is identical and split is deterministic)

    def get_batch(step, iter_A, iter_B):
        if config.curriculum_mode == "none":
            active_iter = iter_A
        elif config.curriculum_mode in ["clean_to_collapse", "collapse_to_clean"]:
            active_iter = iter_A if step <= config.switch_step else iter_B
        elif config.curriculum_mode == "interleaved":
            # Interleave based on ratio, handled probabilistically or by strict step
            # Simple alternating:
            active_iter = iter_A if (step % 2 == 0) else iter_B
            # If interleaved_ratio != 0.5 we can do proper fraction, but let's stick to step-based interleaving for simplicity
            # e.g. step % 10 < config.interleaved_ratio * 10
            cycle = 10
            active_iter = iter_A if (step % cycle) < (config.interleaved_ratio * cycle) else iter_B

        try:
            return next(active_iter)
        except StopIteration:
            # Re-initialize the active iterator when it runs out
            pass

        return None
"""

# Replace the data loading logic
content = re.sub(
    r"    # Set seeds.*    # Create model",
    train_data_setup + "\n    # Create model",
    content,
    flags=re.DOTALL
)

train_loop = """
    # Training loop
    iter_A = iter(train_loader_A)
    iter_B = iter(train_loader_B)
    start_time = time.time()

    for step in range(1, config.max_steps + 1):
        model.train()

        if config.curriculum_mode == "none":
            active_iter = iter_A
        elif config.curriculum_mode in ["clean_to_collapse", "collapse_to_clean"]:
            active_iter = iter_A if step <= config.switch_step else iter_B
        elif config.curriculum_mode == "interleaved":
            cycle = 100
            active_iter = iter_A if (step % cycle) < (config.interleaved_ratio * cycle) else iter_B

        try:
            inputs, targets = next(active_iter)
        except StopIteration:
            if active_iter is iter_A:
                iter_A = iter(train_loader_A)
                inputs, targets = next(iter_A)
            else:
                iter_B = iter(train_loader_B)
                inputs, targets = next(iter_B)

        inputs, targets = inputs.to(device), targets.to(device)
"""

content = re.sub(
    r"    # Training loop.*        inputs, targets = inputs\.to\(device\), targets\.to\(device\)",
    train_loop,
    content,
    flags=re.DOTALL
)

with open("src/train.py", "w") as f:
    f.write(content)
