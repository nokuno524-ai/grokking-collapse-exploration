import re

with open("src/train.py", "r") as f:
    content = f.read()

# Apply get_condition_seed to TrainConfig so it works everywhere
content = re.sub(
    r"    # Set seeds\n    torch\.manual_seed\(config\.seed\)\n    torch\.cuda\.manual_seed_all\(config\.seed\)",
    r"    # Prevent seed reuse across conditions by hashing condition_name into base seed\n    from .data import get_condition_seed\n    effective_seed = get_condition_seed(config.seed, config.condition_name)\n    \n    # Set seeds\n    torch.manual_seed(effective_seed)\n    torch.cuda.manual_seed_all(effective_seed)",
    content
)

# And replace `config.seed` inside `create_loader` and `loader_generator.manual_seed`
content = re.sub(
    r"seed=config\.seed",
    r"seed=effective_seed",
    content
)

content = re.sub(
    r"loader_generator\.manual_seed\(config\.seed\)",
    r"loader_generator.manual_seed(effective_seed)",
    content
)

with open("src/train.py", "w") as f:
    f.write(content)
