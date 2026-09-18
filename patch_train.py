import re

with open("src/train.py", "r") as f:
    content = f.read()

config_block = """    # Data
    collapse_level: float = 0.0
    collapse_severity: float = 0.5
    train_fraction: float = 0.3
    noise_fraction: float = 0.0
    seed: int = 42

    # Curriculum
    curriculum_mode: str = "none" # "none", "clean_to_collapse", "collapse_to_clean", "interleaved"
    switch_step: int = 25000
    interleaved_ratio: float = 0.5 # Fraction of clean vs collapse in interleaved

    # Output
    output_dir: str = "results"
    condition_name: str = "default\""""

content = re.sub(
    r"    # Data.*    condition_name: str = \"default\"",
    config_block,
    content,
    flags=re.DOTALL
)

with open("src/train.py", "w") as f:
    f.write(content)
