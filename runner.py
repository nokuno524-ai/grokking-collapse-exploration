import argparse
import yaml
from src.train import train, TrainConfig

def load_config(config_path: str) -> TrainConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TrainConfig(**config_dict)

def main():
    parser = argparse.ArgumentParser(description="Run a training experiment using a YAML configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)

if __name__ == "__main__":
    main()
