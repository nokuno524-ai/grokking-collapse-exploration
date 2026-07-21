import torch
import sys
import os

# Add the project root to the sys path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ModularArithmeticTransformer

def generate_checkpoint():
    model = ModularArithmeticTransformer()
    x = torch.randint(0, 59, (4, 2))
    _ = model(x)

    # Checkpoints usually store dictionary with at least 'model_state', 'step', 'config'
    checkpoint = {
        'step': 1000,
        'model_state': model.state_dict(),
        'optimizer_state': {},
        'config': {'prime': 59, 'd_model': 128, 'n_heads': 4, 'd_ff': 512, 'n_layers': 1}
    }

    os.makedirs('tests/data', exist_ok=True)
    torch.save(checkpoint, 'tests/data/dummy_checkpoint.pt')
    print("Checkpoint saved to tests/data/dummy_checkpoint.pt")

if __name__ == '__main__':
    generate_checkpoint()
