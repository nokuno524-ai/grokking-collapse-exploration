import torch
from src.model import ModularArithmeticTransformer

model = ModularArithmeticTransformer()
x = torch.randint(0, 59, (4, 2))
logits = model(x)
print(logits.shape)
