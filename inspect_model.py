import torch
from src.model import ModularArithmeticTransformer

model = ModularArithmeticTransformer()
for name, param in model.named_parameters():
    print(name, param.shape)
