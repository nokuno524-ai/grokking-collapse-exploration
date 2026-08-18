from src.model import ModularArithmeticTransformer
import torch

model = ModularArithmeticTransformer()
model.eval()
x = torch.randint(0, 59, (4, 2))
# Can we get attention?
print(model)
