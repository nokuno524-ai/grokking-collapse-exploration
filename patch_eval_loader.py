import re

with open("src/train.py", "r") as f:
    content = f.read()

# Fix UnboundLocalError by replacing `train_loader` with `train_loader_A` for evaluate
content = re.sub(
    r"train_loss, train_acc = evaluate\(model, train_loader, device\)",
    r"train_loss, train_acc = evaluate(model, train_loader_A, device)", # Can just use one of them for tracking train acc, or the currently active one
    content
)

with open("src/train.py", "w") as f:
    f.write(content)
