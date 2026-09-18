import re

with open("src/train.py", "r") as f:
    content = f.read()

# Make it evaluate on the currently active loader to track proper training loss/acc
replacement = """
            # Evaluate periodically
            if step % config.eval_every == 0:
                current_train_loader = train_loader_A if active_iter is iter_A else train_loader_B
                train_loss, train_acc = evaluate(model, current_train_loader, device)
                test_loss, test_acc = evaluate(model, test_loader, device)
"""

content = re.sub(
    r"            # Evaluate periodically\n            if step % config\.eval_every == 0:\n                train_loss, train_acc = evaluate\(model, train_loader_A, device\)\n                test_loss, test_acc = evaluate\(model, test_loader, device\)",
    replacement,
    content
)

with open("src/train.py", "w") as f:
    f.write(content)
