import re

with open("src/train.py", "r") as f:
    content = f.read()

# Fix indentation issue
replacement = """        # Evaluate periodically
        if step % config.eval_every == 0:
            current_train_loader = train_loader_A if active_iter is iter_A else train_loader_B
            train_loss, train_acc = evaluate(model, current_train_loader, device)
            test_loss, test_acc = evaluate(model, test_loader, device)"""

content = re.sub(
    r"\s+# Evaluate periodically\n\s+if step % config\.eval_every == 0:\n\s+current_train_loader = train_loader_A if active_iter is iter_A else train_loader_B\n\s+train_loss, train_acc = evaluate\(model, current_train_loader, device\)\n\s+test_loss, test_acc = evaluate\(model, test_loader, device\)",
    "\n" + replacement,
    content
)

with open("src/train.py", "w") as f:
    f.write(content)
