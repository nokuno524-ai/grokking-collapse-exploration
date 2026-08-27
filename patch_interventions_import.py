import re

with open("src/analysis/interventions.py", "r") as f:
    content = f.read()

content = content.replace("from src.transplant_rescue import evaluate_model", "from src.train import evaluate")

evaluate_call_search = """    # First get baseline accuracy
    metrics = evaluate_model(model, test_loader, test_loader, device)
    baseline_acc = metrics['test_acc']"""

evaluate_call_replace = """    # First get baseline accuracy
    _, baseline_acc = evaluate(model, test_loader, device)"""

content = content.replace(evaluate_call_search, evaluate_call_replace)

abl_call_search = """                abl_metrics = evaluate_model(model, test_loader, test_loader, device)
                abl_acc = abl_metrics['test_acc']"""

abl_call_replace = """                _, abl_acc = evaluate(model, test_loader, device)"""

content = content.replace(abl_call_search, abl_call_replace)

with open("src/analysis/interventions.py", "w") as f:
    f.write(content)
