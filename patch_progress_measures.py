with open("src/progress_measures.py", "r") as f:
    code = f.read()
code = code.replace(
    'def compute_learning_speed(history: List[Dict], metric: str = "test_acc",\n                           window: int = 10) -> List[Dict]:\n    """Compute rate of change of a metric over a sliding window."""\n    speeds = []',
    'def compute_learning_speed(history: List[Dict], metric: str = "test_acc",\n                           window: int = 10) -> List[Dict]:\n    """Compute rate of change of a metric over a sliding window."""\n    if not history:\n        return []\n    speeds = []'
)
with open("src/progress_measures.py", "w") as f:
    f.write(code)
