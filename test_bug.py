import numpy as np

rng = np.random.RandomState(42)
vals = set()
for _ in range(1000):
    vals.add(rng.randint(0, 10))
print("randint(0, 10):", min(vals), "to", max(vals))
