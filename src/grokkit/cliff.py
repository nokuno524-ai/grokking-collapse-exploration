import math
from typing import Dict, List, Any, Optional
import numpy as np

def find_cliff(runs: List[Dict[str, Any]],
               x_key: str,
               y_key: str,
               value_key: str,
               threshold: float = 0.20,
               direction: str = 'below') -> Dict[float, Optional[float]]:
    """
    Generalized cliff detection logic.
    For each unique x_key, return the smallest y_key where the mean of value_key crosses the threshold.
    `direction`: 'below' means looking for values < threshold (e.g., Fourier cliff),
                 'above' means looking for values > threshold.
    """
    cliff = {}

    # Extract unique x and y values
    xs = sorted(list(set(r.get(x_key) for r in runs if r.get(x_key) is not None)))
    ys = sorted(list(set(r.get(y_key) for r in runs if r.get(y_key) is not None)))

    for x in xs:
        for y in ys:
            cell = [
                r.get(value_key) for r in runs
                if r.get(x_key) is not None and math.isclose(r[x_key], x)
                and r.get(y_key) is not None and math.isclose(r[y_key], y)
                and r.get(value_key) is not None and not math.isnan(r[value_key])
            ]

            if not cell:
                continue

            mu = sum(cell) / len(cell)
            if direction == 'below' and mu < threshold:
                cliff[x] = y
                break
            elif direction == 'above' and mu > threshold:
                cliff[x] = y
                break
        cliff.setdefault(x, None)
    return cliff
