"""
Trims dataset1.csv through dataset5.csv to a single row each (the first row).
Overwrites the files in place.
"""

import os
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.join(_BASE, "dataset")

for i in range(1, 6):
    path = os.path.join(_DATASET, f"dataset{i}.csv")
    if not os.path.exists(path):
        print(f"  SKIP  dataset{i}.csv — not found")
        continue
    df = pd.read_csv(path, nrows=1)
    df.to_csv(path, index=False)
    print(f"  DONE  dataset{i}.csv → kept 1 row")

print("\nAll done.")
