import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time


np.random.seed(42)

_start = time.time()

size, features = 1_000_000, 20
dataset = np.random.randn(size, features)
target = np.random.randn(size)

reg = RandomForestRegressor(max_depth=32, random_state=42, n_jobs=30)
reg.fit(dataset, target)

print(f'Time spent: {time.time() - _start}')
