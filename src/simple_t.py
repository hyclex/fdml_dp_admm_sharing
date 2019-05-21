from joblib import Parallel, delayed
from math import sqrt

Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10)) 