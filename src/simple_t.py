import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

import math
def rosen(x):
    """The Rosenbrock function"""
    return np.log(1+np.exp(-x))

def rosen_der(x):
    return  -1 / (1+np.exp(x))

def rosen_hess(x):
	zy = x
	ezy = np.exp(zy)
	return np.diag(1.**2 * ezy / (1+ezy)**2)



x0 = np.array([0])

start = time.time()
for i in range(1):
	# res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': False})
	res = minimize_scalar(rosen, options={'disp':True})
print time.time()-start
print res.x
# total = 600000
# hehe = np.random.rand(total)
# start = time.time()
# for i in range(total):
# 	math.exp(hehe[i])
# print time.time() - start 

# start = time.time()
# for i in range(total):
# 	math.log(hehe[i])
# print time.time() - start