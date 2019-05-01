# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=1)

X_min = 4
X_max = 30
X_n   = 20

# input data
X = 5 + 25 * np.random.rand(X_n)

# target data
param = [180, 108, 0.2]
T = param[0] - param[1] * np.exp(-param[2]*X) \
    + 4 * np.random.rand(X_n)

# save as database
np.savez('age_height_data.npz', X=X, X_min=X_min, X_max=X_max, X_n=X_n, T=T)

# plot generated data
plt.figure(figsize=(4,4))
plt.plot(X, T, marker='o', linestyle='None',
         markeredgecolor='black', color='cornflowerblue')
plt.xlim(X_min, X_max)
plt.xlabel('Age')
plt.ylabel('Height[m]')
plt.grid(True)
plt.show()
