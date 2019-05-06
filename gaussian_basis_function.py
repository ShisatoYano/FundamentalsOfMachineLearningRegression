# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# load database
data  = np.load('age_height_data.npz')
X     = data['X']
X_min = data['X_min']
X_max = data['X_max']
X_n   = data['X_n']
T     = data['T']

# gaussian function
def gaussian(x, mu, s):
    return np.exp(-(x - mu)**2 / (2 * s**2))

# main
M = 4
plt.figure(figsize=(4, 4))
mu = np.linspace(5, 30, M)
s = mu[1] - mu[0]
x_input = np.linspace(X_min, X_max, 100)
for j in range(M):
    y = gaussian(x_input, mu[j], s)
    plt.plot(x_input, y, color='gray', linewidth=3)
plt.grid(True)
plt.xlim(X_min, X_max)
plt.ylim(0, 1.2)
plt.xlabel('Age')
plt.ylabel('$\phi_j(x)$')
plt.show()