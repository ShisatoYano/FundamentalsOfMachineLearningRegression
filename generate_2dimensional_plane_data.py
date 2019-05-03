# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load database
data  = np.load('age_height_data.npz')
X     = data['X']
X_min = data['X_min']
X_max = data['X_max']
X_n   = data['X_n']
T     = data['T']

# generate
X_0 = X
X_0_min = 5
X_0_max = 30
np.random.seed(seed=1)
X_1 = 23 * (T/100)**2 + 2*np.random.randn(X_n)
X_1_min = 40
X_1_max = 75

# show data function
def show_data(ax, x_0, x_1, t):
    for i in range(len(x_0)):
        ax.plot([x_0[i], x_0[i]], [x_1[i], x_1[i]],
                [120, t[i]], color='gray')
    ax.plot(x_0, x_1, t, 'o',
            markeredgecolor='black', color='cornflowerblue',
            markersize=6, markeredgewidth=0.5)
    ax.view_init(elev=35, azim=-75)

# plot generated data
plt.figure(figsize=(6,5))
ax = plt.subplot(1, 1, 1, projection='3d')
show_data(ax, X_0, X_1, T)
plt.xlabel('Age:$x_0$')
plt.ylabel('Weight[Kg]:$x_1$')
plt.grid(True)
plt.show()
