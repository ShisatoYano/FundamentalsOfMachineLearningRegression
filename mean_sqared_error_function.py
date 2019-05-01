# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Mean sqared error function
def mse_line_func(x, t, w):
    y = w[0] * x + w[1]
    mse = np.mean((y - t)**2)
    return mse

# load database
data = np.load('age_height_data.npz')
X    = data['X']
T    = data['T']

# Calculation
x_n = 100
w_0_range = [-25, 25]
w_1_range = [120, 170]
x_0 = np.linspace(w_0_range[0], w_0_range[1], x_n)
x_1 = np.linspace(w_1_range[0], w_1_range[1], x_n)
xx_0, xx_1 = np.meshgrid(x_0, x_1)
J = np.zeros((len(x_0), len(x_1)))
for i_0 in range(x_n):
    for i_1 in range(x_n):
        J[i_1, i_0] = mse_line_func(X, T, (x_0[i_0], x_1[i_1]))

# Display
plt.figure(figsize=(9.5, 4))
plt.subplots_adjust(wspace=0.5)
ax = plt.subplot(1, 2, 1, projection='3d')
ax.plot_surface(xx_0, xx_1, J, rstride=10, cstride=10, alpha=0.3,
                color='blue', edgecolor='black')
ax.set_xticks([-20, 0, 20])
ax.set_yticks([120, 140, 160])
ax.set_xlabel('w_0')
ax.set_ylabel('w_1')
ax.view_init(20, -60)
plt.subplot(1, 2, 2)
cont = plt.contour(xx_0, xx_1, J, 30, color='black',
                   levels=[100, 1000, 10000, 100000])
cont.clabel(fmt='%1.0f', fontsize=8)
plt.xlabel('w_0')
plt.ylabel('w_1')
plt.grid(True)
plt.show()

