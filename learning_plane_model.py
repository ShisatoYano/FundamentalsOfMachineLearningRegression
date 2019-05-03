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

# show plane function
def show_plane(ax, w):
    p_x_0 = np.linspace(X_0_min, X_0_max, 5)
    p_x_1 = np.linspace(X_1_min, X_1_max, 5)
    p_x_0, p_x_1 = np.meshgrid(p_x_0, p_x_1)
    y = w[0]*p_x_0 + w[1]*p_x_1 + w[2]
    ax.plot_surface(p_x_0, p_x_1, y, rstride=1, cstride=1, alpha=0.3,
                    color='blue', edgecolor='black')

# mean squared error of plane
def calc_mse_plane(x_0, x_1, t, w):
    y = w[0]*x_0 + w[1]*x_1 + w[2]
    mse = np.mean((y - t)**2)
    return mse

# calculate optimal parameters
def calc_optimal_param(x_0, x_1, t):
    c_tx_0 = np.mean(t*x_0) - np.mean(t) * np.mean(x_0)
    c_tx_1 = np.mean(t*x_1) - np.mean(t) * np.mean(x_1)
    c_x_01 = np.mean(x_0*x_1) - np.mean(x_0) * np.mean(x_1)
    v_x_0  = np.var(x_0)
    v_x_1  = np.var(x_1)
    w_0    = (c_tx_1*c_x_01-v_x_1*c_tx_0)/(c_x_01**2-v_x_0*v_x_1)
    w_1    = (c_tx_0*c_x_01-v_x_0*c_tx_1)/(c_x_01**2-v_x_0*v_x_1)
    w_2    = -w_0 * np.mean(x_0) - w_1 * np.mean(x_1) + np.mean(t)
    return np.array([w_0, w_1, w_2])

# plot generated data
plt.figure(figsize=(6,5))
ax = plt.subplot(1, 1, 1, projection='3d')
W  = calc_optimal_param(X_0, X_1, T)
print("w0={0:.1f}, w1={1:.1f}, w2={2:.1f}".format(W[0], W[1], W[2]))
show_plane(ax, W)
show_data(ax, X_0, X_1, T)
mse = calc_mse_plane(X_0, X_1, T, W)
print("SD={0:.3f} cm".format(np.sqrt(mse)))
plt.xlabel('Age:$x_0$')
plt.ylabel('Weight[Kg]:$x_1$')
plt.grid(True)
plt.show()
