# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Mean sqared error function
def mse_line_func(x, t, w):
    y = w[0] * x + w[1]
    mse = np.mean((y - t)**2)
    return mse

# Gradient of mean square error
def gradient_mse_line(x, t, w):
    y = w[0] * x + w[1]
    d_w_0 = 2 * np.mean((y - t) * x)
    d_w_1 = 2 * np.mean(y - t)
    return d_w_0, d_w_1

# Gradient method
def gradient_method(x, t):
    w_init    = [10.0, 165.0]
    alpha     = 0.001 # learning rate
    i_max     = 100000
    grad_th   = 0.1
    w_i       = np.zeros([i_max, 2])
    w_i[0, :] = w_init
    for i in range(1, i_max):
        grad_mse = gradient_mse_line(x, t, w_i[i-1])
        w_i[i, 0] = w_i[i-1, 0] - alpha * grad_mse[0]
        w_i[i, 1] = w_i[i-1, 1] - alpha * grad_mse[1]
        # convergence check
        if max(np.absolute(grad_mse)) < grad_th:
            break
    w_0 = w_i[i, 0]
    w_1 = w_i[i, 1]
    w_i = w_i[:i, :]
    return w_0, w_1, grad_mse, w_i

# predicted linear line
def pred_linear_line(w):
    x = np.linspace(X_min, X_max, 100)
    y = w[0] * x + w[1]
    plt.plot(x, y, color=(.5, .5, .5), linewidth=4)

# load database
data  = np.load('age_height_data.npz')
X     = data['X']
X_min = data['X_min']
X_max = data['X_max']
T     = data['T']

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
plt.figure(figsize=(4, 4))
cont = plt.contour(xx_0, xx_1, J, 30, color='black',
                   levels=[100, 1000, 10000, 100000])
cont.clabel(fmt='%1.0f', fontsize=8)
plt.xlabel('$w_0$')
plt.ylabel('$w_1$')
plt.grid(True)
# call gradient_method
w_0, w_1, grad_mse, w_history = gradient_method(X, T)
print('Reat Num: {0}'.format(w_history.shape[0]))
print('W=[{0:.6f}, {1:.6f}]'.format(w_0, w_1))
print('Grad_MSE=[{0:.6f}, {1:.6f}]'.format(grad_mse[0], grad_mse[1]))
print('MSE={0:.6f}'.format(mse_line_func(X, T, [w_0, w_1])))
plt.plot(w_history[:, 0], w_history[:, 1], '.-',
         color='gray', markersize=10, markeredgecolor='cornflowerblue')
# call pred_linear_line
plt.figure(figsize=(4, 4))
W = np.array([w_0, w_1])
mse = mse_line_func(X, T, W)
print('W_0={0:.3f}, W_1={1:.3f}'.format(w_0, w_1))
print('SD={0:.3f} cm'.format(np.sqrt(mse)))
pred_linear_line(W)
plt.plot(X, T, marker='o', linestyle='None',
         color='cornflowerblue', markeredgecolor='black')
plt.xlim(X_min, X_max)
plt.xlabel('Age')
plt.ylabel('Height[m]')
plt.grid(True)
plt.show()

