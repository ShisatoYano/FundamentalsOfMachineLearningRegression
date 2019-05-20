# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# load database
data  = np.load('age_height_data.npz')
X     = data['X']
X_min = data['X_min']
X_max = data['X_max']
X_n   = data['X_n']
T     = data['T']

# model
def model(x, w):
    y = w[0] - w[1] * np.exp(-w[2] * x)
    return y

# show function of model
def show_model(w):
    x = np.linspace(X_min, X_max, 100)
    y = model(x, w)
    plt.plot(x, y, c=[.5, .5, .5], lw=4)

# Mean sqared error function
def mse_model(w, x, t):
    y   = model(x, w)
    mse = np.mean((y - t)**2)
    return mse

# function to optimize parameters of model
def optimization(w_init, x, t):
    res = minimize(mse_model, w_init, args=(x, t), method="powell")
    return res.x

# main
plt.figure(figsize=(4, 4))
W_init = [100, 0, 0]
W = optimization(W_init, X, T)
show_model(W)
print("w0={0:.1f}, w1={1:.1f}, w2={2:.1f}".format(W[0], W[1], W[2]))
plt.plot(X, T, marker='o', linestyle='None',
         color='cornflowerblue', markeredgecolor='black')
plt.grid(True)
plt.xlabel('Age')
plt.ylabel('Height[cm]')
plt.xlim(X_min, X_max)
mse = mse_model(W, X, T)
print("SD={0:.2f}[cm]".format(np.sqrt(mse)))
plt.show()