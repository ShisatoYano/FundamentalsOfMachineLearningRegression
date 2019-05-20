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

# Basis gaussian function
def gaussian(x, mu, s):
    return np.exp(-(x - mu)**2 / (2 * s**2))

# optimization of model A
def optimization_model_a(x, t, m):
    mu  = np.linspace(5, 30, m)
    s   = mu[1] - mu[0]
    n   = x.shape[0]
    psi = np.ones((n, m+1))
    for j in range(m):
        psi[:, j] = gaussian(x, mu[j], s)
    psi_T = np.transpose(psi)
    b = np.linalg.inv(psi_T.dot(psi))
    c = b.dot(psi_T)
    w = c.dot(t)
    return w

# Model A: Linear basis gaussian function model
def model_a(w, x):
    m  = len(w) - 1
    mu = np.linspace(5, 30, m)
    s  = mu[1] - mu[0]
    y  = np.zeros_like(x)
    for j in range(m):
        y = y + w[j] * gaussian(x, mu[j], s)
    y = y + w[m]
    return y

# mean squared error of model A
def mse_model_a(x, t, w):
    y = model_a(w, x)
    mse = np.mean((y - t)**2)
    return mse

# K-hold cross-validation for model A
def k_hold_model_a(x, t, m, k):
    n = x.shape[0]
    mse_training = np.zeros(k)
    mse_test     = np.zeros(k)
    for i in range(0, k):
        x_training = x[np.fmod(range(n), k) != i]
        t_training = t[np.fmod(range(n), k) != i]
        x_test     = x[np.fmod(range(n), k) == i]
        t_test     = t[np.fmod(range(n), k) == i]
        w_m        = optimization_model_a(x_training, t_training, m)
        mse_training[i] = mse_model_a(x_training, t_training, w_m)
        mse_test[i]     = mse_model_a(x_test, t_test, w_m)
    return mse_training, mse_test

# Model B
def model_b(x, w):
    y = w[0] - w[1] * np.exp(-w[2] * x)
    return y

# mean sqared error of model B
def mse_model_b(w, x, t):
    y   = model_b(x, w)
    mse = np.mean((y - t)**2)
    return mse

# optimization of model B
def optimization_model_b(w_init, x, t):
    res = minimize(mse_model_b, w_init, args=(x, t), method="powell")
    return res.x

# K-hold cross-validation for model B
def k_hold_model_b(x, t, k):
    n = len(x)
    mse_training = np.zeros(k)
    mse_test     = np.zeros(k)
    for i in range(0, k):
        x_training = x[np.fmod(range(n), k) != i]
        t_training = t[np.fmod(range(n), k) != i]
        x_test     = x[np.fmod(range(n), k) == i]
        t_test     = t[np.fmod(range(n), k) == i]
        w_m        = optimization_model_b(np.array([182.3, 107.2, 0.2]), x_training, t_training)
        mse_training[i] = mse_model_b(w_m, x_training, t_training)
        mse_test[i]     = mse_model_b(w_m, x_test, t_test)
    return mse_training, mse_test

# main
K = 20

# prediction accuracy by model A
M_a = range(2, 7)
model_a_training = np.zeros((K, len(M_a)))
model_a_test     = np.zeros((K, len(M_a)))
for i in range(0, len(M_a)):
    model_a_training[:, i], model_a_test[:, i] =\
        k_hold_model_a(X, T, M_a[i], K)
mean_model_a_training = np.sqrt(np.mean(model_a_training, axis=0))
mean_model_a_test     = np.sqrt(np.mean(model_a_test, axis=0))

# prediction accuracy by model B
model_b_training, model_b_test = k_hold_model_b(X, T, K)
mean_model_b_test = np.sqrt(np.mean(model_b_test))
print("Model A(M=5) SD={0:.2f}[cm]".format(mean_model_a_test[3]))
print("Model B SD={0:.2f}[cm]".format(mean_model_b_test))
sd = np.append(mean_model_a_test[0:5], mean_model_b_test)
M_b = range(6)
label = ["A(M=2)", "A(M=3)", "A(M=4)", "A(M=5)", "A(M=6)", "B"]
plt.figure(figsize=(5, 3))
plt.bar(M_b, sd, tick_label=label, align="center", facecolor="cornflowerblue")
plt.ylabel("SD[cm]")
plt.show()

