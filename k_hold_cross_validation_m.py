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

# test data
X_test  = X[:int(X_n/4 + 1)]
T_test  = T[:int(X_n/4 + 1)]

# training data
X_train = X[int(X_n/4 + 1):]
T_train = T[int(X_n/4 + 1):]

# gaussian function
def gaussian(x, mu, s):
    return np.exp(-(x - mu)**2 / (2 * s**2))

# fitting function
def fitting_gaussian(x, t, m):
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

# model of linear basis function
def linear_gaussian_model(w, x):
    m  = len(w) - 1
    mu = np.linspace(5, 30, m)
    s  = mu[1] - mu[0]
    y  = np.zeros_like(x)
    for j in range(m):
        y = y + w[j] * gaussian(x, mu[j], s)
    y = y + w[m]
    return y

# mean squared error of model fitting
def mse_fitting_gaussian(x, t, w):
    y = linear_gaussian_model(w, x)
    mse = np.mean((y - t)**2)
    return mse

# display gaussian basis function
def show_gaussian_basis_function(w):
    xb = np.linspace(X_min, X_max, 100)
    y  = linear_gaussian_model(w, xb)
    plt.plot(xb, y, c=[.5, .5, .5], lw=4)

# K-hold cross-validation function
def k_hold_gaussian_function(x, t, m, k):
    n = x.shape[0]
    mse_training = np.zeros(k)
    mse_test     = np.zeros(k)
    for i in range(0, k):
        x_training = x[np.fmod(range(n), k) != i]
        t_training = t[np.fmod(range(n), k) != i]
        x_test     = x[np.fmod(range(n), k) == i]
        t_test     = t[np.fmod(range(n), k) == i]
        w_m        = fitting_gaussian(x_training, t_training, m)
        mse_training[i] = mse_fitting_gaussian(x_training, t_training, w_m)
        mse_test[i]     = mse_fitting_gaussian(x_test, t_test, w_m)
    return mse_training, mse_test

# main
M = range(2, 8)
K = 20
gaussian_training = np.zeros((K, len(M)))
gaussian_test     = np.zeros((K, len(M)))
for i in range(0, len(M)):
    gaussian_training[:, i], gaussian_test[:, i] =\
        k_hold_gaussian_function(X, T, M[i], K)
mean_gaussian_training = np.sqrt(np.mean(gaussian_training, axis=0))
mean_gaussian_test     = np.sqrt(np.mean(gaussian_test, axis=0))

plt.figure(figsize=(4, 3))
plt.plot(M, mean_gaussian_training, marker='o', linestyle='-', 
         color='k', markerfacecolor='w', label='training')
plt.plot(M, mean_gaussian_test, marker='o', linestyle='-', 
         color='cornflowerblue', markeredgecolor='black', label='test')
plt.grid(True)
plt.legend(loc='upper left', fontsize=10)
plt.ylim(0, 20)
plt.xlabel('M')
plt.ylabel('SD[cm]')
plt.show()