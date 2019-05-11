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

# main
plt.figure(figsize=(5, 4))
plt.subplots_adjust(wspace=0.3)
M = range(2, 10)
mse_train = np.zeros(len(M))
mse_test  = np.zeros(len(M))
for i in range(len(M)):
    W = fitting_gaussian(X_train, T_train, M[i])
    mse_train[i] = np.sqrt(mse_fitting_gaussian(X_train, T_train, W))
    mse_test[i]  = np.sqrt(mse_fitting_gaussian(X_test, T_test, W))
plt.plot(M, mse_train, marker='o', linestyle='-', 
         markerfacecolor='white', markeredgecolor='black',
         color='black', label='training')
plt.plot(M, mse_test, marker='o', linestyle='-', 
         color='cornflowerblue', markeredgecolor='black',
         label='test')
plt.grid(True)
plt.legend(loc='lower left', fontsize=10)
plt.xlabel('$M$')
plt.ylabel('SD[cm]')
plt.show()