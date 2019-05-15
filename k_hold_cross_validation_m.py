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
    

# main
plt.figure(figsize=(10, 2.5))
plt.subplots_adjust(wspace=0.3)
M = [2, 4, 7, 9]
for i in range(len(M)):
    plt.subplot(1, len(M), i+1)
    W = fitting_gaussian(X_train, T_train, M[i])
    show_gaussian_basis_function(W)
    plt.plot(X_train, T_train, marker='o', linestyle='None', 
             color='white', markeredgecolor='black', label='training')
    plt.plot(X_test, T_test, marker='o', linestyle='None', 
             color='cornflowerblue', markeredgecolor='black', label='test')
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=10, numpoints=1)
    plt.xlim(X_min, X_max)
    plt.ylim(130, 200)
    plt.xlabel('Age')
    plt.ylabel('Height[cm]')
    mse = mse_fitting_gaussian(X_test, T_test, W)
    plt.title("M={0:d}, SD={1:.1f}".format(M[i], np.sqrt(mse)))
plt.show()