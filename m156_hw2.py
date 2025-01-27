#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:34:03 2025

@author: dongsijia
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

# a. Import file
file_path = "/Users/dongsijia/Desktop/ucla/math/Math_156/HW_2/winequality-red.csv"
df = pd.read_csv(file_path, sep=";")

print(df.head())


# Assign X, y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1,1)

# b. Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=62)

# c. Taining a simple linear regression model with sum-of-squares error function
# using the closed-form solution

# Add a column of ones to X for bias
X_train_b = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Compute direct solution (X^T X)^(-1) X^T y
X_transpose_X = np.dot(X_train_b.T, X_train_b)
X_transpose_X_inv = np.linalg.inv(X_transpose_X)
X_transpose_y = np.dot(X_train_b.T, y_train)

w = np.dot(X_transpose_X_inv, X_transpose_y)

# Function for predicted target value
def predict(X, w):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    return np.dot(X_bias, w)

pred_train = predict(X_train, w)

pred_test = predict(X_test, w)


# d. Plot: actual vs predicted target values - train set
f1 = plt.figure(1)
plt.scatter(y_train, pred_train)
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Target Values')
plt.title('Actual vs Predicted Target Values (Training Data)')
plt.grid(True)
plt.show()

# e. Report root mean square
RMSE_train = np.sqrt(np.mean((y_train - pred_train) ** 2))
RMSE_test = np.sqrt(np.mean((y_test - pred_test) ** 2))

print("Root Mean Square Error - Training Set:", RMSE_train)
print("Root Mean Square Error - Test Set:", RMSE_test)


# f. Least-mean-square algorithm
np.random.seed(80)
def least_mean_squares(X, y, stepsize, epochs):
    n_samples, n_features = X.shape
    X_bias = np.hstack((np.ones((n_samples, 1)), X)) 
    w = np.random.randn(n_features + 1, 1)

    for epoch in range(epochs):
        for i in range(n_samples):
            xi = X_bias[i, :].reshape(1, -1) 
            yi = y[i]
            prediction = np.dot(xi, w)
            w += stepsize * 2 * (yi - prediction) * xi.T 

    return w

stepsize = 0.0001
epochs = 100

weights_lms = least_mean_squares(X_train, y_train, stepsize, epochs)

pred_train_lms = predict(X_train, weights_lms)
pred_test_lms = predict(X_test, weights_lms)

# g. Report root mean square for LMS
RMSE_train_lms = np.sqrt(np.mean((y_train - pred_train_lms) ** 2))
RMSE_test_lms = np.sqrt(np.mean((y_test - pred_test_lms) ** 2))

print("Root Mean Square Error - Training Set (LMS):", RMSE_train_lms)
print("Root Mean Square Error - Test Set (LMS):", RMSE_test_lms)





