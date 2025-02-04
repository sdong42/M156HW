#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:51:59 2025

@author: dongsijia
"""

"""
3. Implement a program to train a binary logistic regression model using mini-batch SGD. Use the logistic
regression model we derived in class, corresponding to Equation (4.90) from the textbook, and where
the feature transformation φ is the identity function

hyperparameters:
    Batch size
    Fixed learning rate
    Maximum number of iterations

"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# binary logistic regression using mini-batch SGD
def logistic_regression(X, y, batch_size, learning_rate, epochs):
    n_samples, n_features = X.shape
    w = np.random.randn(n_features + 1, 1)

    for epoch in range(epochs):
        #shuffles the dataset
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Loop over the mini-batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            
            # create batch
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            b, f = X_batch.shape
            X_batch_bias = np.hstack((np.ones((b, 1)), X_batch)) 
            
            # compute gradient
            z = np.dot(X_batch_bias, w)
            z = z.flatten()
            sig = sigmoid(z)
            error = sig - y_batch
            gradient = np.dot(error, X_batch_bias)
            gradient = gradient.reshape(f+1, 1)
            
            # update weights
            w -= learning_rate * gradient 
                      
    return w
