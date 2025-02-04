#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:48:58 2025

@author: dongsijia
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            #print("X_batch: ", X_batch)
            #print("y_batch: ", y_batch)
            
            b, f = X_batch.shape
            X_batch_bias = np.hstack((np.ones((b, 1)), X_batch)) 
            #print("X_batch_bias: ", X_batch_bias)

            z = np.dot(X_batch_bias, w)
            z = z.flatten()
            #print("z: ", z)
            
            sig = sigmoid(z)
            #print("sigmoid: ", sig)
            
            error = sig - y_batch
            gradient = np.dot(error, X_batch_bias)
            gradient = gradient.reshape(f+1, 1)
            
            #print("gradient: ", gradient)
            w -= learning_rate * gradient    
                     
    return w

# Import file
file_path = "/Users/dongsijia/Desktop/ucla/math/Math_156/HW_3/wdbc_data.csv"
df = pd.read_csv(file_path, header=None)

#print(df.head())

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].map({"M": 0, "B": 1}).values

# Split the dataset into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=62) 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=62) 


# Report the size of each class in training (+ validation) set
class_size = np.bincount(y_train_val)
print("Class 0 (M): ",class_size[0])
print("Class 1 (B): ", class_size[1])


#Hyperparameters
batch_size = 16
learning_rate = 0.001
epochs = 200

#Train the model
weights = logistic_regression(X_train_val, y_train_val, batch_size, learning_rate, epochs)
#print(weights)


#Compute prediction for test sets
X_test_bias = np.hstack((np.ones((len(X_test), 1)), X_test))
z_test = np.dot(X_test_bias, weights)
probabilities = sigmoid(z_test)
y_pred = (probabilities >= 0.5).astype(int)

#Evaluetion metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAccuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1)

