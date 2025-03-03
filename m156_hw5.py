#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 14:57:23 2025

@author: dongsijia
"""

import numpy as np
import pickle
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold


# Load xor toy dataset
with open('xordata.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train'] # 800 training data points with 2 features
y_train = data['y_train'] # training binary labels {0,1}

X_test = data['X_test']
y_test = data['y_test']

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Visualize data
plt.scatter(X_train[:,0], X_train[:,1], s=40, c=y_train, cmap=plt.cm.Spectral)


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def reLu(x):
    return np.maximum(x, 0)

def reLu_derivative(x):
    return np.where(x > 0, 1, 0)

# Loss Function
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-9  
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# Training function of two-layer perception
# With forward pass, backward pass, weight updates
# ReLu for hidden layer, sigmoid for output layer
# cross-entropy loss as loss function
# Mini-batch SGD
def nntrain(X, y, W1, b1, W2, b2, learning_rate, batch_size, epochs, report):
    N = X.shape[0] # number of samples
    loss = 100
    
    for epoch in range(epochs):
        # Shuffles the dataset
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Create Mini-batches
        for start in range(0, N, batch_size):
            end = start + batch_size
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            y_batch = np.array(y_batch).reshape(-1, 1)
            
            # Forward propogation
            hidden_input = np.dot(X_batch, W1) + b1
            hidden_output = reLu(hidden_input)
            
            final_input = np.dot(hidden_output, W2) + b2
            final_output = sigmoid(final_input)
            
            # Compute loss
            loss = cross_entropy_loss(y_batch, final_output)
            
            # Backpropagation
            d_output = final_output - y_batch
            error_hidden = np.dot(d_output, W2.T)
            d_hidden = error_hidden * reLu_derivative(hidden_input) 
            
            # Weight and Bias update
            W2 -= learning_rate * np.dot(hidden_output.T, d_output) / batch_size
            b2 -= learning_rate * np.sum(d_output, axis=0, keepdims=True) / batch_size
            W1 -= learning_rate * np.dot(X_batch.T, d_hidden) / batch_size
            b1 -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True) / batch_size
            
        #if epoch % 10 == 0:
            #print("Epoch: ", epoch, "Loss: ", loss)
    if (report == True):
        y_pred_train = (final_output >= 0.5).astype(int)
        
        accuracy_train = accuracy_score(y_batch, y_pred_train)
        precision_train = precision_score(y_batch, y_pred_train)
        recall_train = recall_score(y_batch, y_pred_train)
        f1_train = f1_score(y_batch, y_pred_train)
        
        print("\nTraining Accuracy: ", accuracy_train)
        print("Training Precision: ", precision_train)
        print("Training Recall: ", recall_train)
        print("Training F1-Score: ", f1_train)
            
    return W1, b1, W2, b2

# Cross validation
def cross_validate(X, y, k_folds, input_neurons, hidden_neurons, output_neurons, learning_rate, batch_size, epochs):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    losses = []
    fold = 1
    
    for train_idx, val_idx in kf.split(X):  # No enumerate()
        print(f"\nFold {fold}/{k_folds}...")
        fold += 1

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize Weights and Biases
        np.random.seed(30)
        W1 = np.random.randn(input_neurons, hidden_neurons)
        b1 = np.random.randn(1, hidden_neurons)
        W2 = np.random.randn(hidden_neurons, output_neurons)
        b2 = np.random.randn(1, output_neurons)
        
        # Train the Model
        W1, b1, W2, b2 = nntrain(X_train, y_train, W1, b1, W2, b2, learning_rate, batch_size, epochs, report = False)

        # Validate the Model
        hidden_val = reLu(np.dot(X_val, W1) + b1)
        final_val = sigmoid(np.dot(hidden_val, W2) + b2)
        val_loss = cross_entropy_loss(y_val, final_val)

        losses.append(val_loss)
        
    return np.mean(losses)
    

# Number of nodes for input/output layers
input_neurons = 2
output_neurons = 1

# Hyperparameters
hidden_neurons = 3
learning_rate = 0.01
batch_size = 64
epochs = 100
k_folds = 5

# Implement cross validation to get best hyperparameters
avg_loss = cross_validate(X_train, y_train, k_folds, input_neurons, hidden_neurons, output_neurons, learning_rate, batch_size, epochs)
print("Average Validation Loss: ", avg_loss)

# Randomly initialize weights and bias
np.random.seed(30)
W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.random.randn(1, hidden_neurons)
W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.random.randn(1, output_neurons)

# Implement training
W1, b1, W2, b2 = nntrain(X_train, y_train, W1, b1, W2, b2, learning_rate, batch_size, epochs, report = True)

hidden_test = reLu(np.dot(X_test, W1) + b1)
final_test = sigmoid(np.dot(hidden_test, W2) + b2)

# Convert predictions to binary (threshold = 0.5)
y_pred = (final_test >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nTesting Accuracy: ", accuracy)
print("Testing Precision: ", precision)
print("Testing Recall: ", recall)
print("Testing F1-Score: ", f1)




























