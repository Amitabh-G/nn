# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:42:32 2018

@author: amitabh.gunjan
"""
import numpy as np
import math 

np.random.seed(seed = 2)

### Genrating dummy data for neural network regression.
def generate_input(frm, to, noise_mean, noise_sd, multiplier):
    inpt = np.arange(frm, to)
    output = inpt*multiplier + np.random.normal(noise_mean, noise_sd, size = (to - frm))
    
    return(inpt, output)

### Input x and y 
    
inpt_array = generate_input(1, 101, 0, 1, 6)[0]
actual_y = generate_input(1, 101, 0, 1, 6)[1]


def init_weights(input_array):
    
    weights = np.random.normal(0, 0.1, size = np.shape(inpt_array))
    return(weights)

def dot_prod(inpt, init_weights):
    print(inpt)
    print(init_weights)
    y_hat = inpt*init_weights
    print(y_hat)
    
    return(y_hat)

weights = init_weights(inpt_array) 

y_hat = dot_prod(inpt_array, weights)

### Activation function must be chosen based on the problem objective.
# Sigmoid is used for classification objective.
def sigmoid_activation(dot_prd):
    sigmoid = 1/(1 + math.exp(-dot_prd ))
    return(sigmoid)


# Relu is used for regression objective.
def relu_activation(y_hat):
    relu = max(0, y_hat)
    return(relu)
    
    
#act = sigmoid_activation(y_hat)
act = relu_activation(y_hat)
print(act)

### Loss function must be chosen based on the problem objective, i.e. regression or classification

# A classification loss function.
def log_loss(actual_y, prob_output):
    loss = -(actual_y)*math.log(prob_output) - (1-actual_y)*math.log(1 - prob_output)
    return(loss)
    
# A regression loss function.
def mean_squared_loss(actual_y, _output):
    loss = ((actual_y - _output)**2)
    return(loss)

loss = log_loss(actual_y, act)

### General procedure for gradient descent for all kinds of problems.
def compute_derivative(loss, weights):
    
    dist = 10**(-4)  
    weights_distrbd = weights + dist
    
    loss = relu_activation(dot_prod(inpt_array, weights))
    loss_distrbd = relu_activation(dot_prod(inpt_array, weights_distrbd))
    loss_differential = (loss_distrbd - loss)
    
    gradient = loss_differential/dist
    return(gradient)


### Weight update - the essence of gradient descent.
def weight_update(loss_gradient, last_weight, eta):
    new_weight = last_weight - eta*(loss_gradient)
    return(new_weight)


