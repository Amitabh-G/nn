# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:42:32 2018

@author: amitabh.gunjan
"""
import numpy as np
import math 

np.random.seed(seed = 2)

'''
Will implement back propagation for the network with a hidden layer and a final fully connected layer. 
Currently the derivative updates are done for the simplest case of just an input layer and an output layer where the output layer has just one neuron.

'''

### Genrating dummy data for neural network regression.

def generate_input(frm, to, noise_mean, noise_sd, multiplier):
    inpt = np.arange(frm, to) + np.random.normal(noise_mean, noise_sd, size = (to - frm))
    output = inpt*multiplier + np.random.normal(noise_mean, noise_sd, size = (to - frm))
    
    return(inpt, output)

def init_weights(input_array):

    # Need to create a weight matrix where earch row is a weight vector -- corresponding to one neuron. 
    # Current case is the one with just one neuron. -- The simplest case.
    weights = np.random.normal(0, 0.1, size = np.shape(inpt_array))
    return(weights)

def network_input(inpt, init_weights):
    y_hat = inpt*init_weights
    # print('Y_hat\n', y_hat)
    return(y_hat)

### Activation function must be chosen based on the problem objective.
### Relu is used for regression objective.
### Gradient becomes zero for relu activation function.

def relu_activation(y_hat):
    relu = []
    for i in y_hat:
        relu.append(max(0, i))
    # print('The relu activated output\n', relu)
    return(relu)

def sigmoid_activation(y_hat):
    sig = []
    for i in y_hat:
        sig.append(1/(1 + math.exp(-i)))
    # print('The sigmoid activated output\n', sig)
    return(sig)


### Loss function must be chosen based on the problem objective, i.e. regression or classification

# A regression loss function.
def mean_squared_loss(actual_y, _output):
    loss = ((actual_y - _output)**2)/len(actual_y)
    # print('The loss\n', loss)
    return(loss)

    
# loss = mean_squared_loss(actual_y, y_hat)
# print(loss)

### General procedure for gradient descent for all kinds of problems.
def compute_derivative_wrt_weights(weights):
    '''
    This is numerical derivative implementation. 
        Nudging the weights a bit and then computing the losses for the original and th nudged weights.
    '''
    
    dist = 10**(-4)
    weights_distrbd = weights + dist
    loss = mean_squared_loss(actual_y, sigmoid_activation(network_input(inpt_array, weights)))
    loss_distrbd = mean_squared_loss(actual_y, sigmoid_activation(network_input(inpt_array, weights_distrbd)))
    loss_differential = (loss_distrbd - loss)
    # print('The loss difference\n', loss_differential)
    gradient = loss_differential/dist
    # print('Gradient\n', gradient)
    return(gradient)


### Weight update - the essence of gradient descent.
def weight_update(gradient, previous_weight, eta):
    '''
    Do the weight update as done in gradient descent.
        eta is the learning rate parameter.
    '''
    new_weight = previous_weight + eta*(gradient)
    return(new_weight)

#############
### Constants
#############
iterations = 100
weightts_list = []
dist = 10**(-1)

### Input x and y  
inpt_array = generate_input(1, 10, 0, 1, 6)[0]
actual_y = generate_input(1, 10, 0, 1, 6)[1]


def train(inpt_array, actual_y):

    weights = init_weights(inpt_array) 
    new_weight = None
    prev_weight = weights

    for i in range(iterations):
        weights_distrbd = weights + dist
        print('Weights\n', weights)
        print('Disturbed weights\n', weights_distrbd)

        gradient = compute_derivative_wrt_weights(weights)
        new_weight = weight_update(gradient, weights, 0.94)
        weights = new_weight
        print('the new weights\n', weights, new_weight)

if __name__ == "__main__": 
    train(inpt_array, actual_y)

print('The input array\n', inpt_array)
print('The actual Y:\n', actual_y)
