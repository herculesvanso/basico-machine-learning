import numpy as np
import pandas as pd

def sigmoid(x,beta=.2):
    if beta < 0:
        raise Exception('beta must be positive')
    return 1/(1+np.exp(-beta*x))

class Perceptron():
    def __init__(self, eta=.2, niter=10):
        self.eta = eta
        self.niter = niter
        
    def fit(self,inputs, targets):
        bias = np.array(inputs.shape[0] * [-1]).reshape(-1,1)
        inputs_bias = np.concatenate((inputs,bias),axis=1)
        self.weights = np.random.rand(inputs_bias.shape[1])
        
        for k in np.arange(self.niter) % inputs.shape[0]:
            f = np.dot(inputs_bias[k],self.weights)
            output = np.where(f>0,1,0)
            delta_weights = self.eta * (targets[k]-output) * inputs_bias[k]
            self.weights += delta_weights
        
    def predict(self,inputs):
        bias = np.array(inputs.shape[0] * [-1]).reshape(-1,1)
        inputs_bias = np.concatenate((inputs,bias),axis=1)
        return np.where(np.dot(inputs_bias,self.weights)>0,1,0)
    

class Mlp():
    def __init__(self,eta=.2,maxit=100) -> None:
        self.eta = eta
        self.maxit = maxit

    def fit(self,inputs,targets):
        n = inputs.shape[1]
        m = targets.shape[1]
        l = 2
        inputs_bias = np.concatenate((inputs,np.array(inputs.shape[0]*[-1]).reshape(-1,1)),axis=1)
        weights_inputs_bias = np.random.rand(l,n+1)
        weights_outhidden_bias = np.random.rand(m,l+1)
        for i in range(self.maxit):
            out_hidden = sigmoid(np.dot(weights_inputs_bias,inputs_bias[i])).reshape(-1,1)
            out_hidden_bias = np.concatenate((out_hidden,np.array([-1]).reshape(-1,1)),axis=0)
            out = sigmoid(np.dot(weights_outhidden_bias,out_hidden_bias))

    
    #to continue...
    #testing commiting in vscode