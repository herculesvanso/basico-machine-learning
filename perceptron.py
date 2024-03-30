import numpy as np
import pandas as pd

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
