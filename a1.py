# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np


# %%
class NeuralNet:

    @staticmethod
    def sigmoid(X):
        X = np.clip( X, -700, 700)
        return 1 / (1. + np.exp(-X))
            
    @staticmethod
    def tanh(X):
        return (1 - np.exp(-2*X)) / (1 + np.exp(-2*X))
        
        
    @staticmethod
    def relu(X):
        return np.where( X<0, 0, X)
        
    @staticmethod
    def activate( X, activation):
        if activation == "sigmoid":
            return NeuralNet.sigmoid(X)
        elif activation == "tanh":
            return NeuralNet.tanh(X)
        elif activation == "relu":
            return NeuralNet.relu(X)
        else:
            raise(ValueError("Unknown activation \"" + activation + "\""))

    def __init__( self, input_size, output_size = 1):
        self.structure = [ input_size, output_size]
        self.weights = []
        self.bias = []
        
    def addlayer( self, layer_size):
        self.structure = self.structure[:-1] + [ layer_size, self.structure[-1]]
        
    def train( self, initialization_type = "random", activation = "sigmoid"):
        self.init_type = initialization_type
        self.activation = activation
        if self.init_type == "random":
            for i in range( 1, len(self.structure)):
                self.weights.append( np.random.rand( self.structure[i], self.structure[i-1]))
                self.bias.append( np.random.rand( self.structure[i], 1))
        elif self.init_type == "xavier":
            #TODO implement xavier initialization
            pass
        else:
            print(self.init_type + ": unidentified initialization type")
    
    def predict( self, X):
        predictions = X
        for i in range(len(self.weights)):
            predictions = NeuralNet.activate( np.matmul( self.weights[i], predictions) +\
                 self.bias[i], self.activation)
        return predictions
    

# %%
