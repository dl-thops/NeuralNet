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
from keras.datasets import fashion_mnist
from tqdm import tqdm
import matplotlib.pyplot as plt

#%%
import wandb
wandb.login()
wandb.init(project="fdl-a1")

# %%
class NeuralNet:

    @staticmethod
    def sigmoid(X):
        X = np.clip( X, -700, 700)
        return 1 / (1. + np.exp(-X))
            
    @staticmethod
    def tanh(X):
        X = np.clip(X,-350,350)
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

    def __init__( self, input_size, output_size = 2):
        self.structure = [ input_size, output_size]
        self.params = {}
        
    def addlayer( self, layer_size):
        self.structure = self.structure[:-1] + [ layer_size, self.structure[-1]]
        
    def train( self,X,Y,numepochs = 100,learning_rate = 0.1,initialization_type = "random", activation = "sigmoid"):
        self.init_type = initialization_type
        self.activation = activation
        if self.init_type == "random":
            for i in range( 1, len(self.structure)):
                self.params["w"+str(i)]= np.random.rand( self.structure[i], self.structure[i-1])-0.5
                self.params["b"+str(i)]= np.random.rand( self.structure[i], 1)-0.5
        elif self.init_type == "xavier":
            #TODO implement xavier initialization
            pass
        else:
            print(self.init_type + ": unidentified initialization type")
        layers = len(self.structure)-1
        nsamples = X.shape[1]
        self.accuracies = []
        for epoch in tqdm(range(numepochs)):
            grads = {}
            values = self.predict(X,returndict=1)
            self.accuracies.append(np.mean(np.argmax(values["h"+str(layers)],axis=0) == Y))
            wandb.log({"train_accuracy":self.accuracies[-1]})
            grads["a"+str(layers)] = -(np.eye(self.structure[-1])[Y]).T + values["h"+str(layers)]
            for ii in np.arange(layers-1,0,-1):
                grads["h"+str(ii)] = np.matmul(self.params["w"+str(ii+1)].T,grads["a"+str(ii+1)])
                grads["a"+str(ii)] = np.multiply(grads["h"+str(ii)],np.multiply(values["h"+str(ii)],(1-values["h"+str(ii)])))
            for ii in np.arange(layers,0,-1):
                grads["w"+str(ii)] = np.matmul((grads["a"+str(ii)].T).reshape(nsamples,-1,1),(values["h"+str(ii-1)].T).reshape(nsamples,1,-1))
                grads["b"+str(ii)] = grads["a"+str(ii)]
            for ii in np.arange(1,layers+1):
                self.params["w"+str(ii)] -= learning_rate * np.mean(grads["w"+str(ii)],axis=0)
                self.params["b"+str(ii)] -= learning_rate * np.mean(grads["b"+str(ii)],axis=1).reshape(-1,1)
    
    
    def predict(self,X,returndict = 0,returnclass = 0):
        predictions = X
        values = {}
        values["h0"] = X
        layers = len(self.structure)-1
        for i in range(layers-1):
            predictions = np.matmul( self.params["w"+str(i+1)], predictions) + self.params["b"+str(i+1)]
            values["a"+str(i+1)]=predictions
            predictions = NeuralNet.activate(predictions,self.activation)
            values["h"+str(i+1)]=predictions
        predictions = np.matmul( self.params["w"+str(layers)], predictions) + self.params["b"+str(layers)]
        values["a"+str(layers)]=predictions
        if returnclass == 1:
            return np.argmax(predictions,axis=0)
        np.clip(predictions,-700,700)
        predictions = np.exp(predictions)/np.sum(np.exp(predictions),axis=0)
        values["h"+str(layers)]=predictions
        if returndict ==0:
            return predictions
        else:
            return values

#%%
(X_train,Y_train),(X_test,Y_test) = fashion_mnist.load_data()

#%%
X_train = X_train.reshape(X_train.shape[0],-1).T/256
X_test = X_test.reshape(X_test.shape[0],-1).T/256

#%%
wandb.config = {"dataset":"fashion_mnist","input_size":784,"output_size":10,"hidden_layers":[20,20,10],"epochs":60,\
                     "learning_rate":2.5,"batch_size":None,"initialization_type":"random","activation":"sigmoid"}

#%%
nn = NeuralNet(wandb.config["input_size"],wandb.config["output_size"])
for hidden_layer_size in wandb.config["hidden_layers"]:
    nn.addlayer(hidden_layer_size)
nn.train(X_train,Y_train,wandb.config["epochs"],wandb.config["learning_rate"],\
         initialization_type=wandb.config["initialization_type"],activation=wandb.config["activation"])

#%%
