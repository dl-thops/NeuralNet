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
import math
# !pip install wandb 
import wandb

# %%
wandb.login()
wandb.init(project="fdl-a1",entity = "fdl-thops")


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
        None
    def initialise_params(self,initialization_type):
        self.init_type = initialization_type
        if self.init_type == "random":
            for i in range( 1, len(self.structure)):
                self.params["w"+str(i)]= np.random.rand( self.structure[i], self.structure[i-1])-0.5
                self.params["b"+str(i)]= np.random.rand( self.structure[i], 1)-0.5
        elif self.init_type == "xavier":
            for i in range(1,len(self.structure)):
                self.params["w"+str(i)]= np.normal(0,1/sqrt(self.structure[i-1]+structure[i]),(self.structure[i],self.structure[i-1]))
                self.params["b"+str(i)]= np.normal(0,1/sqrt(self.structure[i-1]+structure[i]),self.structure[i])
        else:
            print(self.init_type + ": unidentified initialization type")    
    
    def do_sgd(self,X,Y,learning_rate):
        grads = {}
        values = self.predict(X,returndict=1)
        layers = len(self.structure)-1
        grads["a"+str(layers)] = -(np.eye(self.structure[-1])[Y]).T + values["h"+str(layers)]
        nsamples = X.shape[1]
        for ii in np.arange(layers-1,0,-1):
            grads["h"+str(ii)] = np.matmul(self.params["w"+str(ii+1)].T,grads["a"+str(ii+1)])
            grads["a"+str(ii)] = np.multiply(grads["h"+str(ii)],np.multiply(values["h"+str(ii)],(1-values["h"+str(ii)])))
        for ii in np.arange(layers,0,-1):
            grads["w"+str(ii)] = np.matmul((grads["a"+str(ii)].T).reshape(nsamples,-1,1),(values["h"+str(ii-1)].T).reshape(nsamples,1,-1))
            grads["b"+str(ii)] = grads["a"+str(ii)]
        for ii in np.arange(1,layers+1):
            self.params["w"+str(ii)] -= learning_rate * np.sum(grads["w"+str(ii)],axis=0)
            self.params["b"+str(ii)] -= learning_rate * np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)

    def do_momentum(self,X,Y,update,gamma,learning_rate):
        layers = len(self.structure)-1
        grads = {}
        nsamples = X.shape[1]
        values = self.predict(X,returndict=1)
        grads["a"+str(layers)] = -(np.eye(self.structure[-1])[Y]).T + values["h"+str(layers)]
        for ii in np.arange(layers-1,0,-1):
            grads["h"+str(ii)] = np.matmul(self.params["w"+str(ii+1)].T,grads["a"+str(ii+1)])
            grads["a"+str(ii)] = np.multiply(grads["h"+str(ii)],np.multiply(values["h"+str(ii)],(1-values["h"+str(ii)])))
        for ii in np.arange(layers,0,-1):
            grads["w"+str(ii)] = np.matmul((grads["a"+str(ii)].T).reshape(nsamples,-1,1),(values["h"+str(ii-1)].T).reshape(nsamples,1,-1))
            grads["b"+str(ii)] = grads["a"+str(ii)]
        for ii in np.arange(1,layers+1):
            update["w"+str(ii)]=gamma * update["w"+str(ii)] + learning_rate * np.sum(grads["w"+str(ii)],axis=0)
            update["b"+str(ii)]=gamma * update["b"+str(ii)] + learning_rate * np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)
            self.params["w"+str(ii)] -= update["w"+str(ii)]
            self.params["b"+str(ii)] -= update["b"+str(ii)]
        return update

    def do_nesterov(self,X,Y,update,gamma,learning_rate):
        layers = len(self.structure)-1
        grads = {}
        nsamples = X.shape[1]
        for ii in range(1,layers+1):
            self.params["w"+str(ii)] -= gamma * update["w"+str(ii)]
            self.params["b"+str(ii)] -= gamma * update["b"+str(ii)]
        values = self.predict(X,returndict=1)
        grads["a"+str(layers)] = -(np.eye(self.structure[-1])[Y]).T + values["h"+str(layers)]
        for ii in np.arange(layers-1,0,-1):
            grads["h"+str(ii)] = np.matmul(self.params["w"+str(ii+1)].T,grads["a"+str(ii+1)])
            grads["a"+str(ii)] = np.multiply(grads["h"+str(ii)],np.multiply(values["h"+str(ii)],(1-values["h"+str(ii)])))
        for ii in np.arange(layers,0,-1):
            grads["w"+str(ii)] = np.matmul((grads["a"+str(ii)].T).reshape(nsamples,-1,1),(values["h"+str(ii-1)].T).reshape(nsamples,1,-1))
            grads["b"+str(ii)] = grads["a"+str(ii)]
        for ii in np.arange(1,layers+1):
            update["w"+str(ii)]=gamma * update["w"+str(ii)] + learning_rate * np.mean(grads["w"+str(ii)],axis=0)
            update["b"+str(ii)]=gamma * update["b"+str(ii)] + learning_rate * np.mean(grads["b"+str(ii)],axis=1).reshape(-1,1)
            self.params["w"+str(ii)] -= update["w"+str(ii)]
            self.params["b"+str(ii)] -= update["b"+str(ii)]
        return update

    def do_back_prop(self,X,Y,X_cv,Y_cv,optimiser,gamma,numepochs,learning_rate,batch_size):
        layers = len(self.structure)-1
        update = {}
        for i in range(1,layers+1):
            update["w"+str(i)]=np.zeros(self.params["w"+str(i)].shape)
            update["b"+str(i)]=np.zeros(self.params["b"+str(i)].shape) 
        for i in range(numepochs):
            for j in tqdm(range(math.ceil(X.shape[1]/batch_size))):
                X_pass = X[:,j*batch_size:min(X.shape[1],(j+1)*batch_size)]
                Y_pass = Y[j*batch_size:min(X.shape[1],(j+1)*batch_size)]
                if optimiser == "sgd":
                    self.do_sgd(X_pass,Y_pass,learning_rate)
                    Y_pred = self.predict(X)
                    self.accuracies.append(np.mean(np.argmax(Y_pred,axis=0)==Y))
                    self.cvaccuracies.append(np.mean(self.predict(X_cv,returnclass=1)==Y_cv))
                    self.losses.append(np.sum(-np.log(np.choose(Y,Y_pred))))
                    wandb.log({"train_acc":self.accuracies[-1],"train_loss":self.losses[-1],"cv_acc":self.cvaccuracies[-1]})
                elif optimiser == "momentum":
                    update = self.do_momentum(X_pass,Y_pass,update,gamma,learning_rate)
                    Y_pred = self.predict(X)
                    self.accuracies.append(np.mean(np.argmax(Y_pred,axis=0)==Y))
                    self.cvaccuracies.append(np.mean(self.predict(X_cv,returnclass=1)==Y_cv))
                    self.losses.append(np.sum(-np.log(np.choose(Y,Y_pred))))
                    wandb.log({"train_acc":self.accuracies[-1],"train_loss":self.losses[-1],"cv_acc":self.cvaccuracies[-1]})
                elif optimiser == "nesterov":
                    update = self.do_nesterov(X_pass,Y_pass,update,gamma,learning_rate)
                    Y_pred = self.predict(X)
                    self.accuracies.append(np.mean(np.argmax(Y_pred,axis=0)==Y))
                    self.cvaccuracies.append(np.mean(self.predict(X_cv,returnclass=1)==Y_cv))
                    self.losses.append(np.sum(-np.log(np.choose(Y,Y_pred))))
                    wandb.log({"train_acc":self.accuracies[-1],"train_loss":self.losses[-1],"cv_acc":self.cvaccuracies[-1]})


    def train(self,X,Y,numepochs = 100,learning_rate = 0.1,initialization_type = "random",\
              activation = "sigmoid",optimiser = "sgd",gamma=0.1,init_params=True,train_test_split=0.2,seed=3,batch_size = 32):
        #TODO : Assert init_params is true for the first time.
        if init_params == True:
            self.initialise_params(initialization_type)
            self.activation = activation
        permutation = np.arange(X.shape[1])
        np.random.seed(seed)
        np.random.shuffle(permutation)
        X = ((X.T)[permutation]).T
        Y = Y[permutation]
        X_cv = X[:,:int(X.shape[1]*train_test_split)]
        Y_cv = Y[:int(X.shape[1]*train_test_split)]
        temp = X.shape[1]
        X = X[:,int(temp*train_test_split):] 
        Y = Y[int(temp*train_test_split):]
        self.accuracies = []
        self.cvaccuracies = []
        self.losses = []
        self.do_back_prop(X,Y,X_cv,Y_cv,optimiser,gamma,numepochs,learning_rate,batch_size)
        dsffg = '''
        layers = len(self.structure)-1
        nsamples = X.shape[1]
        self.accuracies = []
        if optimiser == "momentum" or optimiser == "nesterov":
            update = {}
            for i in range(1,layers+1):
                update["w"+str(i)]=np.zeros(self.params["w"+str(i)].shape)
                update["b"+str(i)]=np.zeros(self.params["b"+str(i)].shape)
        for epoch in tqdm(range(numepochs)):
            grads = {}
            acc = self.predict(X,returnclass=1)
            if optimser == "nesterov":
                for ii in range(1,layers+1):
                    self.params["w"+str(ii)] -= gamma * update["w"+str(ii)]
                    self.params["b"+str(ii)] -= gamma * update["b"+str(ii)]
            values = self.predict(X,returndict=1)
            self.accuracies.append(np.mean(acc == Y))
            wandb.log({"train_accuracy":self.accuracies[-1]})
            grads["a"+str(layers)] = -(np.eye(self.structure[-1])[Y]).T + values["h"+str(layers)]
            for ii in np.arange(layers-1,0,-1):
                grads["h"+str(ii)] = np.matmul(self.params["w"+str(ii+1)].T,grads["a"+str(ii+1)])
                grads["a"+str(ii)] = np.multiply(grads["h"+str(ii)],np.multiply(values["h"+str(ii)],(1-values["h"+str(ii)])))
            for ii in np.arange(layers,0,-1):
                grads["w"+str(ii)] = np.matmul((grads["a"+str(ii)].T).reshape(nsamples,-1,1),(values["h"+str(ii-1)].T).reshape(nsamples,1,-1))
                grads["b"+str(ii)] = grads["a"+str(ii)]
                
            if optimiser == "sgd":
                for ii in np.arange(1,layers+1):
                    self.params["w"+str(ii)] -= learning_rate * np.mean(grads["w"+str(ii)],axis=0)
                    self.params["b"+str(ii)] -= learning_rate * np.mean(grads["b"+str(ii)],axis=1).reshape(-1,1)
            elif optimiser == "momentum" or optimiser == "nesterov":
                for ii in np.arange(1,layers+1):
                    update["w"+str(ii)]=gamma * update["w"+str(ii)] + learning_rate * np.mean(grads["w"+str(ii)],axis=0)
                    update["b"+str(ii)]=gamma * update["b"+str(ii)] + learning_rate * np.mean(grads["b"+str(ii)],axis=1).reshape(-1,1)
                    self.params["w"+str(ii)] -= update["w"+str(ii)]
                    self.params["b"+str(ii)] -= update["b"+str(ii)]'''

    
    def predict(self,X,returndict = 0,returnclass = 0):
        #TODO : dont calculate dict unless returndict =1
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

# %%
(X_train,Y_train),(X_test,Y_test) = fashion_mnist.load_data()

# %%
X_train = X_train.reshape(X_train.shape[0],-1).T/256
X_test = X_test.reshape(X_test.shape[0],-1).T/256

# %%
wandb.config.update({"dataset":"fashion_mnist","input_size":784,"output_size":10,"hidden_layers":[20,10],"epochs":3,\
                     "learning_rate":0.01,"batch_size":64,"initialization_type":"random","activation":"sigmoid",\
                "optimiser":"nesterov","gamma":0.1,"train_test_split":0.2,"seed":7})

# %%
nn = NeuralNet(wandb.config["input_size"],wandb.config["output_size"])
for hidden_layer_size in wandb.config["hidden_layers"]:
    nn.addlayer(hidden_layer_size)
nn.train(X_train,Y_train,wandb.config["epochs"],wandb.config["learning_rate"],\
         initialization_type=wandb.config["initialization_type"],activation=wandb.config["activation"],optimiser=wandb.config["optimiser"],\
         gamma=wandb.config["gamma"],batch_size=wandb.config["batch_size"],train_test_split=wandb.config["train_test_split"],seed=wandb.config["seed"])

# %%
