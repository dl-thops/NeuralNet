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
# !pip install wandb       #TODO: Comment this before submitting
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
        
    def initialise_params(self,initialization_type):
        self.init_type = initialization_type
        if self.init_type == "random":
            for i in range( 1, len(self.structure)):
                self.params["w"+str(i)]= np.random.rand( self.structure[i], self.structure[i-1])-0.5
                self.params["b"+str(i)]= np.random.rand( self.structure[i], 1)-0.5
        elif self.init_type == "xavier":
            for i in range(1,len(self.structure)):
                self.params["w"+str(i)]= np.random.normal(0,1/np.sqrt(self.structure[i-1]+self.structure[i]),(self.structure[i],self.structure[i-1]))
                self.params["b"+str(i)]= np.random.normal(0,1/np.sqrt(self.structure[i-1]+self.structure[i]),(self.structure[i],1))
        else:
            print(self.init_type + ": unidentified initialization type")    
    
    @staticmethod
    def activation_gradient(A,activation):
        if activation == "sigmoid":
            return np.multiply(A,(1-A))
        elif activation == "tanh":
            return 1 - np.square(A)
        elif activation == "relu":
            A[A>0] = 1
            A[A<0] = 0
            return A
        else:
            raise(ValueError("Unknown activation \"" + activation + "\""))

    def calculate_grads(self,X,Y):
        grads = {}
        values = self.predict(X,returndict=1)
        layers = len(self.structure)-1
        nsamples = X.shape[1]
        grads["a"+str(layers)] = -(np.eye(self.structure[-1])[Y]).T + values["h"+str(layers)]
        for ii in np.arange(layers-1,0,-1):
            grads["h"+str(ii)] = np.matmul(self.params["w"+str(ii+1)].T,grads["a"+str(ii+1)])
            grads["a"+str(ii)] = np.multiply(grads["h"+str(ii)],self.activation_gradient(values["h"+str(ii)],self.activation))
        for ii in np.arange(layers,0,-1):
            grads["w"+str(ii)] = np.matmul((grads["a"+str(ii)].T).reshape(nsamples,-1,1),(values["h"+str(ii-1)].T).reshape(nsamples,1,-1))
            grads["b"+str(ii)] = grads["a"+str(ii)]
        return grads

    def do_sgd(self,X,Y,update,learning_rate):
        layers = len(self.structure)-1
        grads = self.calculate_grads(X,Y)
        for ii in np.arange(1,layers+1):
            self.params["w"+str(ii)] -= learning_rate * np.sum(grads["w"+str(ii)],axis=0)
            self.params["b"+str(ii)] -= learning_rate * np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)
        return update

    def do_momentum(self,X,Y,update,gamma,learning_rate):
        layers = len(self.structure)-1
        grads = self.calculate_grads(X,Y)
        for ii in np.arange(1,layers+1):
            update["w"+str(ii)]=gamma * update["w"+str(ii)] + learning_rate * np.sum(grads["w"+str(ii)],axis=0)
            update["b"+str(ii)]=gamma * update["b"+str(ii)] + learning_rate * np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)
            self.params["w"+str(ii)] -= update["w"+str(ii)]
            self.params["b"+str(ii)] -= update["b"+str(ii)]
        return update

    def do_nesterov(self,X,Y,update,gamma,learning_rate):
        layers = len(self.structure)-1
        for ii in range(1,layers+1):
            self.params["w"+str(ii)] -= gamma * update["w"+str(ii)]
            self.params["b"+str(ii)] -= gamma * update["b"+str(ii)]
        grads = self.calculate_grads(X,Y)
        for ii in np.arange(1,layers+1):
            update["w"+str(ii)]=gamma * update["w"+str(ii)] + learning_rate * np.sum(grads["w"+str(ii)],axis=0)
            update["b"+str(ii)]=gamma * update["b"+str(ii)] + learning_rate * np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)
            self.params["w"+str(ii)] -= update["w"+str(ii)]
            self.params["b"+str(ii)] -= update["b"+str(ii)]
        return update

    def do_rmsprop(self,X,Y,update,learning_rate,beta,epsilon):
        layers = len(self.structure)-1
        grads = self.calculate_grads(X,Y)
        for ii in np.arange(1,layers+1):
            update["w"+str(ii)] = beta*update["w"+str(ii)] + (1-beta)*np.square(np.sum(grads["w"+str(ii)],axis=0))
            update["b"+str(ii)] = beta*update["b"+str(ii)] + (1-beta)*np.square(np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1))
            self.params["w"+str(ii)] -= np.multiply((learning_rate/ np.sqrt(epsilon + update["w"+str(ii)])), np.sum(grads["w"+str(ii)],axis=0))
            self.params["b"+str(ii)] -= np.multiply((learning_rate / np.sqrt(epsilon + update["b"+str(ii)])), np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1))
        return update

    def do_adam(self,X,Y,update,learning_rate,epsilon,beta1,beta2,step_num):
        layers = len(self.structure)-1
        grads = self.calculate_grads(X,Y)
        for ii in np.arange(1,layers+1):
            update["mw"+str(ii)] = beta1*update["mw"+str(ii)] + (1-beta1)*np.sum(grads["w"+str(ii)],axis=0)
            update["mb"+str(ii)] = beta1*update["mb"+str(ii)] + (1-beta1)*np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)
            update["vw"+str(ii)] = beta2*update["vw"+str(ii)] + (1-beta2)*np.square(np.sum(grads["w"+str(ii)],axis=0))
            update["vb"+str(ii)] = beta2*update["vb"+str(ii)] + (1-beta2)*np.square(np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1))
            self.params["w"+str(ii)] -= np.multiply((learning_rate/np.sqrt(epsilon + (update["vw"+str(ii)]/(1-beta2**step_num)))) ,\
                                                    update["mw"+str(ii)]/(1-beta1**step_num))
            self.params["b"+str(ii)] -= np.multiply((learning_rate/np.sqrt(epsilon + (update["vb"+str(ii)]/(1-beta2**step_num)))) ,\
                                                    update["mb"+str(ii)]/(1-beta1**step_num))
        return update
    
    def do_nadam(self,X,Y,update,learning_rate,epsilon,beta1,beta2,step_num):
        layers = len(self.structure)-1
        grads = self.calculate_grads(X,Y)
        for ii in np.arange(1,layers+1):
            update["mw"+str(ii)] = beta1*update["mw"+str(ii)] + (1-beta1)*np.sum(grads["w"+str(ii)],axis=0)
            update["mb"+str(ii)] = beta1*update["mb"+str(ii)] + (1-beta1)*np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)
            update["vw"+str(ii)] = beta2*update["vw"+str(ii)] + (1-beta2)*np.square(np.sum(grads["w"+str(ii)],axis=0))
            update["vb"+str(ii)] = beta2*update["vb"+str(ii)] + (1-beta2)*np.square(np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1))
            self.params["w"+str(ii)] -= np.multiply( (learning_rate / np.sqrt(epsilon + (update["vw"+str(ii)]/(1-beta2**step_num)))),\
                                        beta1*(update["mw"+str(ii)]/(1-beta1**step_num) +\
                                        ((1-beta1)/(1-beta1**step_num))*np.sum(grads["w"+str(ii)],axis=0) ))
            self.params["b"+str(ii)] -= np.multiply( (learning_rate / np.sqrt(epsilon + (update["vb"+str(ii)]/(1-beta2**step_num)))),\
                                        beta1*(update["mb"+str(ii)]/(1-beta1**step_num) +\
                                        ((1-beta1)/(1-beta1**step_num))*np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1) ))
        return update

    def do_back_prop(self,X,Y,X_cv,Y_cv,optimiser,gamma,numepochs,learning_rate,batch_size,beta,epsilon,beta1,beta2):
        layers = len(self.structure)-1
        update = {}
        for i in range(1,layers+1):
            update["w"+str(i)]=np.zeros(self.params["w"+str(i)].shape)
            update["b"+str(i)]=np.zeros(self.params["b"+str(i)].shape) 
            update["mw"+str(i)]=np.zeros(self.params["w"+str(i)].shape)
            update["mb"+str(i)]=np.zeros(self.params["b"+str(i)].shape)
            update["vw"+str(i)]=np.zeros(self.params["w"+str(i)].shape)
            update["vb"+str(i)]=np.zeros(self.params["b"+str(i)].shape) 
            #TODO: remove update initialisation. Pass {}. Enforce dict.get(key,0)
        step_count = 0
        for i in range(numepochs):
            for j in tqdm(range(math.ceil(X.shape[1]/batch_size))):
                X_pass = X[:,j*batch_size:min(X.shape[1],(j+1)*batch_size)]
                Y_pass = Y[j*batch_size:min(X.shape[1],(j+1)*batch_size)]
                step_count +=1
                if optimiser == "sgd":
                    update = self.do_sgd(X_pass,Y_pass,update,learning_rate)
                elif optimiser == "momentum":
                    update = self.do_momentum(X_pass,Y_pass,update,gamma,learning_rate)
                elif optimiser == "nesterov":
                    update = self.do_nesterov(X_pass,Y_pass,update,gamma,learning_rate)
                elif optimiser == "rmsprop":
                    update = self.do_rmsprop(X_pass,Y_pass,update,learning_rate,beta,epsilon)
                elif optimiser == "adam":
                    update = self.do_adam(X_pass,Y_pass,update,learning_rate,epsilon,beta1,beta2,step_count)
                elif optimiser == "nadam":
                    update = self.do_nadam(X_pass,Y_pass,update,learning_rate,epsilon,beta1,beta2,step_count)
                else:
                    raise(ValueError("Unknown optimiser \"" + optimiser + "\""))
                Y_pred = self.predict(X)
                self.accuracies.append(np.mean(np.argmax(Y_pred,axis=0)==Y))
                self.cvaccuracies.append(np.mean(self.predict(X_cv,returnclass=1)==Y_cv))
                self.losses.append(np.sum(-np.log(np.choose(Y,Y_pred))))
                wandb.log({"train_acc":self.accuracies[-1],"train_loss":self.losses[-1],"cv_acc":self.cvaccuracies[-1]})


    def train(self,X,Y,numepochs = 100,learning_rate = 0.1,initialization_type = "random",activation = "sigmoid",\
              optimiser = "sgd",gamma=0.1,init_params=True,train_test_split=0.2,seed=3,batch_size = 32,beta=0.99,\
              epsilon=0.0000001,beta1=0.9,beta2=0.999):
        if init_params == False and self.params == {}:
            raise(UnboundLocalError("Weights and Biases not initialized. Set init_params to True."))
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
        self.do_back_prop(X,Y,X_cv,Y_cv,optimiser,gamma,numepochs,learning_rate,batch_size,beta,epsilon,beta1,beta2)

    
    def predict(self,X,returndict = 0,returnclass = 0):
        predictions = X
        if returndict == 1:
            values = {}
            values["h0"] = X
        layers = len(self.structure)-1
        for i in range(layers-1):
            predictions = np.matmul( self.params["w"+str(i+1)], predictions) + self.params["b"+str(i+1)]
            if returndict == 1:    
                values["a"+str(i+1)]=predictions
            predictions = NeuralNet.activate(predictions,self.activation)
            if returndict == 1:
                values["h"+str(i+1)]=predictions
        predictions = np.matmul( self.params["w"+str(layers)], predictions) + self.params["b"+str(layers)]
        if returndict == 1:
            values["a"+str(layers)]=predictions
        if returnclass == 1:
            return np.argmax(predictions,axis=0)
        np.clip(predictions,-700,700)
        predictions = np.exp(predictions)/np.sum(np.exp(predictions),axis=0)
        if returndict == 1:
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
wandb.config.update({"dataset":"fashion_mnist","input_size":784,"output_size":10,"hidden_layers":[128,64,64,32],"epochs":3,\
                     "learning_rate":0.001,"batch_size":64,"initialization_type":"xavier","activation":"sigmoid",\
                "optimiser":"nadam","gamma":0.1,"train_test_split":0.2,"seed":7,"beta":0.99,"epsilon":0.0000001,\
                "beta1":0.9,"beta2":0.999})

# %%
nn = NeuralNet(wandb.config["input_size"],wandb.config["output_size"])
for hidden_layer_size in wandb.config["hidden_layers"]:
    nn.addlayer(hidden_layer_size)
nn.train(X_train,Y_train,wandb.config["epochs"],wandb.config["learning_rate"],\
         initialization_type=wandb.config["initialization_type"],activation=wandb.config["activation"],optimiser=wandb.config["optimiser"],\
         gamma=wandb.config["gamma"],batch_size=wandb.config["batch_size"],train_test_split=wandb.config["train_test_split"],seed=wandb.config["seed"],\
         beta=wandb.config["beta"],epsilon=wandb.config["epsilon"],beta1=wandb.config["beta1"],beta2=wandb.config["beta2"])

# %%
current_wandb_run.finish()

# %%
