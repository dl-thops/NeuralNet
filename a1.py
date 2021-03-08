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
import wandb

# %%
wandb.login()
current_wandb_run = wandb.init(project = "fdl-a1",entity = "fdl-thops",config = {"dataset":"fashion_mnist","input_size":784,\
                    "output_size":10,"hidden_layers":[16,16],"epochs":1,"learning_rate":0.01,\
                    "batch_size":128,"initialization_type":"xavier","activation":"sigmoid",\
                    "optimiser":"nadam","gamma":0.1,"train_test_split":0.2,"seed":None,"beta":0.99,\
                    "epsilon":0.0000001,"beta1":0.9,"beta2":0.999,"l2_reg_param":0,"init_params":True,\
                    "hidden_1":32,"hidden_2":32,"hidden_3":32,"hidden_4":32,"hidden_5":32})


# %%
class NeuralNet:
    """
    This class implements feed-forward neural-networks and back-propagation using a variety of algorithms.
    Methods
    -------
    sigmoid(numpy.ndarray) : numpy.ndarray

    tanh(numpy.ndarray) : numpy.ndarray

    relu(numpy.ndarray) : numpy.ndarray

    activate( numpy.ndarray, string) : numpy.ndarray

    addlayer(int) : None

    initialise_params(initialization_type = "random") : None

    activation_gradient( numpy.ndarray, string) : numpy.ndarray

    calculate_grads( numpy.ndarray, numpy.ndarray, float) : dict

    do_sgd( numpy.ndarray, numpy.ndarray, dict, float, dict) : dict

    do_momentum( numpy.ndarray, numpy.ndarray, dict, float, dict) : dict

    do_nesterov( numpy.ndarray, numpy.ndarray, dict, float, dict) : dict

    do_rmsprop( numpy.ndarray, numpy.ndarray, dict, float, dict) : dict

    do_adam( numpy.ndarray, numpy.ndarray, dict, float, dict) : dict

    do_nadam( numpy.ndarray, numpy.ndarray, dict, float, dict) : dict

    get_loss( numpy.ndarray, numpy.ndarray, float, numpy.ndarray) : float

    do_back_prop( numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, string, float, int, float, int, float, float, float, float, float) : None

    train( numpy.ndarray, numpy.ndarray, int, float, string, string, string, float, bool, float, int, int, float, float, float, float, float) : None
    
    predict( numpy.ndarray, int, int) : numpy.ndarray or dict
    """
    @staticmethod
    def sigmoid(X):
        """
        Calculates the sigmoid of given input.
        Parameters
        ----------
        X : numpy.ndarray
        Returns
        -------
        numpy.ndarray
        """
        X = np.clip( X, -700, 700)
        return 1 / (1. + np.exp(-X))
            
    @staticmethod
    def tanh(X):
        """
        Calculates the tanh of given input.
        Parameters
        ----------
        X : numpy.ndarray
        Returns
        -------
        numpy.ndarray
        """
        X = np.clip( X, -350, 350)
        return (1 - np.exp(-2*X)) / (1 + np.exp(-2*X))
        
        
    @staticmethod
    def relu(X):
        """
        Calculates the relu of given input.
        Parameters
        ----------
        X : numpy.ndarray
        Returns
        -------
        numpy.ndarray
        """
        return np.where( X<0, 0, X)
        
    @staticmethod
    def activate( X, activation = "sigmoid"):
        """
        Applies the given activation function to given input.
        Parameters
        ----------
        X : numpy.ndarray
        activation : string, default= "sigmoid"
        Returns
        -------
        numpy.ndarray
        Raises
        ------
        ValueError
            If activation function is not known.
        """
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
        self.optimisers = {"sgd":self.do_sgd,"momentum":self.do_momentum,"nesterov":self.do_nesterov,\
                           "rmsprop":self.do_rmsprop,"adam":self.do_adam,"nadam":self.do_nadam}
        
    def addlayer( self, layer_size):
        """
        Adds a hidden layer to current NeuralNet.
        Parameters
        ----------
        self : NeuralNet Class Instance
        layer_size : int
        Returns
        -------
        None
        """
        self.structure = self.structure[:-1] + [ layer_size, self.structure[-1]]
        
    def initialise_params( self, initialization_type = "random"):
        """
        Initialises all weights and biases according to given initialization type.
        Parameters
        ----------
        self : NeuralNet Class Instance
        initialization_type : string, default= "random"
        Returns
        -------
        None
        """
        self.init_type = initialization_type
        if self.init_type == "random":
            for i in range( 1, len(self.structure)):
                self.params["w"+str(i)] = np.random.rand( self.structure[i], self.structure[i-1]) - 0.5
                self.params["b"+str(i)] = np.random.rand( self.structure[i], 1) - 0.5
        elif self.init_type == "xavier":
            for i in range(1,len(self.structure)):
                self.params["w"+str(i)] = np.random.normal(0,1/np.sqrt(self.structure[i-1]+\
                                                    self.structure[i]),(self.structure[i],self.structure[i-1]))
                self.params["b"+str(i)] = np.random.normal(0,1/np.sqrt(self.structure[i-1]+\
                                                    self.structure[i]),(self.structure[i],1))
        else:
            print(self.init_type + ": unidentified initialization type")    
    
    @staticmethod
    def activation_gradient( A, activation = "sigmoid"):
        """
        Applies the derivative function of the given activation function to given input.
        Parameters  
        ----------
        A : numpy.ndarray
        activation : string, default: "sigmoid"
        Returns
        -------
        numpy.ndarray
        Raises
        ------
        ValueError
            If activation function is not known. 
        """
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

    def calculate_grads( self, X, Y, l2_reg_param):
        """
        Calculates the gradients of loss function w.r.t pre-activations,activations,weights and biases in all layers.
        Parameters
        ----------
        self : NeuralNet Class Instance
        X : numpy.ndarray
        Y : numpy.ndarray
        l2_reg_param : float
        Returns
        -------
        dict
        """
        grads = {}
        values = self.predict(X,returndict=1)
        layers = len(self.structure)-1
        nsamples = X.shape[1]
        grads["a"+str(layers)] = -(np.eye(self.structure[-1])[Y]).T + values["h"+str(layers)]
        for ii in np.arange(layers-1,0,-1):
            grads["h"+str(ii)] = np.matmul(self.params["w"+str(ii+1)].T,grads["a"+str(ii+1)])
            grads["a"+str(ii)] = np.multiply(grads["h"+str(ii)],self.activation_gradient(values["h"+str(ii)],self.activation))
        for ii in np.arange(layers,0,-1):
            grads["w"+str(ii)] = np.matmul((grads["a"+str(ii)].T).reshape(nsamples,-1,1),\
                                    (values["h"+str(ii-1)].T).reshape(nsamples,1,-1)) \
                                    + (l2_reg_param/nsamples) * self.params["w"+str(ii)]
            grads["b"+str(ii)] = grads["a"+str(ii)]
        return grads

    def do_sgd( self, X, Y, update, learning_rate, **kwargs):
        layers = len(self.structure)-1
        grads = self.calculate_grads(X,Y,kwargs["l2_reg_param"])
        for ii in np.arange(1,layers+1):
            self.params["w"+str(ii)] -= learning_rate * np.sum(grads["w"+str(ii)],axis=0)
            self.params["b"+str(ii)] -= learning_rate * np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)
        return update

    def do_momentum( self, X, Y, update, learning_rate, **kwargs):
        layers = len(self.structure)-1
        grads = self.calculate_grads(X,Y,kwargs["l2_reg_param"])
        for ii in np.arange(1,layers+1):
            update["w"+str(ii)] = kwargs["gamma"] * update.get("w"+str(ii),0) + learning_rate * np.sum(grads["w"+str(ii)],axis=0)
            update["b"+str(ii)] = kwargs["gamma"] * update.get("b"+str(ii),0) + learning_rate * np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)
            self.params["w"+str(ii)] -= update["w"+str(ii)]
            self.params["b"+str(ii)] -= update["b"+str(ii)]
        return update

    def do_nesterov( self, X, Y, update, learning_rate, **kwargs):
        layers = len(self.structure)-1
        for ii in range(1,layers+1):
            self.params["w"+str(ii)] -= kwargs["gamma"] * update.get("w"+str(ii),0)
            self.params["b"+str(ii)] -= kwargs["gamma"] * update.get("b"+str(ii),0)
        grads = self.calculate_grads(X,Y,kwargs["l2_reg_param"])
        for ii in np.arange(1,layers+1):
            update["w"+str(ii)] = kwargs["gamma"] * update.get("w"+str(ii),0) + learning_rate * np.sum(grads["w"+str(ii)],axis=0)
            update["b"+str(ii)] = kwargs["gamma"] * update.get("b"+str(ii),0) + learning_rate * np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)
            self.params["w"+str(ii)] -= update["w"+str(ii)]
            self.params["b"+str(ii)] -= update["b"+str(ii)]
        return update

    def do_rmsprop( self, X, Y, update, learning_rate, **kwargs):
        layers = len(self.structure)-1
        grads = self.calculate_grads(X,Y,kwargs["l2_reg_param"])
        for ii in np.arange(1,layers+1):
            update["w"+str(ii)] = kwargs["beta"]*update.get("w"+str(ii),0) + (1-kwargs["beta"])*np.square(np.sum(grads["w"+str(ii)],axis=0))
            update["b"+str(ii)] = kwargs["beta"]*update.get("b"+str(ii),0) + (1-kwargs["beta"])*np.square(np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1))
            self.params["w"+str(ii)] -= np.multiply((learning_rate/ np.sqrt(kwargs["epsilon"] + update["w"+str(ii)])),\
                                                    np.sum(grads["w"+str(ii)],axis=0))
            self.params["b"+str(ii)] -= np.multiply((learning_rate / np.sqrt(kwargs["epsilon"] + update["b"+str(ii)])),\
                                                    np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1))
        return update

    def do_adam( self, X, Y, update, learning_rate, **kwargs):
        layers = len(self.structure)-1
        grads = self.calculate_grads(X,Y,kwargs["l2_reg_param"])
        for ii in np.arange(1,layers+1):
            update["mw"+str(ii)] = kwargs["beta1"]*update.get("mw"+str(ii),0) + (1-kwargs["beta1"])*np.sum(grads["w"+str(ii)],axis=0)
            update["mb"+str(ii)] = kwargs["beta1"]*update.get("mb"+str(ii),0) + (1-kwargs["beta1"])*np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)
            update["vw"+str(ii)] = kwargs["beta2"]*update.get("vw"+str(ii),0) + (1-kwargs["beta2"])*np.square(np.sum(grads["w"+str(ii)],axis=0))
            update["vb"+str(ii)] = kwargs["beta2"]*update.get("vb"+str(ii),0) + (1-kwargs["beta2"])*np.square(np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1))
            self.params["w"+str(ii)] -= np.multiply((learning_rate/np.sqrt(kwargs["epsilon"] + (update["vw"+str(ii)]/(1-kwargs["beta2"]**kwargs["step_num"])))) ,\
                                                    update["mw"+str(ii)]/(1-kwargs["beta1"]**kwargs["step_num"]))
            self.params["b"+str(ii)] -= np.multiply((learning_rate/np.sqrt(kwargs["epsilon"] + (update["vb"+str(ii)]/(1-kwargs["beta2"]**kwargs["step_num"])))) ,\
                                                    update["mb"+str(ii)]/(1-kwargs["beta1"]**kwargs["step_num"]))
        return update
    
    def do_nadam( self, X, Y, update, learning_rate, **kwargs):
        layers = len(self.structure)-1
        grads = self.calculate_grads(X,Y,kwargs["l2_reg_param"])
        for ii in np.arange(1,layers+1):
            update["mw"+str(ii)] = kwargs["beta1"]*update.get("mw"+str(ii),0) + (1-kwargs["beta1"])*np.sum(grads["w"+str(ii)],axis=0)
            update["mb"+str(ii)] = kwargs["beta1"]*update.get("mb"+str(ii),0) + (1-kwargs["beta1"])*np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1)
            update["vw"+str(ii)] = kwargs["beta2"]*update.get("vw"+str(ii),0) + (1-kwargs["beta2"])*np.square(np.sum(grads["w"+str(ii)],axis=0))
            update["vb"+str(ii)] = kwargs["beta2"]*update.get("vb"+str(ii),0) + (1-kwargs["beta2"])*np.square(np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1))
            self.params["w"+str(ii)] -= np.multiply( (learning_rate / np.sqrt(kwargs["epsilon"] + (update["vw"+str(ii)]/(1-kwargs["beta2"]**kwargs["step_num"])))),\
                                        kwargs["beta1"]*(update["mw"+str(ii)]/(1-kwargs["beta1"]**kwargs["step_num"]) +\
                                        ((1-kwargs["beta1"])/(1-kwargs["beta1"]**kwargs["step_num"]))*np.sum(grads["w"+str(ii)],axis=0) ))
            self.params["b"+str(ii)] -= np.multiply( (learning_rate / np.sqrt(kwargs["epsilon"] + (update["vb"+str(ii)]/(1-kwargs["beta2"]**kwargs["step_num"])))),\
                                        kwargs["beta1"]*(update["mb"+str(ii)]/(1-kwargs["beta1"]**kwargs["step_num"]) +\
                                        ((1-kwargs["beta1"])/(1-kwargs["beta1"]**kwargs["step_num"]))*np.sum(grads["b"+str(ii)],axis=1).reshape(-1,1) ))
        return update

    def get_loss(self,X,Y,l2_reg_param=0,Y_pred=None):
        weight_sum=0
        for ii in range(1,len(self.structure)):
            weight_sum += np.sum(np.square(self.params["w"+str(ii)]))
        if Y_pred is None:
            Y_pred = self.predict(X)
        return (np.sum(-np.log(np.choose(Y,Y_pred))) + (l2_reg_param/2)*weight_sum) / len(Y)
        

    def do_back_prop(self,X,Y,X_cv,Y_cv,optimiser,gamma,numepochs,learning_rate,batch_size,beta,epsilon,beta1,beta2,l2_reg_param):
        layers = len(self.structure)-1
        update = {}
        step_count = 0
        for i in range(numepochs):
            wandb.log({"Sample Data":[wandb.Image(X[:,jj].reshape(28,28),caption=dataset_labels[Y[jj]])\
                                      for jj in range(20*i,20*i+20)]},commit = False)
            for j in tqdm(range(math.ceil(X.shape[1]/batch_size))):
                X_pass = X[:,j*batch_size:min(X.shape[1],(j+1)*batch_size)]
                Y_pass = Y[j*batch_size:min(X.shape[1],(j+1)*batch_size)]
                step_count += 1
                update = (self.optimisers[optimiser])( X_pass, Y_pass, update, learning_rate, gamma = gamma, beta = beta,\
                        beta1 = beta1, beta2 = beta2, epsilon = epsilon, l2_reg_param = l2_reg_param, step_num = step_count)
                Y_pred = self.predict(X)
                self.accuracies.append(np.mean(np.argmax(Y_pred,axis=0)==Y))
                self.cvaccuracies.append(np.mean(self.predict(X_cv,returnclass=1)==Y_cv))
                self.losses.append(self.get_loss(None,Y,l2_reg_param,Y_pred))
                self.cvlosses.append(self.get_loss(X_cv,Y_cv,l2_reg_param))
                wandb.log({"train_acc":self.accuracies[-1],"train_loss":self.losses[-1],"val_acc":self.cvaccuracies[-1],\
                           "val_loss":self.cvlosses[-1],"step_count":step_count})


    def train(self,X,Y,numepochs = 100,learning_rate = 0.1,initialization_type = "random",activation = "sigmoid",\
              optimiser = "sgd",gamma=0.1,init_params=True,train_test_split=0.2,seed=None,batch_size = 32,beta=0.99,\
              epsilon=0.0000001,beta1=0.9,beta2=0.999,l2_reg_param=0):
        """
        Finds the values of weights and biases using back propagation.
        Parameters
        ----------
        X : numpy.ndarray
        Y : numpy.ndarray
        numepochs : int, default= 100
        learning_rate : float, default= 0.1
        initialization_type : string, default= "random"
        activation : string, default= "sigmoid"
        optimiser : string, default= "sgd"
        gamma : float, default= 0.1
        init_params : bool, default= True
            Specifies whether weights and biases should be initialised.
        train_test_split : float, default= 0.2
            Specifies the fraction of input data to be reserved for cross-validation.
        seed : int, default= None
            Used to seed the random splitting of input data.
        batch_size : int, default= 32
        beta : float, default= 0.99
        epsilon : float, default= 0000001
        beta1 : float, default= 0.9
        beta2 : float, default= 0.999
        l2_reg_param : float, default= 0
        Returns
        -------
        None
        """
        if seed is None:
            seed = np.random.randint(1,1000)
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
        self.cvlosses = []
        self.do_back_prop(X,Y,X_cv,Y_cv,optimiser,gamma,numepochs,learning_rate,batch_size,beta,epsilon,beta1,beta2,l2_reg_param)

    
    def predict(self,X,returndict = 0,returnclass = 0):
        """
        Predicts the probabilities of each class for the given input.
        Parameters
        ----------
        self : NeuralNet Class Instance
        X : numpy.ndarray
        returndict : int, default= 0
            The function will return a dictionary containing values of pre-activation and activation if returndict is 1.
        returnclass : int, default= 0
            The function will return the most probable class if returnClass is 1,else returns probability distribution over the output classes.
        Returns
        -------
        numpy.ndarray
            Probability distribution over the output classes for the given data. 
        or 
        numpy.ndarray
            Most probable class for the given data. 
        or
        dict
            Values of preactivation and activation of all layers.
        """
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
dataset_labels = { 0:"T-shirt/top", 1:"Trouser/pants", 2:"Pullover shirt", 3:"Dress", 4:"Coat",\
                5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}

# %%
nn = NeuralNet(wandb.config["input_size"],wandb.config["output_size"])
nn.addlayer(wandb.config["hidden_1"])
nn.addlayer(wandb.config["hidden_2"])
nn.addlayer(wandb.config["hidden_3"])
if wandb.config["hidden_4"] != 0 and wandb.config["hidden_5"] != 0:
    nn.addlayer(wandb.config["hidden_4"])
    nn.addlayer(wandb.config["hidden_5"])
elif wandb.config["hidden_4"] != 0 and wandb.config["hidden_5"] == 0:
    nn.addlayer(wandb.config["hidden_4"])
elif wandb.config["hidden_4"] == 0 and wandb.config["hidden_5"] == 0:
    pass
else:
    current_wandb_run.finish()

nn.train(X_train,Y_train,wandb.config["epochs"],wandb.config["learning_rate"],\
         initialization_type=wandb.config["initialization_type"],activation=wandb.config["activation"],optimiser=wandb.config["optimiser"],\
         gamma=wandb.config["gamma"],batch_size=wandb.config["batch_size"],train_test_split=wandb.config["train_test_split"],seed=wandb.config["seed"],\
         beta=wandb.config["beta"],epsilon=wandb.config["epsilon"],beta1=wandb.config["beta1"],beta2=wandb.config["beta2"],\
         l2_reg_param=wandb.config["l2_reg_param"],init_params=wandb.config["init_params"])

# %%
current_wandb_run.finish()

# %%