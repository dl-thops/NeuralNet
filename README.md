# NeuralNet
## Table of Contents
1. [Introduction](https://github.com/whitezeta/cs6910_a1#introduction)
2. [Download](https://github.com/whitezeta/cs6910_a1#download)
3. [Quick Start](https://github.com/whitezeta/cs6910_a1#quick-start)
4. [Features](https://github.com/whitezeta/cs6910_a1#features)
5. [Project Report](https://github.com/whitezeta/cs6910_a1#project-report)

## Introduction
This repository contains an implementation of a Feed Forward Neural Network from scratch using numpy libraries. We have achieved a testing accuracy of 97.45% on MNIST Dataset and a 88.80.% testing accuracy on Fashion-MNIST Dataset.

You can also find a GPU version of the class NeuralNet in [ctrain<span>.py</span>](https://github.com/whitezeta/cs6910_a1/blob/master/ctrain.py) (Uses cupy instead of numpy(CuDa compatible)). We have found about 50~100 x speed boost in training time. We will release the cupy version module soon.

## Download
You can view the source code for the NeuralNet class implementation from this [page](https://github.com/whitezeta/cs6910_a1/blob/master/neuralnet.py). Simply download the file neuralnet<span>.py</span> from this repository and follow the quick start code to use the NeuralNet.

The module requires installation of following packages

 - numpy
 
  ```$ pip install numpy```
 - tqdm
 
  ```$ pip install tqdm```
 - pickle
 
  ```$ pip install pickle```
 - wandb (Optional)
 
  ```$ pip install wandb```


## Quick Start
#### Training
```
from neuralnet import NeuralNet
from keras.datasets import mnist

# Import and Preprocess Data
( X_train, Y_train), ( X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],-1).T
X_test = X_test.reshape(X_test.shape[0],-1).T

nn = NeuralNet( input_size = 784, output_size = 10)
nn.addlayer(128)
nn.addlayer(64)
nn.train( X_train, Y_train, numepochs = 10, learning_rate = 0.001)
```
#### Prediction
```
nn.predict( X_test, returnclass = 1)
# Set returnclass = 0 for class probabilities
```
#### Saving a Model
```
nn.save_model( "my_model.bin")
```
#### Loading a Saved Model
```
nn = NeuralNet.load_model( "my_model.bin")
```

## Features
The NeuralNet class has support for the following features/parameters support:
- Activation Type
	- [Tanh](https://mathworld.wolfram.com/HyperbolicTangent.html)
	- [Relu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
	- [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)
- Optimisers
	- [Nadam](https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ)
	- [Rmsprop]()
	- [Adam](https://arxiv.org/abs/1412.6980)
	- [Nesterov](https://paperswithcode.com/method/nesterov-accelerated-gradient)
	- [Momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum)
	- [SGD](https://en.wikipedia.org/wiki/Gradient_descent)
- Variable batch sizes ( Batch Gradient Descent )
- Train - Cross Validation split
- L2 Regularisation / Weight Decay

## Project Report
This repository is a part of our Course Assignment ([CS6910](http://www.cse.iitm.ac.in/~miteshk/#teaching) - Fundamentals of Deep Learning). 

We have trained deep neural networks using this NeuralNet module and summarised our findings in this [W&B Report]()
