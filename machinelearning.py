"""
Jarvis Neural Network Classifier
Author: animo1738
Reference: Based on the tutorial by Bernardino Sassoli at https://towardsdatascience.com/building-a-neural-network-from-scratch-8f03c5c50adc/
Date: 18/01/2026
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from typing import List
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# progress bars in terminal

# --- 1. ACTIVATION FUNCTIONS ---
# some common activation functions 
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))
# takes real values and produces a value between 0 and 1 
def relu(z: np.ndarray)  -> np.ndarray:
    return np.maximum(0, z)
#simply outputs the input if it is positive, and if the input is negative, outputs it as 0
def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)
# similar to sigmoid takes real values and squashes it between -1 and 1
def leaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, z, z*0.01)
#extension of relu, allows negative values to leak in, default is 0.1 of negative input

#output function:
def softmax(z):
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)
# takes numerical inputs and outputs a probability distribution for the digits it could be
#derives function based on names
def derivative(function_name: str, z: np.ndarray) -> np.ndarray:
    if function_name == "sigmoid" : 
        return sigmoid(z) * (1-sigmoid(z))
    if function_name == "tanh":
        return 1 - np.square(tanh(z))
    if function_name == "relu":
        y = (z>0) * 1
        return y
    if function_name == "leaky_relu":
        return np.where(z > 0, 1, 0.01)
    

    return "No such activation"

# --- 2. UTILITY FUNCTIONS ---
def normalise(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))
#scales each input value of the images between 0 and 1 

def one_hot_encode(x : np.ndarray, num_labels: int) ->np.ndarray:
    return np.eye(num_labels)[x]
# connects labels(numbers 0 to 9) to a corresponding matrix 
# matrix contains rows for each labels, where a 1 is at the index corresponding to the labels value

class NN:
    def __init__(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, activation: str, num_labels: int, architecture: List[int]):
            self.X = normalise(X) # normalize training data in range 0,1
            assert np.all((self.X >= 0) | (self.X <= 1)) # test that normalize succeded
            self.X, self.X_test = X.copy(), X_test.copy()
            self.y, self.y_test = y.copy(), y_test.copy()
            self.layers = {} # define dict to store results of activation
            self.architecture = architecture # size of hidden layers as array  
            self.activation = activation # activation function
            assert self.activation in ["relu", "tanh", "sigmoid", "leaky_relu"]
            self.parameters = {}
            self.num_labels = num_labels
            self.m = X.shape[1]
            self.architecture.append(self.num_labels)
            self.num_input_features = X.shape[0]
            self.architecture.insert(0, self.num_input_features)
            self.L = len(architecture) 
            assert self.X.shape == (self.num_input_features, self.m)
            assert self.y.shape == (self.num_labels, self.m)
            # when this is callled, features of the images are normalised (between 0 and 1)
            # adds a neuron layer with size 28x28 with the size of 10, the number of classes(0-9)
            # initialises an empty dictionary to store biases and weights
    def initialize_parameters(self,log_func = None): 
        for i in range(1, self.L):
            if log_func:
                log_func(f"Initialising parameters for layer: {i}")
            
            self.parameters["w"+str(i)] = np.random.randn(self.architecture[i], self.architecture[i-1]) * 0.01
            self.parameters["b"+str(i)] = np.zeros((self.architecture[i], 1))
            # generates a number between 0 and 1 from a bell curve
            #creates a 2D grid storing all of these random weights/biases for the 28x28 images to be inputted

    #this subroutine basically lets the network make a blind guess
    # uses the current weights and biases to calculate an answer
    def forward(self):
        params = self.parameters
        self.layers["a0"] = self.X
        for l in range(1, self.L - 1):
            self.layers["z" + str(l)] =np.dot(params["w" + str(l)],
                                            self.layers["a" + str(l-1)]) + params["b" + str(l)]
            self.layers["a" + str(l)] = eval(self.activation)(self.layers["z"+str(l)])
            assert self.layers["a" + str(l)].shape == (self.architecture[l], self.m)
        self.layers["z" + str(self.L - 1)] = np.dot(params["w" + str(self.L-1)],
                                                    self.layers["a"+str(self.L-2)]) + params["b"+str(self.L-1)]
        self.layers["a"+str(self.L-1)] = softmax(self.layers["z"+str(self.L-1)])
        self.output = self.layers["a" + str(self.L-1)]
        assert self.output.shape == (self.num_labels, self.m)
        assert all([s for s in np.sum(self.output,axis = 1)])

        cost = -np.sum(self.y * np.log(self.output + 0.000000001))
        
        return cost, self.layers
    # basically does this:
    # inputs all 784 pixel values
    # completes formula --> Z = (W*X) + b
    # runs the activation function
    # outputs values using softmax, so probabilities for the numbers between 0 and 9 
    # calculates how far off the random prediction was to nudge the next prediction in a better direction
    # stores the values of the layers to later perform backpropagation 
        #(understanding how it got to that value) 

    def backpropagate(self):
        derivatives = {}
        #dictionary of derivatives being used
        dZ = self.output - self.y
        #difference between the output and the real values 
        # the 'error'
        assert dZ.shape == (self.num_labels, self.m)
        dW = np.dot(dZ, self.layers["a" + str(self.L-2)].T) / self.m
        # Weight derivative: uses chain rule to calculate error while including the previous later
        db = np.sum(dZ, axis = 1, keepdims = True) / self.m
        # Bias derivative: uses chain rule including previous layer
        dAPrev = np.dot(self.parameters["w" + str(self.L-1)].T, dZ)
        # fixes the previous layer, by moving backwards (transpose) and multiply the current error (dZ) and the weights connecting them
        derivatives["dW" + str(self.L - 1)] = dW
        derivatives["db" + str(self.L - 1)] = db
        # adds the calculated derivatives to the dictionary

        for l in range(self.L - 2, 0, -1):
            dZ = dAPrev * derivative(self.activation, self.layers["z" + str(l)])
            dW = 1. / self.m * np.dot(dZ, self.layers["a" + str(l-1)].T)
            db = 1. / self.m * np.sum(dZ, axis=1, keepdims = True)
            if l > 1:
                dAPrev = np.dot(self.parameters["w" + str(l)].T, (dZ))
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
            #adds a log for the derivative dictionary 

        self.derivatives = derivatives
        return self.derivatives
    #runs the tests and sets parameters
    def fit(self, lr = 0.01, epochs = 1000,log_func = None, border_func=None, update_ui_func = None):
        # where lr is the size of adjustments for the weights 
        # epochs are the number of iterations of the whole dataset
        self.costs = []
        self.initialize_parameters()
        self.accuracies = {"train": [], "test": []}
        train_acc = self.accuracy(self.X, self.y)
        test_acc = self.accuracy(self.X_test, self.y_test)
        

        for epoch in tqdm(range(epochs), colour = "BLUE"):
            cost, cache = self.forward()
            self.costs.append(cost)
            derivatives = self.backpropagate()
            for layer in range(1, self.L):
                self.parameters["w"+str(layer)] = self.parameters["w"+str(layer)] - lr * derivatives["dW" + str(layer)]
                self.parameters["b"+str(layer)] = self.parameters["b"+str(layer)] - lr * derivatives["db" + str(layer)]            
            train_accuracy = self.accuracy(self.X, self.y)
            test_accuracy = self.accuracy(self.X_test, self.y_test)
            self.accuracies["train"].append(train_accuracy)
            self.accuracies["test"].append(test_accuracy)

            if update_ui_func:
                stats = {
                    "epoch": epoch,
                    "cost": cost,
                    "train_acc": train_accuracy,
                    "test_acc" : test_accuracy,
                    "nn_instance" : self 
                }
                update_ui_func(stats)
            if epoch % 10 == 0:
                acc = self.accuracy(self.X, self.y)
            
            if border_func:
                border_func(acc)
                
            self.accuracies["train"].append(train_accuracy)
            self.accuracies["test"].append(test_accuracy)
        if log_func:
            log_func("Training terminated")
        return self

        

    def predict(self, x):
        params = self.parameters
        n_layers = self.L - 1
        values = [x]
        for l in range(1, n_layers):
            z = np.dot(params["w" + str(l)], values[l-1]) + params["b" + str(l)]
            a = eval(self.activation)(z)
            values.append(a)
        z = np.dot(params["w" + str(n_layers)], values[n_layers - 1]) + params["b"+str(n_layers)]
        a = softmax(z)
        if x.shape[1] > 1:
            ans = np.argmax(a, axis = 0)
        else: 
            ans = np.argmax(a)
        return ans     

    def accuracy(self, X, y):
        P = self.predict(X)
        return sum(np.equal(P, np.argmax(y, axis=0))) / y.shape[1]*100
    
    # def plot_cost(self):
    #     plt.plot(self.costs)
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Cost")
    #     plt.title("Learning Progress")
    #     plt.show()
 




mnist = fetch_openml(name="mnist_784")
#database of grayscale images of handwritted digits
#784 'features' as they are 28x28 sized images of handwritted images 


data = mnist.data
labels = mnist.target


n = np.random.choice(np.random.choice(data.shape[0]))
#the random choice of the dataset

test_img = data.iloc[n].values
test_label = mnist.target.iloc[n]
#the image of the handwrittem digit and its label assigned to it 

print(test_img.shape) 

side_length = int(np.sqrt(test_img.shape))
reshaped_test_img = test_img.reshape(side_length, side_length)
#gets img length and reshapes it to a standard square






def execute_llm(log_func, update_border,update_ui_func,adjustedEpochs, adjustedLr):
    # --- 4. DATA PREPARATION & EXECUTION ---
    
    log_func("Fetching MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data.T, mnist.target.astype(int)

    # Split
    X_train, X_test = X[:, :60000], X[:, 60000:]
    y_train_raw, y_test_raw = y[:60000], y[60000:]

    # Encode labels
    y_train = one_hot_encode(y_train_raw, 10).T
    y_test = one_hot_encode(y_test_raw, 10).T

    # Initialize and Train
    # Architecture: [Hidden1, Hidden2] -> Output is added automatically
    PARAMS = [X_train, y_train, X_test, y_test, "relu", 10, [128, 32]]
    nn_relu = NN(*PARAMS)
    nn_relu.fit(lr=adjustedLr, epochs=adjustedEpochs, log_func=log_func, update_ui_func=update_ui_func)
    print(adjustedLr, adjustedEpochs)
    return nn_relu




if __name__ == "__main__":
    execute_llm()
