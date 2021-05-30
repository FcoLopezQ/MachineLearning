# ===========================================================================
# Neural Network By Hand - Training of Model
# Author: Francisco Iván López Quihuis
# 
# References: 
# Code Reference:
# Zhang, Samson (2020) Building a neural network FROM SCRATCH. Youtube:
# https://www.youtube.com/watch?v=w8yWXqWQYmU&ab_channel=SamsonZhang
# 
# DataSet Reference: 
# Kaggle (n.d.) Digit Recognizer, MNIST Data. Kaggle:
# https://www.kaggle.com/c/digit-recognizer
# ===========================================================================
import numpy as np 
import pandas as pd 

#Import Train Data (MNIST)
data = pd.read_csv('train.csv')
#Turn imported data into numpy array
data = np.array(data)
m, n = data.shape
#Shuffle data to get random training samples
np.random.shuffle(data)

#Separate Training Data
data_train = data[20000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
#Normalize inputs (Pixel Values 0-255)
X_train = X_train / 255.

#Obtain number of samples
_,m_train = X_train.shape

#Initialize Matrices that will hold the Weights Value
def initParameters():
    #Initialize with values from 0 to 1
    #W1 -> 1st Hidden Layer (10 neurons with 784 inputs each one)
    W1 = np.random.rand(10, 784) - 0.5
    #b1 -> Biases for 1st Hidden Layer (10 neurons)
    b1 = np.random.rand(10, 1) - 0.5
    
    #W2 -> Output Layer (10 neurons with 10 inputs each one)
    W2 = np.random.rand(10, 10) - 0.5
    #b2 -> Biases for Output Layer (10 neurons)
    b2 = np.random.rand(10, 1) - 0.5
    
    return W1,b1,W2,b2

#ReLU funcition
#If value is less than 0, 0 is returned
def ReLU(Z):
    return np.maximum(Z, 0)
    
#Softmax function
#Normalized Exponential function
#Turns value into a 0-1 range
def Softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))
    
#Output Values for 1st Hidden Layer and Output Layer are calculated
#based on the input values (from the image)
def forwardPropagation(W1, b1, W2, b2, X):
    # Raw values for 1st Hidden Layer
    Z1 = W1.dot(X) + b1
    # ReLU Function applied to 1st Hidden Layer
    A1 = ReLU(Z1)
    # Raw values for Output Layer
    Z2 = W2.dot(A1) + b2
    # Softmax function applied to Output Layer
    A2 = Softmax(Z2)
    return Z1, A1, Z2, A2

# One hot encoding applied to the Expected values (Labels)
def one_hot(Y):
    # Dimension -> (Number of samples, number of different values (0 to 9))
    # Dimension -> (samples, 10)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    # Where Label corresponds to 1, it is assigned
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Transpose
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

#Returns Values of Derivative of ReLU function
# If Z > 0 -> Slope (derivate is 1)
# Else it is zero
def derivative_ReLU(Z):
    return Z > 0
    
# Back Propagation Function
def backPropagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    # One Hot Encoding for output values
    one_hot_Y = one_hot(Y)
    # Total Error for Outputs
    # Output - Target
    dZ2 = A2 - one_hot_Y
    # Update of Weights for Output Layer
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    # Update of Weights for Hidden Layer
    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Update Parameters
def updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# Get the output value with the higher score
def get_predictions(A2):
    return np.argmax(A2,0)

# Calculate percentage of correct predictions
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent Implementation
def gradient_descent(X, Y, epochs, alpha):
    W1, b1, W2, b2 = initParameters()
    for i in range(epochs):
        #Forward Propagation
        Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, X)
        #Back Propagation
        dW1, db1, dW2, db2 = backPropagation(Z1, A1, Z2, A2, W1, W2, X, Y)
        #Update Weights
        W1, b1, W2, b2 = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        #Print Epoch and Accuracy
        if i % 100 == 0:
            print("Iteration: ", i)
            print("Accuracy (%): ", get_accuracy(get_predictions(A2), Y)*100)
    
    print("Iteration: ", i)
    print("Accuracy(%): ", get_accuracy(get_predictions(A2), Y)*100)
    return W1, b1, W2, b2

# Train Neural Network
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1000, 0.1)

# Save Weights
np.savez('weights.npz',W1,b1,W2,b2)


