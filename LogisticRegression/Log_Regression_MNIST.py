# ===========================================================================
# Logistic Regression By Hand - Training of Model
# Author: Francisco Iván López Quihuis
# 
# References: 
# Code Reference:
# Valdés Aguirre, Benjamin (2021) perceptron vectorized.py 
# Ng, Andrew (n.d.) Deeplearning AI Course 
# 
# DataSet Reference: 
# Kaggle (n.d.) Digit Recognizer, MNIST Data. Kaggle:
# https://www.kaggle.com/c/digit-recognizer
# 
# ===========================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Obtain dataset from file
data = pd.read_csv("train.csv")
  
# X = feature values, all the columns except the first one
x = data.iloc[:, 1:]

# y = target values, first column of the data frame
y = data.iloc[:, 0]

# Obtain all instances that describe a number "0"
x_0 = x.loc[y == 0]
print("Number of '0' samples:")
print(x_0.shape)
# Obtain all instances that describe a number "1"
x_1 = x.loc[y == 1]
print("Number of '1' samples:")
print(x_1.shape)

# Obtain corresponding values in the output for "0" and "1"
y_0 = y.loc[y == 0]
y_1 = y.loc[y == 1]

# Concatenate all the instances for number "0" and "1"
x = pd.concat([x_0, x_1])
y = pd.concat([y_0, y_1])

# Normalize feature values
x = x / 255.

# Join Feature and Target values
df = pd.concat([x, y], axis = 1)

# Shuffle Randomly all the data and re-generate indeces
df = df.sample(frac=1)
df = df.reset_index()

# Delete first column which is for the index
df = df.iloc[:, 1:]

# Separate in Train and Test datasets
train=df.sample(frac=0.8, random_state =150) #Random state is a seed value
test=df.drop(train.index)

# Dimensions are checked in the Train and the Test Datasets
dimensions = train.shape[0]+test.shape[0]

# Check dimensions
if (dimensions == df.shape[0]):
  print("Dimensions are OK")

#Number of training and testing examples are printed
dim_train = train.shape[0]
dim_test = test.shape[0]
print ("Number of training examples: " + str(dim_train))
print ("Number of testing examples: " + str(dim_test))

print ('\n' + "-------------------------------------------------------" + '\n')

# Data is separated into Training and Testing Data for Features and Target
train_x = train.iloc[: , : -1]
train_y = train.iloc[: , -1]

test_x = test.iloc[: , : -1]
test_y = test.iloc[: , -1]

#========================== Functions =========================================

def sigmoid(z):
    s = 1./(1+np.exp(-z))
    return s


def initialize_with_zeros(dim):
    # Declaration of 'w' for weights and 'b' for bias
    w = np.zeros((dim,1))
    b = 0
    return w, b


def gradient_descent(w, b, X, Y):
    # Number of Samples
    m = X.shape[1]
    # Forward Propagation
    A = sigmoid(np.dot(w.T,X) + b)                            
    cost = (-1./m)*np.sum((-Y*np.log(A)) + ((1.-Y)*np.log(1.-A))) 
    # Backward Progagation
    dw = (1./m)*np.dot(X,(A-Y.T).T)
    db = (1./m)*np.sum(A-Y.T)
    # Check for correct size of parameters
    cost = np.squeeze(cost)
    
    return dw, db, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    # Array for costs
    costs = []
    for i in range(num_iterations):
        # Cost and gradient calculation
        dw, db, cost = gradient_descent(w, b, X, Y)
        # Update rule 
        w = w - learning_rate*dw
        b = b - learning_rate*db
        # Save the costs values
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            
    return w, b, dw, db, costs


def predict(w, b, X):
    # Number of samples
    m = X.shape[1]
    # Number of predictions to make
    Y_prediction = np.zeros((1,m))
    # Reshape of W - Add one dimension
    w = w.reshape(X.shape[0], 1)    
    # Compute vector "A" predicting the probabilities of being a certain number
    A = sigmoid(np.dot(w.T,X) + b)    
    # For each value predicted it is decided if is a 0 or a 1
    for i in range(A.shape[1]):        
        # Convert probabilities to predictions
        if A[0,i] > .5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    
    return Y_prediction


def train(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    # Initialize parameters with zeros
    w, b =  initialize_with_zeros(X_train.shape[0])
    
    # Gradient Descent Calculation
    w, b, dw, db, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print Train/Test Errors
    print("Train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    return costs, w, b

# Learning Cycle

# Convert DataFrames to Vectors (Matrices) -> Numpy
train_x_vect = train_x.to_numpy()
train_y_vect = train_y.to_numpy()
test_x_vect = test_x.to_numpy()
test_y_vect = test_y.to_numpy()
#train_x_vect.astype(float)

# Reshape of Training X Vector and Testing X Vector, a dimension is reduced
train_x_vect = train_x_vect.reshape(train_x_vect.shape[0], -1).T 
test_x_vect = test_x_vect.reshape(test_x_vect.shape[0], -1).T 

# Parameters for training
alpha = 0.0001
epochs = 1500

print ("learning rate is: " + str(alpha))
costs, w, b = train(train_x_vect, train_y_vect, test_x_vect, test_y_vect, num_iterations = epochs, learning_rate = alpha, print_cost = True)
print ('\n' + "-------------------------------------------------------" + '\n')
    
plt.plot(np.squeeze(costs), label= str(alpha))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

with open("weights.txt",'w') as file:
    for weight in w:
        for peso in weight:
            file.write(str(peso))
            file.write(" ")
    file.write("\n")
    file.write(str(b))    
    
