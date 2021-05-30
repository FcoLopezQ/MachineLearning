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
from PIL import Image
from matplotlib import pyplot as plt
    
# =============================================================================
# # Open Image for Testing and obtain Data    
# im = Image.open('Test_0.bmp')
# #im = Image.open('Test_1.bmp')
# #im = Image.open('Test_2.bmp')
# #im = Image.open('Test_3.bmp')
# #im = Image.open('Test_5.bmp')
# #im = Image.open('Test_6.bmp')
# #im = Image.open('Test_7.bmp')
# #im = Image.open('Test_9.bmp')
# test_data = list(im.getdata())
# width, height = im.size 
# # Turn data into numpy Array
# test_data = np.array(test_data)
# =============================================================================

# Select a number between 0 and 27999
test_data = pd.read_csv('test.csv')
test_data = np.array(test_data)
test_data = test_data[279]
if (test_data.shape[0] / 28) == 28:
    width = 28
    height = 28

# Expand dimensions and normalize data
test_data = np.expand_dims(test_data, axis=1)
test_data = test_data / 255.

#Prints Image Weight and Height
print("Image width=" + str(width))
print("Image height=" + str(height))

#Obtains Calculated Weights from Neural Network from File
npzfile = np.load('weights.npz')
W1 = npzfile['arr_0']
b1 = npzfile['arr_1']
W2 = npzfile['arr_2']
b2 = npzfile['arr_3']
    
#ReLU function
def ReLU(Z):
    return np.maximum(Z, 0)
    
#Softmax function
def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

# Perform forward propagation with previously calculated Weights
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

#Return the output that had more affinity
def get_predictions(A2):
    return np.argmax(A2,0)

# Perform Forward Propagation
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Perform test prediction for value entered
def test_prediction( W1, b1, W2, b2):
    current_image = test_data
    prediction = make_predictions(test_data, W1, b1, W2, b2)
    print("Prediction: ", prediction)
    
    #Print Image used for Prediction
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

#Perform Prediction
if width == 28 and height== 28:
    test_prediction(W1, b1, W2, b2)
else:
    print("Dimensions of Input Image are not 28x28")

