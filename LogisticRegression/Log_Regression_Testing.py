# ===========================================================================
# Logistic Regression By Hand - Use of Model for Predictions
# Author: Francisco Iván López Quihuis
# 
# DataSet Reference: 
# Kaggle (n.d.) Digit Recognizer, MNIST Data. Kaggle:
# https://www.kaggle.com/c/digit-recognizer
# ===========================================================================

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def sigmoid(z):
    s = 1./(1+np.exp(-z))
    return s

# Open File with Weights and Bias
file = open("weights.txt",'r')
weights = []

# Read lines and extract Weights and Bias
for line in file.readlines():
    weightsTemp = line.split(" ")
    for indWeight in weightsTemp:
        weights.append(indWeight)

# Empty character is deleted
del weights[-2]
           
# Open Image for Testing and obtain Data    
im = Image.open('Test_0_1.bmp') # <-------
# Get information of the image
test_data = list(im.getdata())
width, height = im.size 

# Turn data into numpy Array
test_data = np.array(test_data)
# Expand dimensions
test_data = np.expand_dims(test_data, axis=1)
test_data = test_data / 255.

#Prints Image Weight and Height
print("Image width=" + str(width))
print("Image height=" + str(height))

#Perform Prediction
if width == 28 and height== 28:
    # Assign Weights and Bias to Variables
    b = float(weights[len(weights)-1])
    weights = weights[0:len(weights)-1]
    W = [float(i) for i in weights]
    # Variable to store Prediction Variables
    X = np.zeros(len(W))
    # Convert input values to Float
    X = [float(i) for i in test_data]
    # Print input image
    current_image = np.array(X)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    # Calculate based in weights and values of input for prediction
    A = sigmoid(np.dot(W,X) + b) 
    print("Category:")
    if A > 0.5:
        print("Image corresponds to a Number 1")
    else: 
        print("Image corresponds to a Number 0")
else:
    print("Dimensions of Input Image are not 28x28")




