### IMPORTS
import numpy as np
from scipy import signal
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

### CONSTANTS
# Number of decimal places
np.set_printoptions(precision=4)										

# Number of Depth of filter & conv nodes
depth 			=	3									

# Input Layer Dimensions
input_rows		=	4
input_cols		=	4

# Filter Dimensions
f_rows 			=	3									
f_cols 			=	3

# Number of Outputs
outputs 		=	4							

# Convolution Layer Dimensions
conv_rows 		=	(input_rows - f_rows) + 1
conv_cols		=	(input_cols - f_cols) + 1

# Number of Total Weights
total_weights	=	depth*conv_rows*conv_cols			

# Number of Iterations
iteration		=	1000					

# Learning Rate
L_rate			=	.05

# Temporary value for incrementing convolution nodes							
temp = 0 												

### INITIALIZATION
# Initialize convolution layer
convolved_nodes = np.zeros([depth,conv_rows,conv_cols])

# Input Filter
input_filter = np.random.rand(depth, f_rows, f_cols)

### FUNCTIONS
## Activation Functions
# Softmax activation implementation
def softmax(x):
    A = x-np.max(x)
    e_x = np.exp(A)
    return e_x / e_x.sum(axis=0), A

# Sigmoid activation implementation
def sigmoid_function(x):
	return 1/(1+np.exp(-x))

# ReLU activation implementation
def relu_activation(data_array):
    return data_array * (data_array>0)

# Normalization Function
def norm(x):
	x = x - np.mean(x) / np.sqrt(np.var(x))
	return x

## CSV and Array Conversion
# Rewriting CSV file to 2d Arrays
def pd_csv_to_2darray(input_filename):
	try:
		return np.genfromtxt(input_filename, delimiter=',')
	except IOError:
		print("File not available!")

# Writing CSV to 3d Arrays
def pd_csv_to_3darray(filename_input):
	try:
		temp_array_1 = np.genfromtxt(filename_input,delimiter=',')
		temp_array_2 = np.zeros([depth,f_rows,f_cols])
		count = 1
		for x in range(0,f_rows):
			for y in range(0,f_cols):
				for z in range(0,depth):
					temp_array_2[z][y][x] = temp_array_1[count][3]
					count += 1
		return temp_array_2
	except IOError:
		print("File not available!")

## File System
def get_image(filename):
	try:
		return np.invert(np.array(Image.open(filename).convert("L")))/255
	except IOError:
		print("Error in getting image file: File not available!")

## CNN Functions
def Recognition(input_img, input_filter):
	### Forward Pass
	# Global variable initialization
	global convolved_nodes
	global temp

	# Convolution
	for i in range(0, input_filter.shape[0]):
		convolved_nodes[i] = signal.correlate(input_img, input_filter[i], mode="valid")

	# Sigmoid activation of convolution node
	convolved_nodes_sigmoid = relu_activation(convolved_nodes)

	# Flattening of sigmoid activated convolution layer
	convolved_nodes_sigmoid_flat = convolved_nodes_sigmoid.reshape(1,total_weights)

	# Fully connected layer
	output_nodes_flat = np.matmul(convolved_nodes_sigmoid_flat, convolved_nodes_to_output_nodes)

	# Softmax activation of output node
	output_nodes_flat_column = np.transpose(output_nodes_flat)#.reshape(10,1)#for column comparison
	softmax_output , softmax_1_minus_max = softmax(output_nodes_flat_column)

	return output_nodes_flat_column, softmax_output

### Testing
# Getting CSV values from training
convolved_nodes_to_output_nodes = pd_csv_to_2darray("FC.csv")
input_filter = pd_csv_to_3darray("filters.csv")
# image1 = "number0.png"
# image2 = "number1.png"
# image3 = "number4.png"
# image4 = "number7.png"

# image1 = "number7.png"
# image2 = "number1.png"
# image3 = "number6.png"
# image4 = "number9.png"

# image1 = "number3.png"
# image2 = "number5.png"
# image3 = "number4.png"
# image4 = "number1.png"

image1 = "number0.png"
image2 = "number1.png"
image3 = "number2.png"
image4 = "number3.png"
image5 = "number4.png"
image6 = "number5.png"
image7 = "number6.png"
image8 = "number7.png"
image9 = "number8.png"
image10 = "number9.png"


input_image = get_image(image1)
out, out_s = Recognition(input_image, input_filter)
print("Predicted Image:",(np.argmax(out_s)+1))
print(out_s)
# print("Confidence:%.2f" % (np.amax(out_s)*100)+"%")


input_image = get_image(image2)
out, out_s = Recognition(input_image, input_filter)
print("Predicted Image:",(np.argmax(out_s)+1))
print(out_s)


input_image = get_image(image3)
out, out_s = Recognition(input_image, input_filter)
print("Predicted Image:",(np.argmax(out_s)+1))
print(out_s)

input_image = get_image(image4)
out, out_s = Recognition(input_image, input_filter)
print("Predicted Image:",(np.argmax(out_s)+1))
print(out_s)

input_image = get_image(image5)
out, out_s = Recognition(input_image, input_filter)
print("Predicted Image:",(np.argmax(out_s)+1))
print(out_s)

input_image = get_image(image6)
out, out_s = Recognition(input_image, input_filter)
print("Predicted Image:",(np.argmax(out_s)+1))
print(out_s)

input_image = get_image(image7)
out, out_s = Recognition(input_image, input_filter)
print("Predicted Image:",(np.argmax(out_s)+1))
print(out_s)

input_image = get_image(image8)
out, out_s = Recognition(input_image, input_filter)
print("Predicted Image:",(np.argmax(out_s)+1))
print(out_s)

input_image = get_image(image9)
out, out_s = Recognition(input_image, input_filter)
print("Predicted Image:",(np.argmax(out_s)+1))
print(out_s)

input_image = get_image(image10)
out, out_s = Recognition(input_image, input_filter)
print("Predicted Image:",(np.argmax(out_s)+1))
print(out_s)