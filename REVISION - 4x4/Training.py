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
iteration		=	894				

# Learning Rate
L_rate			=	.125

# Temporary value for incrementing convolution nodes							
temp = 0 												

### INITIALIZATION
# Initialize convolution layer
convolved_nodes = np.zeros([depth,conv_rows,conv_cols])
# Rewriting CSV file to 2d Arrays
def pd_csv_to_2darray(input_filename):
	try:
		return np.genfromtxt(input_filename, delimiter=',')
	except IOError:
		print("File not available!")
# Random initial weights for convolution layer to output layer
convolved_nodes_to_output_nodes = np.random.rand(total_weights, outputs)#pd_csv_to_2darray("initial_weights.csv")np.random.rand(total_weights, outputs)

# Input Filter
input_filter = np.random.rand(depth, f_rows, f_cols)
# input_filter = np.array([[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]]])
# input_filter = input_filter*.125#input_filter/np.amax(input_filter)

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

def relu_deriv(data_array):
	return np.heaviside(data_array, 0)

# Normalization Function
def norm(x):
	x = x - np.mean(x) / np.sqrt(np.var(x))
	return x

## Backpropagation Needs
# Error calculation
def error_calculation(target_array, output_array):
	return output_array - target_array

# # Set Target Array
# def set_target(x):
# 	if x==1:
# 		return np.array([[1,0,0,0,0,0,0,0]])
# 	elif x==2:
# 		return np.array([[0,1,0,0,0,0,0,0]])
# 	elif x==3:
# 		return np.array([[0,0,1,0,0,0,0,0]])
# 	elif x==4:
# 		return np.array([[0,0,0,1,0,0,0,0]])
# 	elif x==5:
# 		return np.array([[0,0,0,0,1,0,0,0]])
# 	elif x==6:
# 		return np.array([[0,0,0,0,0,1,0,0]])
# 	elif x==7:
# 		return np.array([[0,0,0,0,0,0,1,0]])
# 	elif x==8:
# 		return np.array([[0,0,0,0,0,0,0,1]])

def set_target(x):
	if x==1:
		return np.array([[1,0,0,0]])
	elif x==2:
		return np.array([[0,1,0,0]])
	elif x==3:
		return np.array([[0,0,1,0]])
	elif x==4:
		return np.array([[0,0,0,1]])

## CSV and Array Conversion
# Writing 2d arrays to a CSV file
def pd_2darray_to_csv(input_array, filename_output):
	np.savetxt(filename_output, input_array, delimiter=",")
	print("Generated CSV for FC weights")
	
# Writing 2d arrays to a CSV file
def pd_3darray_to_csv(input_array, filename_output):
	stacked = pd.Panel(input_array.swapaxes(1,2)).to_frame().stack().reset_index()
	stacked.columns = ['x', 'y', 'z', 'value']
	stacked.to_csv(filename_output, index=False)
	print("Generated CSV for filters weights")

## File System
def get_image(filename):
	try:
		return np.invert(np.array(Image.open(filename).convert("L")))/255
	except IOError:
		print("Error in getting image file: File not available!")

## CNN Functions
def Training(input_img, input_filter, convolved_nodes_to_output_nodes, target_value):
	### Forward Pass
	# Global variable initialization
	global convolved_nodes
	global temp

	# Convolution
	for i in range(0, input_filter.shape[0]):
		convolved_nodes[i] = signal.correlate(input_img, input_filter[i], mode="valid")

	# Sigmoid activation of convolution node
	convolved_nodes_sigmoid = relu_activation(convolved_nodes)#sigmoid_function(convolved_nodes)

	# Flattening of sigmoid activated convolution layer
	convolved_nodes_sigmoid_flat = convolved_nodes_sigmoid.reshape(1,total_weights)

	# Fully connected layer
	output_nodes_flat = np.matmul(convolved_nodes_sigmoid_flat, convolved_nodes_to_output_nodes)

	# Softmax activation of output node
	output_nodes_flat_column = np.transpose(output_nodes_flat)
	# print("FC OUT")
	# print(output_nodes_flat_column)
	softmax_output , softmax_1_minus_max = softmax(output_nodes_flat_column)

	### Backpropagation
	# Error calculation on Output Layer
	softmax_output_row = np.transpose(softmax_output)
	
	error_array = error_calculation(set_target(target_value), softmax_output_row)

	# Error * Sigmoid backpropagation formula
	A = error_array * softmax_output_row * relu_deriv(softmax_output_row)#(1-softmax_output_row)

	# Transpose result A
	A = np.transpose(A)

	# SOP of A and Nodes in flat form
	B = np.matmul(A,convolved_nodes_sigmoid_flat)

	# Transpose result B
	B = np.transpose(B)

	# Updating of FC weights --- Cost Function
	for i in range(0,total_weights):
		for j in range(0,outputs):
			convolved_nodes_to_output_nodes[i][j] = convolved_nodes_to_output_nodes[i][j] - L_rate * B[i][j]
	# print("FC Weights")
	# print(convolved_nodes_to_output_nodes)
	# Updating of Convolution layer nodes
	for i in range(0,total_weights):
		for j in range(0,outputs):
			convolved_nodes_sigmoid_flat[0][i] = (softmax_output_row[0][j] * convolved_nodes_to_output_nodes[i][j]) + temp 
			temp = convolved_nodes_sigmoid_flat[0][i]
		temp = 0

	# Reshape sigmoid flat back to 3 dimension
	convolved_nodes = convolved_nodes_sigmoid_flat.reshape(depth,conv_rows,conv_cols)

	# Updating the filters
	for x in range(0, input_filter.shape[0]):
		input_filter[x] = input_filter[x] + signal.correlate(input_img, convolved_nodes[x], mode="valid") * L_rate
	# print("Filters")
	# print(input_filter)
	# input_filter = norm(input_filter)

	input_filter = input_filter*.5#input_filter/np.amax(input_filter)
	# convolved_nodes_to_output_nodes = convolved_nodes_to_output_nodes*.5

	return input_filter, convolved_nodes_to_output_nodes, output_nodes_flat_column, softmax_output

### Training
# Images used
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

image1 = "number1.png"
image2 = "number8.png"
image3 = "number2.png"
image4 = "number7.png"


for x in range(0, iteration):
	print(x)

	input_image = get_image(image1)
	input_filter, convolved_nodes_to_output_nodes, out, out_s = Training(input_image, input_filter, convolved_nodes_to_output_nodes, 1)
	print("SOFTMAX 1")
	print(out_s)

	input_image = get_image(image2)
	input_filter, convolved_nodes_to_output_nodes, out, out_s = Training(input_image, input_filter, convolved_nodes_to_output_nodes, 2)
	print("SOFTMAX 2")
	print(out_s)

	input_image = get_image(image3)
	input_filter, convolved_nodes_to_output_nodes, out, out_s = Training(input_image, input_filter, convolved_nodes_to_output_nodes, 3)
	print("SOFTMAX 3")
	print(out_s)

	input_image = get_image(image4)
	input_filter, convolved_nodes_to_output_nodes, out, out_s = Training(input_image, input_filter, convolved_nodes_to_output_nodes, 4)
	print("SOFTMAX 4")
	print(out_s)

	# input_image = get_image(image5)
	# input_filter, convolved_nodes_to_output_nodes, out, out_s = Training(input_image, input_filter, convolved_nodes_to_output_nodes, 5)

	# input_image = get_image(image6)
	# input_filter, convolved_nodes_to_output_nodes, out, out_s = Training(input_image, input_filter, convolved_nodes_to_output_nodes, 6)

	# input_image = get_image(image7)
	# input_filter, convolved_nodes_to_output_nodes, out, out_s = Training(input_image, input_filter, convolved_nodes_to_output_nodes, 7)

	# input_image = get_image(image8)
	# input_filter, convolved_nodes_to_output_nodes, out, out_s = Training(input_image, input_filter, convolved_nodes_to_output_nodes, 8)

# print("FC Weights")
# print(convolved_nodes_to_output_nodes)

# print("Filters")
# print(input_filter)

# CSV conversion of weights and filters
pd_2darray_to_csv(convolved_nodes_to_output_nodes, "FC.csv")
pd_3darray_to_csv(input_filter, "filters.csv")