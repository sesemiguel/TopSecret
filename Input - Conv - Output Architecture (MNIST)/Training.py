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
depth 			=	16									

#
mnist_depth		=	100

# Input Layer Dimensions
input_rows		=	14
input_cols		=	14

# Filter Dimensions
f_rows 			=	1									
f_cols 			=	1

# Number of Outputs
outputs 		=	10							

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
mnist_array = np.zeros([mnist_depth, input_rows, input_cols])											

### INITIALIZATION
# Initialize convolution layer
convolved_nodes = np.zeros([depth,conv_rows,conv_cols])

# Random initial weights for convolution layer to output layer
convolved_nodes_to_output_nodes = np.random.rand(total_weights, outputs)#pd_csv_to_2darray("Initial_weights_small_scale.csv")new_weights_random(total_weights, outputs, 0)

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
    return np.maximum(data_array, 0)

# Normalization Function
def norm(x):
	x = x - np.mean(x) / np.sqrt(np.var(x))
	return x

## Backpropagation Needs
# Error calculation
def error_calculation(target_array, output_array):
	return output_array - target_array

# Set Target Array
def set_target(x):
	if x==0:
		return np.array([[1,0,0,0,0,0,0,0,0,0]])
	elif x==1:
		return np.array([[0,1,0,0,0,0,0,0,0,0]])
	elif x==2:
		return np.array([[0,0,1,0,0,0,0,0,0,0]])
	elif x==3:
		return np.array([[0,0,0,1,0,0,0,0,0,0]])
	elif x==4:
		return np.array([[0,0,0,0,1,0,0,0,0,0]])
	elif x==5:
		return np.array([[0,0,0,0,0,1,0,0,0,0]])
	elif x==6:
		return np.array([[0,0,0,0,0,0,1,0,0,0]])
	elif x==7:
		return np.array([[0,0,0,0,0,0,0,1,0,0]])
	elif x==8:
		return np.array([[0,0,0,0,0,0,0,0,1,0]])
	elif x==9:
		return np.array([[0,0,0,0,0,0,0,0,0,1]])

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

def get_mnist(filename):
	try:
		##@TODO
		mnist_array_ = pd_csv_to_3darray(filename)
		return mnist_array
	except IOError:
		print("Error in getting image file: File not available!")

def pd_csv_to_2darray(input_filename):
	try:
		return np.genfromtxt(input_filename, delimiter=',')
	except IOError:
		print("File not available!")

# Writing CSV to 3d Arrays
def pd_csv_to_3darray(filename_input):
	try:
		temp_array_1 = np.genfromtxt(filename_input,delimiter=',')
		temp_array_2 = np.zeros([mnist_depth,input_rows,input_cols])
		count = 1
		for x in range(0,input_rows):
			for y in range(0,input_cols):
				for z in range(0,mnist_depth):
					temp_array_2[z][y][x] = temp_array_1[count][3]
					count += 1
		return temp_array_2
	except IOError:
		print("File not available!")



## CNN Functions
def Training(input_img, input_filter, convolved_nodes_to_output_nodes, target_value):
	### Forward Pass
	# Global variable initialization
	global convolved_nodes
	global temp

	# Convolution
	for i in range(0, input_filter.shape[0]):
		convolved_nodes[i] = signal.convolve(input_img, input_filter[i], mode="valid")

	# Sigmoid activation of convolution node
	convolved_nodes_sigmoid = sigmoid_function(convolved_nodes)

	# Flattening of sigmoid activated convolution layer
	convolved_nodes_sigmoid_flat = convolved_nodes_sigmoid.reshape(1,total_weights)

	# Fully connected layer
	output_nodes_flat = np.matmul(convolved_nodes_sigmoid_flat, convolved_nodes_to_output_nodes)

	# Softmax activation of output node
	output_nodes_flat_column = np.transpose(output_nodes_flat)
	softmax_output , softmax_1_minus_max = softmax(output_nodes_flat_column)

	### Backpropagation
	# Error calculation on Output Layer
	softmax_output_row = np.transpose(softmax_output)
	error_array = error_calculation(set_target(target_value), softmax_output_row)

	# Error * Sigmoid backpropagation formula
	A = error_array * softmax_output_row * (1-softmax_output_row)

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
		input_filter[x] = input_filter[x] + (signal.convolve(input_img, convolved_nodes[x], mode="valid")) * L_rate

	# input_filter = norm(input_filter)

	return input_filter, convolved_nodes_to_output_nodes, output_nodes_flat_column, softmax_output

### Training
# Images used
image1 = "1.png"
image2 = "2.png"
image3 = "3.png"
image4 = "4.png"
image5 = "5.png" 
image6 = "6.png" 
image7 = "7.png" 
image8 = "8.png" 
image_test = "test.png"

mnist_array = get_mnist("3Dmaxpool.csv")
target_array = pd_csv_to_2darray("target_array.csv")
print("Shape")
print(mnist_array)

for jakol in range(0,100):
	print("Epoch: " , jakol+1)
	for x in range(0, mnist_depth):
		print(x)
		input_image = mnist_array[x]
		input_filter, convolved_nodes_to_output_nodes, out, out_s = Training(input_image, input_filter, convolved_nodes_to_output_nodes, target_array[x])

print("FC Weights")
print(convolved_nodes_to_output_nodes)

print("Filters")
print(input_filter)

# CSV conversion of weights and filters
pd_2darray_to_csv(convolved_nodes_to_output_nodes, "FC.csv")
pd_3darray_to_csv(input_filter, "filters.csv")