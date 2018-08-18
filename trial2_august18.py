##################################################
#		      IMPORTS    		                 #
##################################################
import numpy as np
from scipy import signal
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

##################################################
#		   			CONSTANTS 		             #
##################################################
filename = "yappie.png"													# Input file
np.set_printoptions(precision=4)										# Number of decimal places
input_img = np.invert(np.array(Image.open(filename).convert("L")))/255	# Input for convolution array

### VALUES
depth 			=	3									# Number of Depth of filter & conv nodes
f_rows 			=	3									# Number of Filter Rows
f_cols 			=	3									# Number of Filter Columns
conv_rows 		=	(input_img.shape[0] - f_rows) + 1	# Number of Rows in Convolution Nodes
conv_cols		=	(input_img.shape[1] - f_cols) + 1	# Number of Columns in Convolution Nodes
total_weights	=	depth*conv_rows*conv_cols			# Number of Total Weights
iteration		=	1									# Number of Iterations
L_rate			=	.5									# Learning Rate
temp = 0 												# Temporary value for incrementing convolution nodes

h_rows 			= 	1
h_cols			=	2
o_rows			=	1
o_cols			=	2

# Target Array
target_array = np.array([1,0])

##################################################
#		     FUNCTIONS    		                 #
##################################################
# Generating new random values for the weights
def new_weights_random(x_size, y_size, flag):
	if flag == 0:
		return np.random.rand(x_size, y_size)#np.zeros((x_size,y_size))np.zeros((x_size,y_size))
	elif flag == 1:
		return fc_weight

# Softmax activation implementation
def softmax(x):
    A = x-np.max(x)
    e_x = np.exp(A)
    return e_x / e_x.sum(axis=1)

# Sigmoid activation implementation
def sigmoid_function(x):
	return 1/(1+np.exp(-x))

# Error calculation
def error_calculation(target_array, output_array):
	return output_array - target_array

# ReLU activation implementation
def relu_activation(data_array):
    return np.maximum(data_array, 0)

# Getting the max filter weights for 3D
def get_max_filter_weights(max_filter_weights, input_filter):
	for i in range(0,depth):
		for j in range(0,f_rows):
			for k in range(0,f_cols):
				if(max_filter_weights[i][j][k] == 0):
						max_filter_weights[i][j][k] = input_filter[i][j][k]
				else:
						if(input_filter[i][j][k] > max_filter_weights[i][j][k]):
							max_filter_weights[i][j][k] = input_filter[i][j][k]

# Getting the min filter weights for 3D 
def get_min_filter_weights(min_filter_weights, input_filter):
	for i in range(0,depth):
		for j in range(0,f_rows):
			for k in range(0,f_cols):
				if(min_filter_weights[i][j][k] == 0):
						min_filter_weights[i][j][k] = input_filter[i][j][k]
				else:
						if(input_filter[i][j][k] < min_filter_weights[i][j][k]):
							min_filter_weights[i][j][k] = input_filter[i][j][k]

# Getting the max FC weights for 2D
def get_max_FC_weights(max_FC_weights, input_filter):
	for i in range(0,total_weights):
		for j in range(0,outputs):
				if(max_FC_weights[i][j] == 0):
						max_FC_weights[i][j] = input_filter[i][j]
				else:
						if(input_filter[i][j] > max_FC_weights[i][j]):
							max_FC_weights[i][j] = input_filter[i][j]

# Getting the min FC weights for 2D
def get_min_FC_weights(min_FC_weights, input_filter):
	for i in range(0,total_weights):
		for j in range(0,outputs):
				if(min_FC_weights[i][j] == 0):
						min_FC_weights[i][j] = input_filter[i][j]
				else:
						if(input_filter[i][j] < min_FC_weights[i][j]):
							min_FC_weights[i][j] = input_filter[i][j]

# Getting max output nodes
def get_max_output_nodes(max_output_nodes, input_filter):
	for j in range(0,outputs):
		if(max_output_nodes[0][j] == 0):
				max_output_nodes[0][j] = input_filter[0][j]
		else:
				if(input_filter[0][j] > max_output_nodes[0][j]):
					max_output_nodes[0][j] = input_filter[0][j]

# Getting min output nodes
def get_min_output_nodes(min_output_nodes, input_filter):
	for j in range(0,outputs):
		if(min_output_nodes[0][j] == 0):
				min_output_nodes[0][j] = input_filter[0][j]
		else:
				if(input_filter[0][j] < min_output_nodes[0][j]):
					min_output_nodes[0][j] = input_filter[0][j]

# Getting max conv nodes
def get_max_conv_nodes(max_conv_nodes, input_filter):
	for i in range(0,depth):
		for j in range(0,conv_rows):
			for k in range(0,conv_cols):
				if(max_conv_nodes[i][j][k] == 0):
						max_conv_nodes[i][j][k] = input_filter[i][j][k]
				else:
						if(input_filter[i][j][k] > max_conv_nodes[i][j][k]):
							max_conv_nodes[i][j][k] = input_filter[i][j][k]

# Getting min conv nodes
def get_min_conv_nodes(min_conv_nodes, input_filter):
	for i in range(0,depth):
		for j in range(0,conv_rows):
			for k in range(0,conv_cols):
				if(min_conv_nodes[i][j][k] == 0):
						min_conv_nodes[i][j][k] = input_filter[i][j][k]
				else:
						if(input_filter[i][j][k] < min_conv_nodes[i][j][k]):
							min_conv_nodes[i][j][k] = input_filter[i][j][k]

def get_max_softmax_1_minus_max(max_softmax_1_minus_max, input_filter):
	for j in range(0,outputs):
		if(max_softmax_1_minus_max[j][0] == 0):
				max_softmax_1_minus_max[j][0] = input_filter[j][0]
		else:
				if(input_filter[j][0] > max_softmax_1_minus_max[j][0]):
					max_softmax_1_minus_max[j][0] = input_filter[j][0]

def get_min_softmax_1_minus_max(min_softmax_1_minus_max, input_filter):
	for j in range(0,outputs):
		if(min_softmax_1_minus_max[j][0] == 0):
				min_softmax_1_minus_max[j][0] = input_filter[j][0]
		else:
				if(input_filter[j][0] < min_softmax_1_minus_max[j][0]):
					min_softmax_1_minus_max[j][0] = input_filter[j][0]

# Getting max conv nodes
def get_max_conv_sigmoid(max_conv_sigmoid, input_filter):
	for i in range(0,depth):
		for j in range(0,conv_rows):
			for k in range(0,conv_cols):
				if(max_conv_sigmoid[i][j][k] == 0):
						max_conv_sigmoid[i][j][k] = input_filter[i][j][k]
				else:
						if(input_filter[i][j][k] > max_conv_sigmoid[i][j][k]):
							max_conv_sigmoid[i][j][k] = input_filter[i][j][k]

# Getting min conv nodes
def get_min_conv_sigmoid(min_conv_sigmoid, input_filter):
	for i in range(0,depth):
		for j in range(0,conv_rows):
			for k in range(0,conv_cols):
				if(min_conv_sigmoid[i][j][k] == 0):
						min_conv_sigmoid[i][j][k] = input_filter[i][j][k]
				else:
						if(input_filter[i][j][k] < min_conv_sigmoid[i][j][k]):
							min_conv_sigmoid[i][j][k] = input_filter[i][j][k]

# Writing 2d arrays to a CSV file
def pd_2darray_to_csv(input_array, filename_output):
	np.savetxt(filename_output, input_array, delimiter=",")

# Writing 2d arrays to a CSV file
def pd_3darray_to_csv(input_array, filename_output):
	stacked = pd.Panel(input_array.swapaxes(1,2)).to_frame().stack().reset_index()
	stacked.columns = ['x', 'y', 'z', 'value']
	stacked.to_csv(filename_output, index=False)

# Rewriting CSV file to 2d arrays
def pd_csv_to_2darray(input_filename):
	try:
		return np.genfromtxt(input_filename, delimiter=',')
	except IOError:
		print("File not available!")

##################################################
#		   		   INITIALIZATION   		     #
##################################################

# Initialize convolution layer
convolved_nodes = np.zeros([depth,conv_rows,conv_cols])

# Random initial weights for convolution layer to output layer
convolved_nodes_to_hidden_nodes = new_weights_random(12,2,0)
hidden_nodes_to_output_nodes = new_weights_random(2,2,0)

# Input Filter
input_filter = np.zeros((depth,f_rows,f_rows))
input_filter[0] = new_weights_random(f_rows, f_cols, 0)
input_filter[1] = new_weights_random(f_rows, f_cols, 0)
input_filter[2] = new_weights_random(f_rows, f_cols, 0)

for x in range(0,iteration):

	# Print iteration number
	print("ITERATION: ", x)

####################################################
#                   FORWARD PASS         	   	   #
####################################################

	# Convolution
	for x in range(0, input_filter.shape[0]):
		convolved_nodes[x] = signal.correlate(input_img, input_filter[x], mode="valid")

	# Sigmoid activation of convolution node
	convolved_nodes_sigmoid = sigmoid_function(convolved_nodes)

	# Flattening of sigmoid activated convolution layer
	convolved_nodes_sigmoid_flat = convolved_nodes_sigmoid.reshape(1,total_weights)

	# Hidden Layer
	hidden_layer = np.matmul(convolved_nodes_sigmoid_flat,convolved_nodes_to_hidden_nodes)

	# Sigmoid activation of hidden layer node
	hidden_layer_sigmoid = sigmoid_function(hidden_layer)

	# Output Layer
	output_layer = np.matmul(hidden_layer_sigmoid,hidden_nodes_to_output_nodes)

	# Softmax Activiation of the Output Layer
	softmax_output = softmax(output_layer)

	####################################################
	#                    BACKPROPAGATION       	       #
	####################################################

	# Error Calculation
	error_array = error_calculation(target_array,softmax_output)

	# Error * Sigmoid Backpropagation Formula
	A = error_array * softmax_output * (1-softmax_output)

	B = np.matmul(A, hidden_nodes_to_output_nodes)

	C = 0

	for i in range(0,2):
		for j in range(0,2):
			C = C + (A*hidden_nodes_to_output_nodes[i][j])
		
		# D = np.matmul(C,np.transpose(hidden_layer_sigmoid))
	
	

	# Updating of hl to ol weights
	for i in range(0,2):
		for j in range(0,2):
			hidden_nodes_to_output_nodes[i][j] = hidden_nodes_to_output_nodes[i][j] - L_rate * B[0][j]

	# Updating of Hidden Layer Nodes
	for i in range(0,2):
		for j in range(0,2):
			hidden_layer_sigmoid[0][i] = (softmax_output[0][j] * hidden_nodes_to_output_nodes[i][j]) + temp 
			temp = hidden_layer_sigmoid[0][i]
		temp = 0

