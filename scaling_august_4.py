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
f_rows 			=	2									# Number of Filter Rows
f_cols 			=	2									# Number of Filter Columns
outputs 		=	4									# Number of Outputs
conv_rows 		=	(input_img.shape[0] - f_rows) + 1	# Number of Rows in Convolution Nodes
conv_cols		=	(input_img.shape[1] - f_cols) + 1	# Number of Columns in Convolution Nodes
total_weights	=	depth*conv_rows*conv_cols			# Number of Total Weights
iteration		=	10									# Number of Iterations
L_rate			=	.5									# Learning Rate
temp = 0 												# Temporary value for incrementing convolution nodes


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
    return e_x / e_x.sum(axis=0), A

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

########### MAX AND MIN INITIALIZATION

# Initialize convolution layer
convolved_nodes = np.zeros([depth,conv_rows,conv_cols])

# Random initial weights for convolution layer to output layer
convolved_nodes_to_output_nodes = new_weights_random(total_weights, outputs, 0)#pd_csv_to_2darray("Initial_weights_small_scale.csv")

# Target Array
target_array = np.array([1,0,0,0])

# Initialize maximum and minimum filter weights
max_filter_weights = np.zeros((depth,f_rows,f_cols))
min_filter_weights = np.zeros((depth,f_rows,f_cols))

# Initialize maximum and minimum FC weights
max_FC_weights = np.zeros((total_weights,conv_rows*conv_cols))
min_FC_weights = np.zeros((total_weights,conv_rows*conv_cols))

# Initialize maximum and minimum conv nodes
max_conv_nodes = np.zeros((depth,conv_rows,conv_cols))
min_conv_nodes = np.zeros((depth,conv_rows,conv_cols))

# Initialize maximum and minimum out nodes
max_out_nodes = np.zeros((1,outputs))
min_out_nodes = np.zeros((1,outputs))

# Initialize maximum and minimum softmax_1_minus_max nodes
max_softmax_1_minus_max = np.zeros((outputs,1))
min_softmax_1_minus_max = np.zeros((outputs,1))

# Initialize maximum and minimum conv_nodes_sigmoid
max_conv_nodes_sigmoid = np.zeros((depth,conv_rows,conv_cols))
min_conv_nodes_sigmoid = np.zeros((depth,conv_rows,conv_cols))

########### INPUT INITIALIZATION

# Input Filter
input_filter = np.zeros((depth,f_rows,f_rows))
input_filter[0] = new_weights_random(f_rows, f_cols, 0)
input_filter[1] = new_weights_random(f_rows, f_cols, 0)
input_filter[2] = new_weights_random(f_rows, f_cols, 0)#np.array([[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]]])#np.random.rand(3, 3, 3)#np.array([[[0,1,0],[0,1,0],[0,1,0]],[[0,0,0],[1,1,1],[0,0,0]],[[0,0,1],[0,1,0],[1,0,0]]])#new_weights_random(3,3,0)#np.array([[0,0,0,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,0]])

for x in range(1, iteration):
	# Print iteration number
	print("ITERATION: ", x)

	####################################################
	#                    FORWARD PASS         	       #
	####################################################

	# Convolution
	for i in range(0, input_filter.shape[0]):
		convolved_nodes[i] = signal.convolve(input_img, input_filter[i], mode="valid")

	# Sigmoid activation of convolution node
	convolved_nodes_sigmoid = sigmoid_function(convolved_nodes)

	# Get max/min conv sigmoid
	get_max_conv_sigmoid(max_conv_nodes_sigmoid, convolved_nodes_sigmoid)
	get_min_conv_sigmoid(min_conv_nodes_sigmoid, convolved_nodes_sigmoid)

	# Flattening of sigmoid activated convolution layer
	convolved_nodes_sigmoid_flat = convolved_nodes_sigmoid.reshape(1,total_weights)

	# Fully connected layer
	output_nodes_flat = np.matmul(convolved_nodes_sigmoid_flat, convolved_nodes_to_output_nodes)

	# Softmax activation of output node
	output_nodes_flat_column = np.transpose(output_nodes_flat)#.reshape(10,1)#for column comparison
	softmax_output , softmax_1_minus_max = softmax(output_nodes_flat_column)

	####################################################
	#                    BACKPROPAGATION       	       #
	####################################################

	##### BACKPROPAGATION CALCULATION

	# Error calculation
	softmax_output_row = np.transpose(softmax_output)#.reshape(1,10)
	error_array = error_calculation(target_array,softmax_output_row)

	# Error * Sigmoid backpropagation formula
	A = error_array * softmax_output_row * (1-softmax_output_row)

	# Transpose result A
	A = np.transpose(A)

	# SOP of A and Nodes in flat form
	B = np.matmul(A,convolved_nodes_sigmoid_flat)

	# Transpose result B
	B = np.transpose(B)

	##### UPDATING

	# Updating of FC weights
	for i in range(0,total_weights):
		for j in range(0,outputs):
			convolved_nodes_to_output_nodes[i][j] = convolved_nodes_to_output_nodes[i][j] - L_rate * B[i][j]

	# Updating of Convolution layer nodes
	for i in range(0,total_weights):
		for j in range(0,outputs):
			convolved_nodes_sigmoid_flat[0][i] = (softmax_output_row[0][j] * convolved_nodes_to_output_nodes[i][j]) + temp 
			temp = convolved_nodes_sigmoid_flat[0][i]
		temp = 0

	##### UPDATING MAX AND MIN VALUES

	# Get max/min conv nodes
	get_max_conv_nodes(max_conv_nodes, convolved_nodes)
	get_min_conv_nodes(min_conv_nodes, convolved_nodes)

	# Reshape sigmoid flat back to 3 dimension
	convolved_nodes = convolved_nodes_sigmoid_flat.reshape(depth,conv_rows,conv_cols)

	# Get max/min filter weights
	get_max_filter_weights(max_filter_weights, input_filter)
	get_min_filter_weights(min_filter_weights, input_filter)

	# Get max/min FC weights
	get_max_FC_weights(max_FC_weights, convolved_nodes_to_output_nodes)
	get_min_FC_weights(min_FC_weights, convolved_nodes_to_output_nodes)

	# Get max/min output nodes
	get_max_output_nodes(max_out_nodes, output_nodes_flat)
	get_min_output_nodes(min_out_nodes, output_nodes_flat)

	# Get max/min softmax_1_minus_max
	get_max_softmax_1_minus_max(max_softmax_1_minus_max, softmax_1_minus_max)
	get_min_softmax_1_minus_max(min_softmax_1_minus_max, softmax_1_minus_max)

	# Updating the filters
	for x in range(0, input_filter.shape[0]):
		input_filter[x] = signal.convolve(input_img, convolved_nodes[x], mode="valid")
	print("UPDATED FILTERS")

print(input_filter)
pd_3darray_to_csv(input_filter, "Final_filters_small_scale.csv")
