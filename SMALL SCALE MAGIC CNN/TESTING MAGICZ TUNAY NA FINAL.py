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
np.set_printoptions(precision=4)										# Number of decimal places

### VALUES
depth 			=	3									# Number of Depth of filter & conv nodes

input_rows		=	4
input_cols		=	4

f_rows 			=	3									# Number of Filter Rows
f_cols 			=	3									# Number of Filter Columns
outputs 		=	4									# Number of Outputs
conv_rows 		=	(input_rows - f_rows) + 1	# Number of Rows in Convolution Nodes
conv_cols		=	(input_cols - f_cols) + 1	# Number of Columns in Convolution Nodes
total_weights	=	depth*conv_rows*conv_cols			# Number of Total Weights
iteration		=	500								# Number of Iterations
L_rate			=	.1									# Learning Rate
temp = 0 												# Temporary value for incrementing convolution nodes

##################################################
#		   		   INITIALIZATION   		     #
##################################################


########### INPUT INITIALIZATION


# Initialize convolution layer
convolved_nodes = np.zeros([depth,conv_rows,conv_cols])

# Random initial weights for convolution layer to output layer
convolved_nodes_to_output_nodes = np.random.rand(total_weights, outputs)#pd_csv_to_2darray("Initial_weights_small_scale.csv")new_weights_random(total_weights, outputs, 0)

# Input Filter
input_filter = np.random.rand(depth, f_rows, f_cols)

##################################################
#		     FUNCTIONS    		                 #
##################################################

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

def set_target(x):
	if x==1:
		return np.array([[1,0,0,0]])
	elif x==2:
		return np.array([[0,1,0,0]])
	elif x==3:
		return np.array([[0,0,1,0]])
	elif x==4:
		return np.array([[0,0,0,1]])

def get_image(filename):
	try:
		return np.invert(np.array(Image.open(filename).convert("L")))/255
	except IOError:
		print("Error in getting image file: File not available!")

def CNN_TEST(input_img, input_filter):
	####################################################
	#                    FORWARD PASS         	       #
	####################################################

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
	output_nodes_flat_column = np.transpose(output_nodes_flat)#.reshape(10,1)#for column comparison
	softmax_output , softmax_1_minus_max = softmax(output_nodes_flat_column)

	return output_nodes_flat_column, softmax_output

def CNN(input_img, input_filter, convolved_nodes_to_output_nodes, target_value):
	####################################################
	#                    FORWARD PASS         	       #
	####################################################

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
	output_nodes_flat_column = np.transpose(output_nodes_flat)#.reshape(10,1)#for column comparison
	softmax_output , softmax_1_minus_max = softmax(output_nodes_flat_column)

	####################################################
	#                    BACKPROPAGATION       	       #
	####################################################

	##### BACKPROPAGATION CALCULATION

	# Error calculation
	softmax_output_row = np.transpose(softmax_output)#.reshape(1,10)
	error_array = error_calculation(set_target(target_value), softmax_output_row)

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

	# Reshape sigmoid flat back to 3 dimension
	convolved_nodes = convolved_nodes_sigmoid_flat.reshape(depth,conv_rows,conv_cols)

	# Updating the filters
	for x in range(0, input_filter.shape[0]):
		input_filter[x] = input_filter[x] - (signal.convolve(input_img, convolved_nodes[x], mode="valid")) * L_rate

	return input_filter, convolved_nodes_to_output_nodes, output_nodes_flat_column, softmax_output

# MAGICZ MAGICZ MAGICZ

image1 = "1.png"
image2 = "4.png"
image3 = "2.png"
image4 = "3.png"

for x in range(0, iteration):
	print(x)

	input_image = get_image(image1)
	input_filter, convolved_nodes_to_output_nodes, out, out_s = CNN(input_image, input_filter, convolved_nodes_to_output_nodes, 1)
	print("IMAGE 1 OUTPUT SOFTMAX VALUES:")
	print(out)
	print(out_s)

	input_image = get_image(image2)
	input_filter, convolved_nodes_to_output_nodes, out, out_s = CNN(input_image, input_filter, convolved_nodes_to_output_nodes, 2)
	print("IMAGE 2 OUTPUT SOFTMAX VALUES:")
	print(out)
	print(out_s)

print("FINAL FILTER VALUES:")
print(input_filter)
pd_3darray_to_csv(input_filter, "Final_filters_small_scale.csv")

# TEST THE MAGICZ

input_image = get_image(image1)
out, out_s = CNN_TEST(input_image, input_filter)
print("TEST IMAGE OUTPUT SOFTMAX VALUES:")
print(out)
print(out_s)

########## PLOTZSZS ###########
plt.subplot(221).set_title("Final Filter 1")
plt.imshow(input_filter[0])
plt.axis('off')
plt.colorbar()
plt.subplot(222).set_title("Final Filter 2")
plt.imshow(input_filter[1])
plt.axis('off')
plt.colorbar()
plt.subplot(223).set_title("Final Filter 3")
plt.imshow(input_filter[2])
plt.axis('off')
plt.colorbar()
plt.show()