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
depth 			=	8

# Number of Test Data
mnist_depth		=	10000	
mnist_depth_input = 10000							

# Input Layer Dimensions
input_rows		=	9
input_cols		=	9

# Filter Dimensions
f_rows 			=	3									
f_cols 			=	3

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
mnist_array = np.zeros([mnist_depth,input_rows,input_cols])									

### INITIALIZATION
# Initialize convolution layer
convolved_nodes = np.zeros([depth,conv_rows,conv_cols])

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
		print(temp_array_1)
		temp_array_2 = np.zeros([mnist_depth_input,input_rows,input_cols])
		count = 1
		for x in range(0,input_rows):
			for y in range(0,input_cols):
				for z in range(0,mnist_depth_input):
					temp_array_2[z][y][x] = temp_array_1[count][3]
					count += 1
		return temp_array_2
	except IOError:
		print("File not available!")

def pd_csv_to_3darray_filter(filename_input):
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

def get_mnist(filename):
	try:
		##@TODO
		mnist_array = pd_csv_to_3darray(filename)
		#mnist_array = np.where(mnist_array > 65, 1, 0)
		return mnist_array
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
		convolved_nodes[i] = signal.convolve(input_img, input_filter[i], mode="valid")
	print("MAX MIN")
	print(np.argmax(convolved_nodes))
	print(np.argmin(convolved_nodes))

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
input_filter = pd_csv_to_3darray_filter("filters.csv")
target_test = pd_csv_to_2darray("target_array_1.csv")





input_image = get_mnist("mnist_full_data_3d_9x9_train_1.csv")

print(input_image[0])
print(input_image[1])
print(input_image[2])
print(input_image[3])
print(input_image[4])
print(input_image[5])
print(input_image[6])
print(input_image[7])
print(input_image[8])
print(input_image[9])
print(input_filter)

count = 0

for x in range (0, mnist_depth):
	out, out_s = Recognition(input_image[x], input_filter)

	print("Predicted Image:",(np.argmax(out_s)))
	print("Actual Number:",target_test[x])
	# print("Confidence:%.2f" % (np.amax(out_s)*100)+"%")
	print(out_s)
	if np.argmax(out_s) == target_test[x]:
		count = count + 1

print(count)
# ### Plotting
# plt.subplot(441).set_title("Final Filter 1")
# plt.imshow(input_filter[0])
# plt.axis('off')
# plt.subplot(442).set_title("Final Filter 2")
# plt.imshow(input_filter[1])
# plt.axis('off')
# plt.subplot(443).set_title("Final Filter 3")
# plt.imshow(input_filter[2])
# plt.axis('off')
# plt.subplot(444).set_title("Final Filter 4")
# plt.imshow(input_filter[3])
# plt.axis('off')
# plt.subplot(445).set_title("Final Filter 5")
# plt.imshow(input_filter[4])
# plt.axis('off')
# plt.subplot(446).set_title("Final Filter 6")
# plt.imshow(input_filter[5])
# plt.axis('off')
# plt.subplot(447).set_title("Final Filter 7")
# plt.imshow(input_filter[6])
# plt.axis('off')
# plt.subplot(448).set_title("Final Filter 8")
# plt.imshow(input_filter[7])
# plt.axis('off')
# plt.subplot(449).set_title("Final Filter 9")
# plt.imshow(input_filter[8])
# plt.axis('off')
# plt.subplot(4,4,10).set_title("Final Filter 10")
# plt.imshow(input_filter[9])
# plt.axis('off')
# plt.subplot(4,4,11).set_title("Final Filter 11")
# plt.imshow(input_filter[10])
# plt.axis('off')
# plt.subplot(4,4,12).set_title("Final Filter 12")
# plt.imshow(input_filter[11])
# plt.axis('off')
# plt.subplot(4,4,13).set_title("Final Filter 13")
# plt.imshow(input_filter[12])
# plt.axis('off')
# plt.subplot(4,4,14).set_title("Final Filter 14")
# plt.imshow(input_filter[13])
# plt.axis('off')
# plt.subplot(4,4,15).set_title("Final Filter 15")
# plt.imshow(input_filter[14])
# plt.axis('off')
# plt.subplot(4,4,16).set_title("Final Filter 16")
# plt.imshow(input_filter[15])
# plt.axis('off')
# plt.show()