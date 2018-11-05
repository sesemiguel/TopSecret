### IMPORTS
import numpy as np
from scipy import signal
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


max_conv = 0
min_conv = 99999
max_conv_t = 0
min_conv_t = 0


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
		# mnist_array = mnist_array/255
		mnist_array = np.where(mnist_array > 65, 1, 0)
		return mnist_array
	except IOError:
		print("Error in getting image file: File not available!")

## CNN Functions
def Recognition(input_img, input_filter, max_conv, min_conv):
	### Forward Pass
	# Global variable initialization
	global convolved_nodes
	global temp
	global max_conv_t
	global min_conv_t

	# Convolution
	for i in range(0, input_filter.shape[0]):
		convolved_nodes[i] = signal.correlate(input_img, input_filter[i], mode="valid")
	# print("MAX MIN")
	# print(np.amax(convolved_nodes))
	# print(np.amin(convolved_nodes))
	# max_conv_t = np.amax(convolved_nodes)
	# if max_conv_t > max_conv:
	# 	max_conv = max_conv_t
	# else:
	# 	max_conv = max_conv

	# min_conv_t = np.amin(convolved_nodes)
	# if min_conv_t < min_conv:
	# 	min_conv = min_conv_t
	# else:
	# 	min_conv = min_conv
	# print("CONV NODES:")
	# print(convolved_nodes)


	# Sigmoid activation of convolution node
	convolved_nodes_sigmoid = relu_activation(convolved_nodes)

	# print("CONV RELU:")
	# print(convolved_nodes_sigmoid)
	# print(convolved_nodes_sigmoid.shape)

	# Flattening of sigmoid activated convolution layer
	convolved_nodes_sigmoid_flat = convolved_nodes_sigmoid.reshape(1,total_weights)
	# print("CONV FLAT:")
	# print(convolved_nodes_sigmoid_flat)

	# Fully connected layer
	output_nodes_flat = np.matmul(convolved_nodes_sigmoid_flat, convolved_nodes_to_output_nodes)
	# print("MAX MIN FC")
	# print(np.amax(output_nodes_flat))
	# print(np.amin(output_nodes_flat))
	# max_conv_t = np.amax(output_nodes_flat)
	# if max_conv_t > max_conv:
	# 	max_conv = max_conv_t
	# else:
	# 	max_conv = max_conv

	# min_conv_t = np.amin(output_nodes_flat)
	# if min_conv_t < min_conv:
	# 	min_conv = min_conv_t
	# else:
	# 	min_conv = min_conv
	print("OUTPUT NODES:")
	print(output_nodes_flat)

	# Softmax activation of output node
	output_nodes_flat_column = np.transpose(output_nodes_flat)#.reshape(10,1)#for column comparison
	softmax_output , softmax_1_minus_max = softmax(output_nodes_flat_column)
	# print("MAX MIN")
	# print(np.amax(softmax_output))
	# print(np.amin(softmax_output))
	max_conv_t = np.amax(softmax_output)
	if max_conv_t > max_conv:
		max_conv = max_conv_t
	else:
		max_conv = max_conv

	min_conv_t = np.amin(softmax_output)
	if min_conv_t < min_conv:
		min_conv = min_conv_t
	else:
		min_conv = min_conv

	return output_nodes_flat_column, softmax_output, min_conv, max_conv

### Testing
# Getting CSV values from training
convolved_nodes_to_output_nodes = pd_csv_to_2darray("FC_85.02.csv")
input_filter = pd_csv_to_3darray_filter("filters_85.02.csv")
target_test = pd_csv_to_2darray("target_array_test.csv")
print(convolved_nodes_to_output_nodes)





input_image = get_mnist("mnist_full_data_3d_9x9_test.csv")

# print(input_image[0])
# print(input_image[1])
# print(input_image[2])
# print(input_image[3])
# print(input_image[4])
# print(input_image[5])
# print(input_image[6])
# print(input_image[7])
# print(input_image[8])
# print(input_image[9])
# print(input_filter)

count = 0
count0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
count9 = 0
tcount0 = 0
tcount1 = 0
tcount2 = 0
tcount3 = 0
tcount4 = 0
tcount5 = 0
tcount6 = 0
tcount7 = 0
tcount8 = 0
tcount9 = 0

for x in range (0, mnist_depth):
	print("MNIST Depth",x+1)
	out, out_s, min_conv, max_conv = Recognition(input_image[x], input_filter, max_conv, min_conv)
	print("Predicted Image:",(np.argmax(out_s)))
	print("Actual Number:",target_test[x])
	# print("Confidence:%.2f" % (np.amax(out_s)*100)+"%")
	print(out_s)
	if np.argmax(out_s) == target_test[x]:
		count = count + 1

	if target_test[x] == 0:
		tcount0 = tcount0 + 1
		if np.argmax(out_s) == target_test[x]:
			count0 = count0 + 1
	if target_test[x] == 1:
		tcount1 = tcount1 + 1
		if np.argmax(out_s) == target_test[x]:
			count1 = count1 + 1
	if target_test[x] == 2:
		tcount2 = tcount2 + 1
		if np.argmax(out_s) == target_test[x]:
			count2 = count2 + 1
	if target_test[x] == 3:
		tcount3 = tcount3 + 1
		if np.argmax(out_s) == target_test[x]:
			count3 = count3 + 1
	if target_test[x] == 4:
		tcount4 = tcount4 + 1
		if np.argmax(out_s) == target_test[x]:
			count4 = count4 + 1
	if target_test[x] == 5:
		tcount5 = tcount5 + 1
		if np.argmax(out_s) == target_test[x]:
			count5 = count5 + 1
	if target_test[x] == 6:
		tcount6 = tcount6 + 1
		if np.argmax(out_s) == target_test[x]:
			count6 = count6 + 1
	if target_test[x] == 7:
		tcount7 = tcount7 + 1
		if np.argmax(out_s) == target_test[x]:
			count7 = count7 + 1
	if target_test[x] == 8:
		tcount8 = tcount8 + 1
		if np.argmax(out_s) == target_test[x]:
			count8 = count8 + 1
	if target_test[x] == 9:
		tcount9 = tcount9 + 1
		if np.argmax(out_s) == target_test[x]:
			count9 = count9 + 1

print("OVERALL PERCENTAGE:")
print(count,"/",mnist_depth_input)
count = count/100
print(count,"%")
print("")

print("DIGIT 0:")
print(count0,"/",tcount0)
count0 = count0/tcount0
count0 = count0*100
print(count0,"%")
print("")

print("DIGIT 1:")
print(count1,"/",tcount1)
count1 = count1/tcount1
count1 = count1*100
print(count1,"%")
print("")

print("DIGIT 2:")
print(count2,"/",tcount2)
count2 = count2/tcount2
count2 = count2*100
print(count2,"%")
print("")

print("DIGIT 3:")
print(count3,"/",tcount3)
count3 = count3/tcount3
count3 = count3*100
print(count3,"%")
print("")

print("DIGIT 4:")
print(count4,"/",tcount4)
count4 = count4/tcount4
count4 = count4*100
print(count4,"%")
print("")

print("DIGIT 5:")
print(count5,"/",tcount5)
count5 = count5/tcount5
count5 = count5*100
print(count5,"%")
print("")

print("DIGIT 6:")
print(count6,"/",tcount6)
count6 = count6/tcount6
count6 = count6*100
print(count6,"%")
print("")

print("DIGIT 7:")
print(count7,"/",tcount7)
count7 = count7/tcount7
count7 = count7*100
print(count7,"%")
print("")

print("DIGIT 8:")
print(count8,"/",tcount8)
count8 = count8/tcount8
count8 = count8*100
print(count8,"%")
print("")

print("DIGIT 9:")
print(count9,"/",tcount9)
count9 = count9/tcount9
count9 = count9*100
print(count9,"%")
print("")

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