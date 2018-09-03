##### 16-12-4-4 CNN Architecture

### Imports
import numpy as np
from PIL import Image
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

### Constants
# 2D input image (black&white)
input_rows		=		4
input_cols		=		4

# 3D filter
filter_rows		=		4
filter_cols		=		4
filter_depth	=		8

# 3D convolutional layer
conv_rows		=		(input_rows - filter_rows) + 1
conv_cols		=		(input_cols - filter_cols) + 1
conv_depth		=		8								### MUST BE THE SAME WITH FILTER DEPTH
conv_total		=		conv_rows*conv_cols*conv_depth

# 2D hidden layer (4 nodes)
hidden_rows		=		2
hidden_cols		=		2
hidden_total	=		hidden_rows*hidden_cols

# 2D output layer (4 outputs)
output_rows		=		2
output_cols		=		2
output_total	=		output_rows*output_cols

# Epoch
epoch = 1

# Learning Rate
lr = 0.1

### Initialization
## Layers
conv_layer = np.zeros([conv_depth, conv_rows,conv_cols])

hidden_layer = np.zeros([hidden_rows, hidden_cols])

output_layer = np.zeros([output_rows, output_cols])

## Weights (Random values with a distribution from 0 to 1)
filter_weights = np.random.rand(filter_depth, filter_rows, filter_cols)

cl_to_hl_weights = np.random.rand(conv_total, hidden_total)

hl_to_ol_weights = np.random.rand(hidden_total, output_total)

### Functions
# Set number of decimal places in printing
np.set_printoptions(precision=4)

# Sigmoid Function for forward and backprog
def sigmoid_function(x,deriv):
	if deriv==False:
		return 1/(1+np.exp(-x))
	else:
		return x*(1-x)

# Sets the target array
def set_target(x):
	if x==1:
		return np.array([[1,0,0,0]])
	elif x==2:
		return np.array([[0,1,0,0]])
	elif x==3:
		return np.array([[0,0,1,0]])
	elif x==4:
		return np.array([[0,0,0,1]])

# Softmax Function for forward and backprog
def softmax(x,deriv):
	if deriv==False:
	    e_x = np.exp(x-np.max(x))
	    return e_x / e_x.sum(axis=1)
	else:
		return x*(1-x)


# Gets the image for the input layer then convert to numpy array
def get_image(filename):
	try:
		return np.invert(np.array(Image.open(filename).convert("L")))/255
	except IOError:
		print("Error in getting image file: File not available!")

def cnn(input_layer, input_filter, conv_layer, fc1, hidden_layer, fc2,output_layer, target_value):
	for i in range(0, epoch):
		##### START OF FORWARD PASS
		# Convolution
		for i in range(0, input_filter.shape[0]):
			conv_layer[i] = signal.convolve(input_layer, input_filter[i], mode="valid")

		# Non-linearity for conv layer
		conv_layer_sigmoid = sigmoid_function(conv_layer,deriv=False)

		# Flattening of sigmoid activated conv layer ((3,2,2) to (1,12))
		conv_layer_sigmoid_flat = conv_layer_sigmoid.reshape(1,conv_total)

		# Dot product of conv layer and fc1 to produce hidden layer
		hidden_layer = np.matmul(conv_layer_sigmoid_flat, fc1)

		# Non-linearity for hidden layer
		hidden_layer_sigmoid = sigmoid_function(hidden_layer, deriv=False)

		# Dot product of hidden layer and fc2 to produce output layer
		output_layer = np.matmul(hidden_layer_sigmoid, fc2)

		# Softmax activation of the output layer
		output_layer_softmax = softmax(output_layer, deriv=False)

		##### START OF BACKPROPAGATION
		### Slope calculation
		slope_output_layer = sigmoid_function(output_layer_softmax, deriv=True)
		slope_hidden_layer = sigmoid_function(hidden_layer_sigmoid, deriv=True)
		slope_conv_layer = sigmoid_function(conv_layer_sigmoid_flat, deriv=True)

		# Error at output layer (1x4)
		error_at_output_layer = output_layer_softmax - set_target(target_value) 

		# delta at output layer (1x4)
		delta_at_output_layer = error_at_output_layer * slope_output_layer

		# Error at hidden layer (1x4)
		error_at_hidden_layer = np.matmul(delta_at_output_layer, np.transpose(fc2))

		# delta at hidden layer (1x4)
		delta_at_hidden_layer = error_at_hidden_layer * slope_hidden_layer

		# Error at conv layer (1x12)
		error_at_conv_layer = np.matmul(delta_at_hidden_layer, np.transpose(fc1))

		# delta at conv layer (1x12)
		delta_at_conv_layer = error_at_conv_layer * slope_conv_layer

		### Updating of weights
		# Weights from hidden layer to output layer (4x4)
		fc2 = fc2 + (np.matmul(np.transpose(hidden_layer_sigmoid), delta_at_output_layer) * lr)

		# Weights from conv layer to hidden layer (12x4)
		fc1 = fc1 + (np.matmul(np.transpose(conv_layer_sigmoid_flat), delta_at_hidden_layer) * lr)

		# Filters
		###### Extra note: Previously, we did not update with respect to the learning rate. We simply convolved, hence the filters were overwritten
		for x in range(0, input_filter.shape[0]):
			input_filter[x] = input_filter[x] + (signal.convolve(input_layer, slope_conv_layer.reshape(conv_depth,conv_rows,conv_cols)[x], mode="valid") * lr)
	return input_filter, fc1, fc2, output_layer, output_layer_softmax

### Program Flow
image1 = "yappie.png"
image2 = "5.png"
image3 = "6.png"
image4 = "3.png"

print("Before")
print(filter_weights)

for z in range(0,1000):
	print(z)
	input_layer = get_image(image1)
	f,fc1,fc2,out,out_s = cnn(input_layer, filter_weights, conv_layer, cl_to_hl_weights, hidden_layer, hl_to_ol_weights, output_layer, 1)

	input_layer = get_image(image2)
	f,fc1,fc2,out,out_s = cnn(input_layer, filter_weights, conv_layer, fc1, hidden_layer, fc2, output_layer, 2)

	input_layer = get_image(image3)
	f,fc1,fc2,out,out_s = cnn(input_layer, filter_weights, conv_layer, fc1, hidden_layer, fc2, output_layer, 3)

	# input_layer = get_image(image4)
	# f,fc1,fc2,out,out_s = cnn(input_layer, filter_weights, conv_layer, fc1, hidden_layer, fc2, output_layer, 4)

print("Filter after iterations:")
print(f)
print("Output Layer after iterations:")
print(out)
print(out_s)
plt.subplot(4,4,1).set_title("Final Filter 1")
plt.imshow(f[0])
plt.axis('off')
plt.subplot(4,4,2).set_title("Final Filter 2")
plt.imshow(f[1])
plt.axis('off')
plt.subplot(4,4,3).set_title("Final Filter 3")
plt.imshow(f[2])
plt.axis('off')
plt.subplot(4,4,4).set_title("Final Filter 4")
plt.imshow(f[3])
plt.axis('off')
plt.subplot(4,4,5).set_title("Final Filter 5")
plt.imshow(f[4])
plt.axis('off')
plt.subplot(4,4,6).set_title("Final Filter 6")
plt.imshow(f[5])
plt.axis('off')
plt.subplot(4,4,7).set_title("Final Filter 7")
plt.imshow(f[6])
plt.axis('off')
plt.subplot(4,4,8).set_title("Final Filter 8")
plt.imshow(f[7])
plt.axis('off')
# plt.subplot(4,4,9).set_title("Final Filter 9")
# plt.imshow(f[8])
# plt.axis('off')
# plt.subplot(4,4,10).set_title("Final Filter 10")
# plt.imshow(f[9])
# plt.axis('off')
# plt.subplot(4,4,11).set_title("Final Filter 11")
# plt.imshow(f[10])
# plt.axis('off')
# plt.subplot(4,4,12).set_title("Final Filter 12")
# plt.imshow(f[11])
# plt.axis('off')
# plt.subplot(4,4,13).set_title("Final Filter 13")
# plt.imshow(f[12])
# plt.axis('off')
# plt.subplot(4,4,14).set_title("Final Filter 14")
# plt.imshow(f[13])
# plt.axis('off')
# plt.subplot(4,4,15).set_title("Final Filter 15")
# plt.imshow(f[14])
# plt.axis('off')
# plt.subplot(4,4,16).set_title("Final Filter 16")
# plt.imshow(f[15])
# plt.axis('off')
plt.show()