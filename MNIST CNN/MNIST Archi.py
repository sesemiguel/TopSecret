##### MNIST Architecture
### Imports
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

### Constants
# 2D input image (black&white)
input_rows		=		28
input_cols		=		28

# 3D filter
filter_rows		=		14
filter_cols		=		14
filter_depth	=		16

# 3D convolutional layer
conv_rows		=		(input_rows - filter_rows) + 1
conv_cols		=		(input_cols - filter_cols) + 1
conv_depth		=		filter_depth	
conv_total		=		conv_rows*conv_cols*conv_depth

# 2D hidden layer (4 nodes)
hidden_rows		=		3
hidden_cols		=		3
hidden_total	=		hidden_rows*hidden_cols

# 2D output layer (4 outputs)
output_rows		=		1
output_cols		=		10
output_total	=		output_rows*output_cols

# Epoch
epoch = 25

# Learning Rate
lr = 0.5

# MNIST
mnist_d 			= 5000
mnist_r 			= 28
mnist_c 			= 28
mnist_filename_train = "mnist_train_5000.csv"
mnist_filename_test = "mnist_test_10.csv"

### Initialization
# Layers
conv_layer = np.zeros([conv_depth, conv_rows,conv_cols])
hidden_layer = np.zeros([hidden_rows, hidden_cols])
output_layer = np.zeros([output_rows, output_cols])

# Weights (Random values with a distribution from 0 to 1)
filter_weights = np.random.rand(filter_depth, filter_rows, filter_cols)
cl_to_hl_weights = np.random.rand(conv_total, hidden_total)
hl_to_ol_weights = np.random.rand(hidden_total, output_total)

# Set number of decimal places in printing
np.set_printoptions(precision=4)

### Functions
# Sigmoid Function for forward and backprog
def sigmoid_function(x,deriv):
	if deriv==False:
		return 1/(1+np.exp(-x))
	else:
		return x*(1-x)

# Sets the target array
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

# Softmax Function for forward and backprog
def softmax(x,deriv):
	if deriv==False:
	    e_x = np.exp(x-np.max(x))
	    return e_x / e_x.sum(axis=1)
	else:
		return x*(1-x)

# Rewriting CSV file to 2d arrays
def pd_csv_to_2darray(input_filename):
	try:
		return np.genfromtxt(input_filename, delimiter=',')
	except IOError:
		print("File not available!")

def mnist_csv_to_numpy_array(mnist_d, mnist_r, mnist_c, mnist_filename):
	# Initialize Holder Array for MNIST Digits
	mnist_train_batch 	= np.zeros((mnist_d,mnist_r,mnist_c))

	# Get MNIST CSV file
	mnist_train = pd_csv_to_2darray(mnist_filename)

	# Extract the Targets
	mnist_train_target = mnist_train[:,0] 

	# Remove the First Column
	mnist_train_clean = np.delete(mnist_train, 0, axis=1) #remove the first column

	# Extract Data and Reshape to Specific Row and Column
	for i in range(0, mnist_d):
		mnist_train_batch[i] = np.reshape(mnist_train_clean[i,:], (mnist_r, mnist_c))

	return mnist_train_batch, mnist_train_target

def cnn(input_layer, input_filter, fc1, fc2, target_value):
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
	# Slope calculation
	slope_output_layer = softmax(output_layer_softmax, deriv=True)
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
	# Weights from hidden layer to output layer (4x4) --- Cost function
	fc2 = fc2 + (np.matmul(np.transpose(hidden_layer_sigmoid), delta_at_output_layer) * lr)

	# Weights from conv layer to hidden layer (12x4) --- Cost function
	fc1 = fc1 + (np.matmul(np.transpose(conv_layer_sigmoid_flat), delta_at_hidden_layer) * lr)

	# Filters
	###### Extra note: Previously, we did not update with respect to the learning rate. We simply convolved, hence the filters were overwritten
	for x in range(0, input_filter.shape[0]):
		input_filter[x] = input_filter[x] + (signal.convolve(input_layer, delta_at_conv_layer.reshape(conv_depth,conv_rows,conv_cols)[x], mode="valid")) * lr
	# Return the edges and the output
	return input_filter, fc1, fc2, output_layer, output_layer_softmax

### Program Flow
# Get MNIST Inputs and Corresponding Digit
MNIST_inputs_train, MNIST_digit_train = mnist_csv_to_numpy_array(mnist_d, mnist_r, mnist_c, mnist_filename_train)

MNIST_inputs_test, MNIST_digit_test = mnist_csv_to_numpy_array(10, mnist_r, mnist_c, mnist_filename_test)

# Passing first randomized weights to iterative weights
f = filter_weights
fc1 = cl_to_hl_weights
fc2 = hl_to_ol_weights


# Training
for i in range(0, mnist_d):
	print(i)
	f,fc1,fc2,out,out_s = cnn(MNIST_inputs_train[i], f, fc1, fc2, MNIST_digit_train[i])

f,fc1,fc2,out,out_s = cnn(MNIST_inputs_test[2], f, fc1, fc2, MNIST_digit_test[2])

### Output
print(f)

print("Output Layer after iterations:")
print(out_s)

### Plotting
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
plt.subplot(4,4,9).set_title("Final Filter 9")
plt.imshow(f[8])
plt.axis('off')
plt.subplot(4,4,10).set_title("Final Filter 10")
plt.imshow(f[9])
plt.axis('off')
plt.subplot(4,4,11).set_title("Final Filter 11")
plt.imshow(f[10])
plt.axis('off')
plt.subplot(4,4,12).set_title("Final Filter 12")
plt.imshow(f[11])
plt.axis('off')
plt.subplot(4,4,13).set_title("Final Filter 13")
plt.imshow(f[12])
plt.axis('off')
plt.subplot(4,4,14).set_title("Final Filter 14")
plt.imshow(f[13])
plt.axis('off')
plt.subplot(4,4,15).set_title("Final Filter 15")
plt.imshow(f[14])
plt.axis('off')
plt.subplot(4,4,16).set_title("Final Filter 16")
plt.imshow(f[15])
plt.axis('off')
plt.show()