# LIBRARIES USED
import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# SET NUMBER OF DECIMAL PLACES
np.set_printoptions(precision=4)

# TARGET ARRAYS VALUES
target_array_0 = np.array([1,0,0,0,0,0,0,0,0,0])
target_array_1 = np.array([0,1,0,0,0,0,0,0,0,0])
target_array_2 = np.array([0,0,1,0,0,0,0,0,0,0])
target_array_3 = np.array([0,0,0,1,0,0,0,0,0,0])
target_array_4 = np.array([0,0,0,0,1,0,0,0,0,0])
target_array_5 = np.array([0,0,0,0,0,1,0,0,0,0])
target_array_6 = np.array([0,0,0,0,0,0,1,0,0,0])
target_array_7 = np.array([0,0,0,0,0,0,0,1,0,0])
target_array_8 = np.array([0,0,0,0,0,0,0,0,1,0])
target_array_9 = np.array([0,0,0,0,0,0,0,0,0,1])

# SET TARGET ARRAY
target_array = target_array_3

####################################################
#           INPUT FOR CONVOLUTION ARRAY            #
####################################################
input_img = np.invert(np.array(Image.open("sample.png").convert("L")))
print(input_img)
# plt.imshow(input_img)
# plt.title("Input Image")
# plt.axis('off')
# plt.show()

# CONSTANTS
f_depth 		=	8
f_rows 			=	5
f_cols 			=	5
outputs 		=	10
conv_rows 		=	(input_img.shape[0] - f_rows) + 1
conv_cols		=	(input_img.shape[1] - f_cols) + 1
total_weights	=	f_depth*conv_rows*conv_cols
iterations		=	10
L_rate			=	.001

filename_FC 	=   "FC_weights.csv"
filename_Filter =	"Filter_weights.csv"

# 0 if no, 1 if yes
training 		= 	1

# 0 if yes, 1 if no
first_run		= 	0

# FUNCTIONS
def new_weights_random(x_size, y_size, flag):
	if flag == 0:
		return np.random.rand(x_size, y_size)
	elif flag == 1:
		return fc_weight

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def relu_function(input_layer):
	return np.maximum(input_layer, 0)

def sigmoid_function(x):
	return 1/(1+np.exp(-x))

def error_calculation(target_array, output_array):
	return output_array - target_array

def relu_activation(data_array):
    return np.maximum(data_array, 0)

def pd_2darray_to_csv(input_array, filename_output):
	np.savetxt(filename_output, input_array, delimiter=",")


def pd_csv_to_2darray(input_filename):
	try:
		return np.genfromtxt(input_filename, delimiter=',')
	except IOError:
		print("File not available!")

def pd_3darray_to_csv(input_array, filename_output):
	stacked = pd.Panel(input_array.swapaxes(1,2)).to_frame().stack().reset_index()
	stacked.columns = ['x', 'y', 'z', 'value']
	stacked.to_csv(filename_output, index=False)

def pd_csv_to_3darray(filename_input):
	try:
		temp_array_1 = np.genfromtxt(filename_input,delimiter=',')
		temp_array_2 = np.zeros([f_depth,f_rows,f_cols])
		count = 1
		for x in range(0,f_rows):
			for y in range(0,f_cols):
				for z in range(0,f_depth):
					temp_array_2[z][y][x] = temp_array_1[count][3]
					count += 1
		return temp_array_2
	except IOError:
		print("File not available!")

def forwardpass(convolved_nodes, input_img, input_filter, total_weights, convolved_nodes_to_output_nodes, target_array):

	# CONVOLUTION
	for x in range(0, input_filter.shape[0]):
		convolved_nodes[x] = signal.convolve(input_img, input_filter[x], mode="valid")

	# SIGMOID ACTIVATION OF CONVOLUTION NODE
	convolved_nodes_sigmoid = sigmoid_function(convolved_nodes)

	# FLATTENING OF SIGMOID ACTIVATED CONVOLUTION LAYER
	convolved_nodes_sigmoid_flat = convolved_nodes_sigmoid.reshape(1,total_weights)

	# FULLY CONNECTED LAYER
	output_nodes_flat = np.matmul(convolved_nodes_sigmoid_flat, convolved_nodes_to_output_nodes)

	# SOFTMAX ACTIVATION OF OUTPUT NODE
	output_nodes_flat_column = np.transpose(output_nodes_flat)#.reshape(10,1)#for column comparison
	softmax_output = softmax(output_nodes_flat_column)

	# ERROR CALCULATION
	softmax_output_row = np.transpose(softmax_output)#.reshape(1,10)
	error_array = error_calculation(target_array,softmax_output_row)


	return error_array, softmax_output_row, convolved_nodes_sigmoid_flat, convolved_nodes_to_output_nodes, input_filter, convolved_nodes

def backpropagation(error_array, softmax_output_row, convolved_nodes_sigmoid_flat, total_weights, convolved_nodes_to_output_nodes, L_rate, f_depth, conv_rows, conv_cols, input_img, convolved_nodes):
	
	temp = 0
	A = error_array * softmax_output_row*(1-softmax_output_row)
	A = np.transpose(A)
	B = np.matmul(A,convolved_nodes_sigmoid_flat)
	B = np.transpose(B)

	for i in range(0,total_weights):
		for j in range(0,outputs):
			convolved_nodes_to_output_nodes[i][j] = convolved_nodes_to_output_nodes[i][j] - L_rate * B[i][j]

	for i in range(0,total_weights):
		for j in range(0,outputs):
			convolved_nodes_sigmoid_flat[0][i] = (softmax_output_row[0][j] * convolved_nodes_to_output_nodes[i][j]) + temp 
			temp = convolved_nodes_sigmoid_flat[0][i]
		temp = 0
	convolved_nodes = convolved_nodes_sigmoid_flat.reshape(f_depth,conv_rows,conv_cols)

	for x in range(0, input_filter.shape[0]):
		input_filter[x] = signal.convolve(input_img, convolved_nodes[x], mode="valid")

	return input_filter, convolved_nodes_to_output_nodes

def get_predicted_digit(softmax_array):
	return np.unravel_index(np.argmax(softmax_array, axis=None),softmax_array.shape)

# INPUT FILTER
#input_filter = np.random.rand(f_depth ,f_rows, f_cols)*255#new_weights_random(5,5,0)#np.array([[0,0,0,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,0]])
#print(input_filter)
#plt.imshow(input_filter)
#plt.show()

#convolved_nodes_to_output_nodes = new_weights_random(total_weights, outputs, 0)
convolved_nodes = np.zeros([f_depth,conv_rows,conv_cols])

if first_run == 0:
	convolved_nodes_to_output_nodes = new_weights_random(total_weights, outputs, 0)
	input_filter = np.random.rand(f_depth ,f_rows, f_cols)*255
else:
	convolved_nodes_to_output_nodes = pd_csv_to_2darray(filename_FC)
	input_filter = pd_csv_to_3darray(filename_Filter)
	
if training == 1:
	
	for i in range(0,iterations):
		error_array, softmax_output_row, convolved_nodes_sigmoid_flat, convolved_nodes_to_output_nodes, input_filter, convolved_nodes = forwardpass(convolved_nodes, input_img, input_filter, total_weights, convolved_nodes_to_output_nodes, target_array)
		input_filter, convolved_nodes_to_output_nodes = backpropagation(error_array, softmax_output_row, convolved_nodes_sigmoid_flat, total_weights, convolved_nodes_to_output_nodes, L_rate, f_depth, conv_rows, conv_cols, input_img, convolved_nodes)
		pd_2darray_to_csv(convolved_nodes_to_output_nodes, filename_FC)
		pd_3darray_to_csv(input_filter, filename_Filter)
		print(i)

elif training == 0:
	error_array, softmax_output_row, convolved_nodes_sigmoid_flat, convolved_nodes_to_output_nodes, input_filter, convolved_nodes = forwardpass(convolved_nodes, input_img, input_filter, total_weights, convolved_nodes_to_output_nodes, target_array)

	# TESTING VARIABLES
	print("ERROR")
	print(error_array)
	print("SOFTMAX")
	print(softmax_output_row)
	print("The predicted digit is", get_predicted_digit(softmax_output_row))


plt.subplot(341).set_title("Filter 1")
plt.imshow(input_filter[0])
plt.axis('off')
plt.subplot(342).set_title("Filter 2")
plt.imshow(input_filter[1])
plt.axis('off')
plt.subplot(343).set_title("Filter 3")
plt.imshow(input_filter[2])
plt.axis('off')
plt.subplot(344).set_title("Filter 4")
plt.imshow(input_filter[3])
plt.axis('off')
plt.subplot(345).set_title("Filter 5")
plt.imshow(input_filter[4])
plt.axis('off')
plt.subplot(346).set_title("Filter 6")
plt.imshow(input_filter[5])
plt.axis('off')
plt.subplot(347).set_title("Filter 7")
plt.imshow(input_filter[6])
plt.axis('off')
plt.subplot(348).set_title("Filter 8")
plt.imshow(input_filter[7])
plt.axis('off')
plt.show()