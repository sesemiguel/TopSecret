import numpy as np
# Number of Depth of filter & conv nodes
depth 			=	8									

#
mnist_depth		=	100

# Input Layer Dimensions
input_rows		=	14
input_cols		=	14

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
iteration		=	20				

# Learning Rate
L_rate			=	.001

# Temporary value for incrementing convolution nodes							
temp = 0 	
mnist_array = np.zeros([mnist_depth, input_rows, input_cols])		
# Writing CSV to 3d Arrays
def pd_csv_to_3darray(filename_input):
	try:
		temp_array_1 = np.genfromtxt(filename_input,delimiter=',')
		temp_array_2 = np.zeros([depth,input_rows,input_cols])
		count = 1
		for x in range(0,f_rows):
			for y in range(0,f_cols):
				for z in range(0,depth):
					temp_array_2[z][y][x] = temp_array_1[count][3]
					count += 1
		np.delete(temp_array_2, (0), axis=0)
		return temp_array_2
	except IOError:
		print("File not available!")

print(pd_csv_to_3darray("testarray.csv"))