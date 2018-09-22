import numpy as np
import pandas as pd


mnist_d 			= 100
mnist_r 			= 28
mnist_c 			= 28
max_pool_size 		= 3
Max_Pool_Shape_X 	= int(mnist_r/max_pool_size)
Max_Pool_Shape_Y 	= int(mnist_c/max_pool_size)
maxpooled_array 	= np.zeros((mnist_d,Max_Pool_Shape_X,Max_Pool_Shape_Y))
target_csv 			= np.zeros(mnist_d)
target_array		= np.zeros(10)


# Rewriting CSV file to 2d arrays
def pd_csv_to_2darray(input_filename):
	try:
		return np.genfromtxt(input_filename, delimiter=',')
	except IOError:
		print("File not available!")

def pd_3darray_to_csv(input_array, filename_output):
	stacked = pd.Panel(input_array.swapaxes(1,2)).to_frame().stack().reset_index()
	stacked.columns = ['x', 'y', 'z', 'value']
	stacked.to_csv(filename_output, index=False)
	print("Generated CSV for filters weights")

def automate_target():
	target_csv = pd_csv_to_2darray("target_array.csv")
	print(target_csv)
	for y in range (0, mnist_d):
		if target_csv[y] == 0:
			target_array = [1,0,0,0,0,0,0,0,0,0]
		elif target_csv[y] == 1:
			target_array = [0,1,0,0,0,0,0,0,0,0]
		elif target_csv[y] == 2:
			target_array = [0,0,1,0,0,0,0,0,0,0]
		elif target_csv[y] == 3:
			target_array = [0,0,0,1,0,0,0,0,0,0]
		elif target_csv[y] == 4:
			target_array = [0,0,0,0,1,0,0,0,0,0]
		elif target_csv[y] == 5:
			target_array = [0,0,0,0,0,1,0,0,0,0]
		elif target_csv[y] == 6:
			target_array = [0,0,0,0,0,0,1,0,0,0]
		elif target_csv[y] == 7:
			target_array = [0,0,0,0,0,0,0,1,0,0]
		elif target_csv[y] == 8:
			target_array = [0,0,0,0,0,0,0,0,1,0]
		elif target_csv[y] == 9:
			target_array = [0,0,0,0,0,0,0,0,0,1]
		else:
			target_array = [0,0,0,0,0,0,0,0,0,0]
		print(target_array)

def max_pooling_area(matrix,i=0,j=0,data=0):
	temp_i = i
	temp_j = j
	data = matrix[i][j] #get data
	j=j+1
	tempdata = matrix[i][j]
	if data <= tempdata:
		data = tempdata
		temp_i = i
		temp_j = j
	else:
		data = data

	j=j-1
	i=i+1

	tempdata = matrix[i][j]
	if data <= tempdata:
		data = tempdata
		temp_i = i
		temp_j = j
	else:
		data = data

	j=j+1
	tempdata = matrix[i][j]
	if data <= tempdata:
		data = tempdata
		temp_i = i
		temp_j = j
	else:
		data = data
	return data, temp_i, temp_j

def max_pooling(img_array_input):
	i=0
	j=0
	data=0

	Input_array_shape_X = mnist_r
	Max_Pool_Shape_X = int(Input_array_shape_X/max_pool_size)

	Input_array_shape_Y = mnist_c
	Max_Pool_Shape_Y = int(Input_array_shape_Y/max_pool_size)

	max_pooled_list = np.zeros((Max_Pool_Shape_X,Max_Pool_Shape_Y))
	max_pooled_weights = np.zeros((Input_array_shape_X,Input_array_shape_Y))

	for x in range(0,Max_Pool_Shape_X):
		for y in range (0,Max_Pool_Shape_Y):
			data, m, n = max_pooling_area(img_array_input,i,j,data)
			max_pooled_weights[m][n] = 1
			j = j+max_pool_size
			
			max_pooled_list[x][y] = data
		j = 0
		i = i+max_pool_size
	return max_pooled_list



mnist_train = pd_csv_to_2darray("mnist_train_100.csv")
# print(mnist_train)

mnist_train_target = mnist_train[:,0] #extract the targets
# print("EXTRACTED")
# print(mnist_train_target)

mnist_train_clean = np.delete(mnist_train, 0, axis=1) #remove the first column
# print("No First Column")
# print(mnist_train_clean)

mnist_train_batch = np.zeros((mnist_d,mnist_r,mnist_c)) #initialize the placeholding shit

for i in range(0, mnist_d):
	mnist_train_batch[i] = np.reshape(mnist_train_clean[i,:], (mnist_r, mnist_c)) #EXTRACT DATA AND RESHAPE TO SPECIFIC ROW COL
	print("Matrix Number: " + str(i))
	# print(mnist_train_batch[i])

for x in range(0, mnist_d):
	maxpooled_array[x] = max_pooling(mnist_train_batch[x]) 

pd_3darray_to_csv(maxpooled_array, "mnist_full_data_3d_9x9_test_100.csv")


############### DATA ##############
mnist_train_batch = mnist_train_batch.transpose(0,1,2).reshape(-1,mnist_train_batch.shape[1])
# np.savetxt("mnist_full_9x9_train_1.csv", mnist_train_batch, delimiter=",")

######### MAX POOLED DATA #########
maxpooled_array = maxpooled_array.transpose(0,1,2).reshape(-1,maxpooled_array.shape[1])
np.savetxt("mnist_full_maxpooled_9x9_test_100.csv", maxpooled_array, delimiter=",")

np.savetxt("target_array_test_100.csv", mnist_train_target, delimiter=",")