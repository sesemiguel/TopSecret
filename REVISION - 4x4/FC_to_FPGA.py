import numpy as np
from scipy import signal
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

def pd_csv_to_2darray(input_filename):
	try:
		return np.genfromtxt(input_filename, delimiter=',')
	except IOError:
		print("File not available!")

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val & ((2 ** bits) - 1)     # return positive value as is

FC = pd_csv_to_2darray("FC.csv")
FC_reshape = FC.ravel()

# for x in range(0, FC.size):
# 	# if FC_reshape[x] > 0: #POSITIVE
# 	# 	binary = "0"+str(format(int(FC_reshape[x]*8192), '017b'))
# 		print("w"+str(x+1),"<= "+str(FC_reshape[x])+";")

# 	if FC_reshape[x] < 0: #NEGATIVE
# 		binary = str(format(int(FC_reshape[x]*8192), '018b'))
# 		binary = twos_comp(int(binary,2), len(binary))
# 		binary = "{0:b}".format(binary)
# 		print("w"+str(x+1),"<= \""+binary+"\""+";")

for x in range(0, FC.size):
	if FC_reshape[x] > 0: #POSITIVE
		binary = "0"+str(format(int(FC_reshape[x]*256), '011b'))
		print("w"+str(x+1),"<= \""+binary+"\""+";")

	if FC_reshape[x] < 0: #NEGATIVE
		binary = str(format(int(FC_reshape[x]*256), '012b'))
		binary = twos_comp(int(binary,2), len(binary))
		binary = "{0:b}".format(binary)
		print("w"+str(x+1),"<= \""+binary+"\""+";")

filter = [1.030389247,
0.329104487,
0.838894304,
1.29434134,
0.955414892,
0.757311319,
1.287986903,
1.183253723,
0.759425073,
1.235858509,
0.317233123,
0.938138752,
1.51403056,
1.114477104,
0.958244252,
1.40286732,
1.356468868,
0.873553076,
1.246937998,
0.405677919,
1.004736287,
1.526415106,
1.170582719,
0.883142947,
1.461518892,
1.381261404,
0.898359274
]

char = ["b","c","d","e","f","g","h","i"]

# for x in range(0, 72):
# 		binary[] = "0"+str(format(int(filter[x]*1024), '011b'))
# for y in range(0,3):

# 		print("w"+str(x+1),"<= \""+binary+"\"")
counter = 0
for r in range (0,9):
	for t in range (0,3):
		for y in range (0,3):
			print(char[r]+"("+str(t)+","+str(y)+") <= "+ "\"0"+str(format(int(filter[counter]*128), '011b'))+"\""+";")
			counter = counter + 1

# for x in range(0, FC.size):
# 	# if FC_reshape[x] > 0: #POSITIVE
# 	# 	binary = "0"+str(format(int(FC_reshape[x]*8192), '017b'))
# 		print(str(FC_reshape[x]))