### IMPORTS
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

asdfasdf = pd_csv_to_2darray('mnist_30_digit_maxpooled_9x9_test_subset_5.csv')
print("RAW CSV")
print(asdfasdf)

asdfasdf = np.where(asdfasdf==1.0, 1, 0)
print(asdfasdf)

print("force -freeze sim:/controller/clk 1 0, 0 {50 ns} -r 100")

def init():
	print("force res 1")
	print("force extctrl 1")
	print("force convchange 1")
	print("force convstart 1")
	print("force fcchange 1")

def reset():
	print("force res 0")
	print("force extctrl 0")
	print("force convchange 0")
	print("force convstart 0")
	print("force fcchange 0")
	print("run 1ns")

def cancer():
	print("run 1ns")
	print("force res 0")
	print("run 1500000ns")

n = 1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x][y])
cancer()
reset()

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1

init()
for x in range(0,9):
	for y in range(0,9):
		print("force a"+str(x)+str(y), asdfasdf[x+9*n][y])
cancer()
reset()
n=n+1