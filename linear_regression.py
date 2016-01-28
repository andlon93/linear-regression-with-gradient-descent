import numpy as np
# --- Function that read data from file
#     Input: type e.g. train or test
#     Output: lists with inputs and outputs
def read_file(type):
	fil=open("data-"+type+".csv","r")
	x=[]
	y=[]
	for line in fil:
		_x1,_x2,_y = line.strip().split(",")
		x.append([float(_x1),float(_x2)])
		y.append(float(_y))
	fil.close()
	return x, y
#
# --- Function maps x-vector inputs to y-values
#     Input: x-vector, bias, weight
#     Output: y
def f(x_vector, bias, weight):
	y = bias
	for x in x_vector:
		y = np.add(y, np.dot(weight, x))
	return y
#
# --- Function that calculate how good the current model is
#     Input: bias, weight(the model), y_results, y_fasit
#     Output: Error
def MSE(weight, bias, data, svar):
	Error = 0
	n = len(data)
	for i in range(n):
		Error += (f(data[i], bias, weight) - svar[i])**2
	return Error/n
#
def gradients(bias, weight, data, svar, lr):
    n = len(data)
    for i in range(n):
    	x = np.array(data[i])
    	y = np.array(svar[i])
    	b_grad = ((2/n) * (f(data[i], bias, weight) - y)*-lr)
    	w_grad = ( (2/n) * np.dot(x, (f(data[i], bias, weight)-y)) )*-lr
    	bias += b_grad
    	weight += w_grad
    #print ("B,W",[bias, weight])
    return bias, weight



x_train, y_train = read_file("train")
x_test, y_test = read_file("test")
bias=0
weight=0
for n in range(1000):
	#print(f(x_train[n], 1, 2))
	bias, weight = gradients(bias, weight, x_train, y_train, 0.01)
	if n==4 or n==9:
		print("Train data: ",MSE(weight, bias, x_train, y_train))
		print("Test data: ",MSE(weight, bias, x_test, y_test))
		print ("weight:", weight, " Bias:", bias, "\n")
print("Train data: ",MSE(weight, bias, x_train, y_train))
print("Test data: ",MSE(weight, bias, x_test, y_test))
print ("weight:", weight, " Bias:", bias)
#x_test, y_test = read_file("test")'''
