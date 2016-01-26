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
		y += x*weight
	return y
#
# --- Function that calculate how good the current model is
#     Input: bias, weight(the model), y_results, y_fasit
#     Output: Error
def MSE(weight, bias, x_vector, y_fasit):
	Loss = 0
	for i in range(len(y_fasit)):
		Loss += ((f(x_vector[i], bias, weight) - y_fasit[i])**2)
	return Loss/len(y_fasit)
#
def partial_derivative_L_of_W(weight, bias, x_vector, y_fasit):
	result = 0
	for i in range(len(y_fasit)):
		result += (f(x_vector[i], bias, weight) - y_fasit)*x_vector[i]
	return result * (2/len(y_fasit))
def partial_derivative_L_of_B(weight, bias, x_vector, y_fasit):
	result = 0
	for i in range(len(y_fasit)):
		result += (f(x_vector[i], bias, weight) - y_fasit)
	return result * (2/len(y_fasit))


#x_train, y_train = read_file("train")
#x_test, y_test = read_file("test")