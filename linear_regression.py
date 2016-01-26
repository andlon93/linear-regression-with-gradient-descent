# --- Function that read data from file
#     Input: type e.g. train or test
#     Output: lists with inputs and outputs
def read_file(type):
	fil=open("data-"+type+".csv","r")
	x1=[]
	x2=[]
	y=[]
	for line in fil:
		_x1,_x2,_y = line.strip().split(",")
		x1.append(float(_x1))
		x2.append(float(_x2))
		y.append(float(_y))
	fil.close()
	return x1, x2, y
#
x1_train, x2_train, y_train = read_file("train")
x1_test, x2_test, y_test = read_file("test")