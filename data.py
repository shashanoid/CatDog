import numpy as np 
import random

data = np.load('/Users/shashwatsingh/Desktop/catdog/train_data.npy')

def create_data():
	trndata = []
	results = []

	for element in data:
		trndata.append(element[0])
		results.append(element[1])

	train_array = np.asarray(trndata)
	result_array = np.asarray(results)

	train_new = [np.true_divide(x, [255.0]) for x in train_array]

	inputs = [np.reshape(x, (2500, 1)) for x in train_new]
	results = [np.reshape(x, (2, 1)) for x in result_array]

	
	training_data = zip(inputs, results)

	res = []
	for x, y in training_data:
		if np.max(y[1][0]) == 1:
			res.append(1)
		else:
			res.append(0)

	test_data = zip(inputs, res)
	random.shuffle(test_data)

	return (training_data, test_data[1:3000])