
from __future__ import division
import math
import numpy as np
from random import shuffle

def h(theta, x):
	return 1/(1 + np.exp(-1*(theta.T.dot(x))))

def gradient_descent(X , Y , theta , alpha , m):
	new_theta = theta - ((alpha/m) * np.dot((h(theta, X.T ) - Y) , X))
	return new_theta


with open("data_banknote_authentication.txt") as file:
	data = []
	for line in file:
		data.extend(line.rstrip("\r\n").split(","))

train_size = 1098
m = 1372
iterations = 5

avg_accuracy = 0.0
avg_recall = 0.0
avg_precision = 0.0
avg_f1score = 0.0
avg_times_to_converge = 0.0
for i in range(iterations):
	print "Iteration #"+str(i+1)

	np_data = np.array(data).reshape(-1,5)
	np_data = np_data.astype(float)
	np.random.shuffle(np_data)
	theta = np.zeros(4)
	X = np_data[:train_size,0:4]
	Y = np_data[:train_size,4]
	
	old_delta_theta = theta
	delta_theta = theta
	converged = False
	counter = 0
	while not converged:
		counter +=1
		new_theta = gradient_descent(X,Y,theta, 0.5 , m )
		old_delta_theta = delta_theta
		delta_theta = np.abs(new_theta - theta)
		theta = new_theta
		#print "Theta:"
		#print theta
		#print "Theta Difference:"
		#print (old_delta_theta - delta_theta)
		converged = ((old_delta_theta - delta_theta) ==0.0 ).all()
	avg_times_to_converge += counter
	print "converged with theta:"
	print theta
	print "Testing..."
	X_test  = np_data[train_size:,0:4]
	Y_test = np_data[train_size:, 4]
	result = h(theta, X_test.T)
	result = result > 0.5
	#print result == Y_test
	true_positive = 0
	false_positive = 0
	true_negative = 0
	false_negative = 0

	for i in range(Y_test.size):
		if Y_test[i]:
			if result[i]:
				true_positive +=1
			else:
				false_negative+=1
		else:
			if result[i]:
				false_positive +=1
			else:
				true_negative +=1

	Y_test = Y_test > 0.5
	accuracy = np.sum(Y_test == result) / (Y_test == result).size 
	print "accuracy= "+ str(accuracy)
	avg_times_to_converge //=iterations
	recall = true_positive / (true_positive + false_negative)
	precision = true_positive / (true_positive+ false_positive)
	f1_score = 2*(precision * recall) / (precision + recall)
	avg_accuracy += (accuracy/iterations)
	avg_recall += (recall/iterations)
	avg_precision += (precision/iterations)
	avg_f1score += (f1_score/iterations)
	print "recall= "+str(recall)
	print "precision= "+str(precision)
	print "F1-Score= "+str(f1_score)

print "Average Accuracy: "+str(avg_accuracy)
print "Average Recall: "+str(avg_recall)
print "Average Precision: "+str(avg_precision)
print "Average F1 Score: "+str(avg_f1score)
print "Average Count To Converge: "+str(avg_times_to_converge)
