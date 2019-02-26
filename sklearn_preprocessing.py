'''
Here, I use four common data scaling techniques, provided in the sci-kit learn module for preprocessing data. These are:
	MinMaxScaler	- Scales every data sample within a specified range.
	StandardScaler	- Scales every sample in such a way to ensure that the mean=0 and std=1. This is done by subtracting the mean from each sample and dividing the result by std.
	Normalizer		- L1 norm finds the absolute difference between each sample and the mean. L2 norm finds the square of the same difference.
	Binarizer		- Each data sample is either represented as 0 or 1 depending on a specified threshold value thata lies between 0 and 1.
Each data sample is thus transformed differently in each of the four cases and the dataset is scaled accordingly.

Running the file, returns the dataset scaled to a randomly selected scaling method. If you want to see the data scaled in all methods, assign scale_range the value 'all' where it is
passed to the scale_data function.
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
import pandas as pd 
import scipy as sp 
import numpy as np
import random 
pd.set_option('display.expand_frame_repr', False)								#To show a full view of columns as opposed to summary view

def main():
	#Set the parameters as desired
	scale_types = ['range', 'std', 'norm', 'binary']
	scale_type = random.choice(scale_types)
	scale_range = (0,1)															#For the MinMaxScaler()
	norm_type = 'l2'															#L1 or L2 norm
	threshold = 0.6																#Between 0 and 1

	url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
	names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = pd.read_csv(url, delimiter = ',', names = names)
	print(df.head(), end="\n\n")

	array = df.values															#Each row as a list and the dataframe as a list of lists
	X = array[:,:8]																#input variables
	Y = array[:,8]																#output variables
	X = scale_data(X, scale_type, scale_range=scale_range, norm_type=norm_type, threshold=threshold)
	if X is not None:
		np.set_printoptions(precision = 3)
		print(X[0:5,:])															#0-5 in the first dimension(rows) and all in the second dimesnion(cols)
	
def scale_data(X, scale_type, scale_range=(None,None), norm_type=None, threshold=None):
	if scale_type == 'range':
		transform_type = MinMaxScaler(feature_range=scale_range)				#range can be a desired range
		X = fit_and_trans(transform_type, X)
		print(transform_type)
		return X
	elif scale_type == 'std':
		transform_type = StandardScaler()										#Mean = 0, std = 1 | Normally distributed around 0
		X = fit_and_trans(transform_type, X)
		print(transform_type)
		return X
	elif scale_type == 'norm':
		transform_type = Normalizer(norm = norm_type)							#norm can be l1, l2 or max
		X = fit_and_trans(transform_type, X)
		print(transform_type)
		return X
	elif scale_type == 'binary':
		transform_type = Binarizer(threshold = threshold)						#set a threshold
		X = fit_and_trans(transform_type, X)
		print(transform_type)
		return X
	elif scale_type == 'all':
		np.set_printoptions(precision = 3)										#To get upto 3 decimal digits
		#MinMaxScaler
		transform_type = MinMaxScaler(feature_range=scale_range)
		X = fit_and_trans(transform_type, X)
		print(transform_type)
		print(X[0:5,:], end="\n\n")
		#Standard Scaler
		transform_type = StandardScaler()
		X = fit_and_trans(transform_type, X)
		print(transform_type)
		print(X[0:5,:], end="\n\n")
		#Normalizer
		transform_type = Normalizer(norm = norm_type)
		X = fit_and_trans(transform_type, X)
		print(transform_type)
		print(X[0:5,:], end="\n\n")
		#Binarizer
		transform_type = Binarizer(threshold = threshold)
		X = fit_and_trans(transform_type, X)
		print(transform_type)
		print(X[0:5,:])
		return None

def fit_and_trans(transform_type, X):
	transformer = transform_type.fit(X)
	transformedX = transformer.transform(X)
	return transformedX

if __name__ == "__main__":
	main()