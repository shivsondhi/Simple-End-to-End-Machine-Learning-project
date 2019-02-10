'''
Analyse and visualise data.				OK
Preprocess if necessary.				NO
Spot check algorithms.					OK
Compare shortlisted algorithms.			OK
Finetune hyperparameters.				NO
Save and Load + Predict.				OK
'''

import pickle as pkl
import pandas as pd 
from pandas.plotting import scatter_matrix
from sklearn import datasets
from utilities import shuffle_data
import matplotlib.pyplot as plt 
from spotChecking import spot_check 
from modelComparison import compare_models
from predict_and_save import split_and_save
from evaluationMetrics_classification import class_eval 
pd.set_option('display.expand_frame_repr', False)

def main():
	#load and print iris dataset (shuffled)
	i = datasets.load_iris()
	df = pd.DataFrame(i.data, columns=i.feature_names)
	s = pd.Series(i.target)
	df, s = shuffle_data(df, s)
	print("Dataset:\n", df.head(), end="\n\n")
	print("Targets:\n", s.head(), end="\n\n")

	#Define task for the dataset as 'c' or 'r'
	task = 'c'																		#'c' for Classification and 'r' for regression

	#print some basic statistics: Mean, Std, Quartile Ranges, Count etc. 
	print("Mean of each attribute:\n", df.mean(), end="\n\n")
	print("Standard Deviation of each attribute\n", df.std(), end="\n\n")
	print("Data summary:\n", df.describe(), end="\n\n")

	#print the correlation metrics and plots of the data
	df['target'] = s
	print("Correlation between attributes:\n", df.corr(), end="\n\n")
	print("Covariance between attributes:\n", df.cov(), end="\n\n")
	df.plot(kind='box', figsize=(12,6))												#Box Plot
	plt.title("Box Plot of every attribute")
	plt.show()
	del df['target']
	scatter = scatter_matrix(df, alpha=0.2)											#Scatter Plot
	plt.suptitle("Scatter plot of every attribute pair")
	plt.show()

	#Spot check the classification algorithms on the dataset 
	seed = 7
	splits = 5
	print("Spot checking algorithms: ")
	res = spot_check(task, seed, splits, df, s)
	print("Successful spot-check!", end="\n\n")
	print(res)

	#Compare the qualified models
	finals = compare_models(task, res, seed, splits, df, s)
	if len(finals) > 1:
		fin = finals[0]
		print("Compared! Selected model is: {0}".format(fin), end="\n\n")
	else:
		fin = finals[0]
		print("Compared! Best model is: {0}".format(fin), end="\n\n")

	#Split into train and test sets and save the model to file
	test_size = 0.2
	filename = "iris_model-2.sav"
	X_test, Y_test = split_and_save(test_size, fin, filename, seed, df, s)

	#Load model and calculate accuracy
	with open(filename, 'rb') as f:
		loaded_model = pkl.load(f)
	print("Model loaded!")
	result = loaded_model.score(X_test, Y_test)
	print("Accuracy is %.3f%%" % (result * 100), end="\n\n")

	if task in ('c', 'C'):
		#Confusion matrix and classification report on predicted test values
		predicted = loaded_model.predict(X_test)
		class_eval(Y_test, predicted)
		print("Evaluated!")

if __name__ == "__main__":
	main()