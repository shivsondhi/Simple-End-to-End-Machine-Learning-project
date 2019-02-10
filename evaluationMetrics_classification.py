'''
Notes:
Demonstrating 2 convenience metrics for a classification problem: Confusion Matrix | Classification Report
Using Logistic Regression model.
Works on a classification dataset only.
'''

import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
pd.set_option('display.expand_frame_repr', False)

def main():
	wines = datasets.load_wine()
	df = pd.DataFrame(wines.data, columns = wines.feature_names)
	y = pd.Series(wines.target)
	print("Dataset Sample: \n", df.sample(5), end="\n\n")
	print("Target Sample: \n", y.sample(10), end="\n\n")

	test_size = 0.33
	seed = 7

	X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = test_size, random_state = seed)
	model = LogisticRegression()
	model.fit(X_train, y_train)
	predicted = model.predict(X_test)
	class_eval(y_test, predicted)
	print("Evaluated!")

def class_eval(y_test, predicted):
	#Confusion Matrix
	matrix = confusion_matrix(y_test, predicted)								#structure of matrix defines actual along rows and predicted along columns.
	print("Confusion Matrix: \n", matrix, end="\n\n")							# 0 1 2 x 0 1 2 to give a 3x3 confusion matrix

	#Classification report
	report = classification_report(y_test, predicted)							#calculates precision | recall | f-1 score | support for each class.
	print("Classification Report: \n", report)
	return

if __name__ == "__main__":
	main()