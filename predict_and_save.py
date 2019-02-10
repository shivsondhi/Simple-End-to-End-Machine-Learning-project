'''
Save model using pickle.
Then load the model and calculate accuracy of the predictions.
Uses the LDA model and the breast cancer dataset from the sci-kit learn library (Classification problem).
'''

import pandas as pd 
import pickle as pkl 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from utilities import shuffle_data
pd.set_option('display.expand_frame_repr', False)

def main():
	bc = datasets.load_breast_cancer()
	df = pd.DataFrame(bc.data, columns=bc.feature_names)
	s = pd.Series(bc.target)
	df, s = shuffle_data(df, s)
	print("First 9 columns of dataset:\n", df[df.columns[range(9)]].head(), "\n\nTargets of the dataset:\n", s.head(), end="\n\n")
	
	seed = 7
	test_size = 0.2
	model = 'LDA'
	filename = 'LDA_model-1.sav'
	X_test, Y_test = split_and_save(test_size, model, filename, seed, df, s)

	#Load model and predict
	loaded_model = pkl.load(open(filename, 'rb'))
	print("Model loaded!")
	result = loaded_model.score(X_test, Y_test)
	print("Accuracy is %.3f%%" % (result * 100))

def split_and_save(test_size, model, filename, seed, df, s):
	#Create model dictionary
	models = {"Lor": LogisticRegression(), 
			  "LDA": LinearDiscriminantAnalysis(), 
			  "KNNC": KNeighborsClassifier(), 
			  "NBayes": GaussianNB(), 
			  "CARTC": DecisionTreeClassifier(), 
			  "SVC": SVC(), 
			  "Lir": LinearRegression(), 
			  "RR": Ridge(), 
			  "LASSO": Lasso(), 
			  "ENet": ElasticNet(), 
			  "KNNR": KNeighborsRegressor(),
			  "CARTR": DecisionTreeRegressor(),
			  "SVR": SVR()
			 }

	#Split data
	X_train, X_test, Y_train, Y_test = train_test_split(df, s, test_size=test_size, random_state=seed)

	#Select and train model
	filename = filename
	model = models[model]
	model.fit(X_train, Y_train)
	#Save model
	pkl.dump(model, open(filename, 'wb'))
	print("Model saved to file succesfully!", end="\n\n")
	return X_test, Y_test

if __name__ == "__main__":
	main()