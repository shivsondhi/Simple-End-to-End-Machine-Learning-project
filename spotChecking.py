'''
Spot Checking models for a classification problem.
It is a method to check what algorithm will work well for a given problem and the corresponding dataset. It is only the first step though!


Toggle the value in variable task to check classification (c) and regression (r) models. 
'''

import pandas as pd 
import operator as op 
from sklearn import datasets, linear_model, discriminant_analysis, svm, tree, neighbors, naive_bayes
from sklearn.model_selection import KFold, cross_val_score
pd.set_option('display.expand_frame_repr', False)

def main():
	task = 'r'
	if task in ('c', 'C'):
		breast_cancer = datasets.load_breast_cancer()
		df = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
		y = pd.Series(breast_cancer.target)
	elif task in ('r', 'R'):
		boston = datasets.load_boston()
		df = pd.DataFrame(boston.data, columns=boston.feature_names)
		y = pd.Series(boston.target)
	
	seed = 7
	splits = 10
	results = spot_check(task, seed, splits, df, y)
	print("Successful spot check!\n\nTop three algorithms are-\n{0} {1} and {2}".format(results[0], results[1], results[2]))

def spot_check(task, seed, splits, df, y):
	results = {}
	seed = seed
	kfold = KFold(n_splits = splits, random_state = seed)

	if task in ('c', 'C'):
		print("Spot Checking Classification Algorithms: ")
		#Classification Models!
		#LINEAR Classification models:
		#Logistic Regression 
		model = linear_model.LogisticRegression()
		result = cross_val_score(model, df, y, cv=kfold)
		results['LoR'] = result.mean()
		print("Linear-\nLogistic Regression: ", results['LoR'])
		#LDA
		model = discriminant_analysis.LinearDiscriminantAnalysis()
		result = cross_val_score(model, df, y, cv=kfold)
		results['LDA'] = result.mean()
		print("LDA score: ", results['LDA'], end="\n\n")
		#NON-LINEAR Classification models:
		#KNN
		model = neighbors.KNeighborsClassifier()									#Careful of the spelling of Neighbors
		result = cross_val_score(model, df, y, cv = kfold)
		results['KNNC'] = result.mean()
		print("Non-Linear-\nKNN: ", results['KNNC'])
		#Naive Bayes
		model = naive_bayes.GaussianNB()
		result = cross_val_score(model, df, y, cv=kfold)
		results['NBayes'] = result.mean()
		print("Naive Bayes: ", results['NBayes'])
		#Classification and Regression Trees / decision trees
		model = tree.DecisionTreeClassifier()
		result = cross_val_score(model, df, y, cv=kfold)
		results['CARTC'] = result.mean()
		print("CART: ", results['CARTC'])
		#Support Vector Machines
		model = svm.SVC()
		result = cross_val_score(model, df, y, cv=kfold)
		results['SVC'] = result.mean()
		print("Support Vector Machine: ", results['SVC'])
	elif task in ('r', 'R'):
		print("Spot Checking Regression Algorithms: ")
		scoring='neg_mean_squared_error'
		#Regression Models
		#LINEAR Regression Models
		#Linear Regression
		model = linear_model.LinearRegression()
		result = cross_val_score(model, df, y, cv=kfold, scoring=scoring)
		results['LiR'] = result.mean()
		print("Linear-\nLinear Regression: ", results['LiR'])
		#Ridge Regression (L2 norm)
		model = linear_model.Ridge()
		result = cross_val_score(model, df, y, cv=kfold, scoring=scoring)
		results['RR'] = result.mean()
		print("Ridge Regression: ", results['RR'])
		#Least Absolute Shrinkage and Selection Operator (L1 norm)
		model = linear_model.Lasso()
		result = cross_val_score(model, df, y, cv=kfold, scoring=scoring)
		results['LASSO'] = result.mean()		
		print("LASSO: ", results['LASSO'])
		#ElasticNet Regression (L1 and L2 norm)
		model = linear_model.ElasticNet()
		result = cross_val_score(model, df, y, cv=kfold, scoring=scoring)
		results['ENet'] = result.mean()		
		print("ElasticNet Regression: ", results['ENet'])
		#NON LINEAR Regression models
		#K-Nearest Neighbours
		model = neighbors.KNeighborsRegressor() 
		result = cross_val_score(model, df, y, cv=kfold, scoring=scoring)
		results['KNNR'] = result.mean()		
		print("Non-Linear-\nKNN: ", results['KNNR'])
		#Classification and Regression Trees
		model = tree.DecisionTreeRegressor()
		result = cross_val_score(model, df, y, cv=kfold, scoring=scoring)
		results['CARTR'] = result.mean()		
		print("CART: ", results['CARTR'])
		#Support Vector Machine
		model = svm.SVR()
		result = cross_val_score(model, df, y, cv=kfold, scoring=scoring)
		results['SVR'] = result.mean()		
		print("Support Vector Machine: ", results['SVR'])
	else:
		print("Invalid task definition (r/c)!")

	#Select top three in spot checked algorithms
	res = []
	for i in range(3):
		res.append(max(results.items(), key=op.itemgetter(1))[0])
		del results[res[i]]
	return res

if __name__ == "__main__":
	main()