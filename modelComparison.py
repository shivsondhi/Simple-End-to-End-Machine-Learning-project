'''
Model Comparison is Step 2 in the process of selecting a model. 
Once you have spot checked a lot of possible models and selected a few that are worth considering, use model comparison to choose the best model!
Here, I compare three models each on a classification and a regression dataset.


Toggle the value in variable task to check classification (c) and regression (r) models. 
'''

import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.utils import shuffle as sh 
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from utilities import shuffle_data
pd.set_option('display.expand_frame_repr', False)

def main():
	task = 'c' 
	if task in ('c', 'C'):
		bc = datasets.load_breast_cancer()
		selected = ["LDA", "KNNC", "NBayes"]
	elif task in ('r', 'R'):
		bc = datasets.load_boston()
		selected = ["RR", "ENet", "LASSO"]

	df = pd.DataFrame(bc.data, columns=bc.feature_names)
	s = pd.Series(bc.target)
	df, s = shuffle_data(df, s)														#shuffle to mix up classes
	print("First 9 columns of dataset:\n", df[df.columns[range(9)]].head(), "\n\nTargets of the dataset:\n", s.head(), end="\n\n")
	seed = 7
	splits = 10
	compare_models(task, selected, seed, splits, df, s)
	print("Completed model comparison!")


def compare_models(task, selected, seed, splits, df, s):
	seed = seed
	
	#create list of all models to compare --> list of tuples containing name,model()
	models = []
	if task in ('c', 'C'):
		models.append(("LoR", LogisticRegression()))
		models.append(("LDA", LinearDiscriminantAnalysis()))
		models.append(("KNNC", KNeighborsClassifier()))
		models.append(("NBayes", GaussianNB()))
		models.append(("CARTC", DecisionTreeClassifier()))
		models.append(("SVC", SVC()))
		scoring = 'accuracy'
	elif task in ('r', 'R'):	
		models.append(("LiR", LinearRegression()))
		models.append(("RR", Ridge()))
		models.append(("LASSO", Lasso()))
		models.append(("ENet", ElasticNet()))
		models.append(("KNNR", KNeighborsRegressor()))
		models.append(("CARTR", DecisionTreeRegressor()))
		models.append("SVR", SVR())
		scoring = "neg_mean_squared_error"

	#Obtain results of cross_val for each model and store it in a list called results
	results = []
	names = []
	means = []
	spread = {}	
	for name, model in models:
		if name in selected:
			kfold = KFold(n_splits=splits, random_state=seed)
			result = cross_val_score(model, df, s, cv=kfold, scoring=scoring)
			results.append(result)																	#For plot
			names.append(name)																		#For plot and to find highest mean
			means.append(result.mean())																#To find highest mean
			spread[name] = result.mean() - result.std()												#To record lowest 2-sigma-spread bound
			print("%s: %.3f (%.3f)" % (name, result.mean(), result.std()))

	fig = plt.figure()
	fig.suptitle("Algorithm Comparison")
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()

	max_mean = max(means)
	ind = [i for i, j in enumerate(means) if j==max_mean]
	res = [names[i] for i in ind]
	if len(res) > 1:
		fin = [name for name, val in spread.items() if (name in res) and val == max(spread.values())]
		return fin
	return res

if __name__ == "__main__":
	main()
