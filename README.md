# Simple End-to-End Machine Learning project


A simple project that uses machine learning tools on one classification (discrete predictions into classes) and one regression (continuous value predictions) task. 

### Models used:

For classification I use: 
- Logistic regression 
- Linear discriminant analysis
- K Neighbours classifier
- Gaussian Naive Bayes' classifier
- Decision tree classifier
- Support vector classifier

Fo the regression task I use:
 - Linear regression
 - Ridge regression
 - Least Absolute Shrinkage and Selection Operator (LASSO)
 - ElasticNet regression
 - K neighbors regressor
 - Decision tree regressor 
 - Support vector regressor 
 

### **The steps followed in the project are**:
 
 - Data scaling (preprocessing)
 - Data analysis and visualisation
 - Spot check the models
 - Evaluate the models
 - Save and load models
 - Make and evaluate predictions
 

### Execution

Execution begins in 'final_iris.py'. However, the other files can be run individually to view the working of every step in the process. In `final_iris.py`, we conduct basic statistical analyses on the iris dataset and spot-check relevant classifiers from the ones listed above. Spot-checking here means cross-validating the model on the iris dataset. The top three models are selected for further evaluation, and the algorithm with the best results makes the final predictions on the dataset. Each model is fit onto the dataset and saved to file before being loaded and evaluated again. The sklearn_preprocessing file does not feature in the final_iris execution program. However, it demonstrates data scaling methods, which can be used to transform the input to fit different ML algorithms. 

### Datasets 

The Iris dataset is available in the sci-kit learn library - this is a classification dataset. In many of the supporting files I have used other datasets (also available in the sci-kit learn library). Amongst these are both regression as well as classification datasets. Overall I have used three datasets: 
- Boston housing dataset (regression) 
- Breast cancer dataset, and
- Wine dataset 

The mode of the algorithms can be toggled from classification to regression by changing the `task` variable in any file to 'c' or 'r' respectively. The iris dataset can be replaced by any other dataset in `final_iris.py`; however for large datasets the plots may take longer to render.
