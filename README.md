# Simple End-to-End Machine Learning project


This is a very basic project that helped me build and strengthen my understanding of various ML concepts. 

### The project uses the following **models**:

 - Logistic Regression
 - Linear Discriminant Analysis
 - Naive Bayes 
 - Linear Regression
 - Ridge Regression
 - Least Absolute Shrinkage and Selection Operator (LASSO)
 - ElasticNet Regression
 - Decision Tree 
 - K Nearest Neighbors 
 - Support Vector Machines 
 

### **The steps followed in this project are**:
 
 - Basic Analysis and Visualisation of data
 - Spot checking models
 - Model Evaluation
 - Save and Load a model
 - Make predictions and evaluate them
 
### Datasets 

The main dataset I have used is the Iris dataset which is available in the sci-kit learn library. This is a classification dataset. The main file is 'final_iris.py'. In addition to this all of the other files can also be executed separately to depict the working of every step in the process. In many of the supporting files I have used other datasets (also available in the sci-kit learn library). Amongst these are both regression as well as classification datasets. The regression dataset I have used is the boston_housing dataset. The classification datasets I have used are the breast_cancer and wine datasets. The mode of the algorithms can be toggled from Classification to Regression by changing the 'task' variable in the corresponding file to 'c' or 'r' respectively.
 
### Execution

The main algorithm here, conducts basic statistical analyses on the iris dataset and performs the spot-check with the applicable models listed above. The top three models are then selected for further evaluation and the algorithm with the best mean results (and lower standard deviation of results in case of a tie) is selected to make predictions on the dataset. The model is fit onto the dataset and saved to file before being loaded and evaluated again. 

### The list of all **modules** used in the project is as follows:
- pandas
- sklearn
- matplotlib
- operator (used for dictionary manipulations)
- pickle
 
I will keep updating parts of the project as and when applicable. Probable addition includes data preprocessing which is an important step in most Machine Learning algorithms! 