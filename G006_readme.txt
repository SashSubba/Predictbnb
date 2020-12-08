Predictbnb

Airbnb price predictor, for COMP432 Machine Learning @ Concordia University.


Goal
The purpose of this project is to investigate various models to predict Airbnb listings in the Montreal area.
The investigated models are :

Linear Regression
Support Vector Regression
Random Forest Regression
Neural Network


Data
The dataset was taken from : http://insideairbnb.com/get-the-data.html


Setup
To run this project, a GPU is not required but Python3 and pip3 is required.
The following packages with minimum version level must be installed via pip3:

Scikit-learn 0.22.1
Pandas 1.0.1
Matplotlib 3.1
Scipy 1.4.1


Running the project
To run the project, simply run the following command from your terminal and within the repository:

python main.py

or

python3 main.py

depending on how you set up your path variable name for python3.

Note : It is not necessary to run preprocessing.py before running main.py, because it is automatically imported into main.py.


Sources
For preprocessing and training the models, we relied on the following documentations for each Model:

Linear Regression
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

Support Vector Regression
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR

Random Forest Regression
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
https://scikit-learn.org/stable/modules/ensemble.html
https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters
https://medium.com/datadriveninvestor/random-forest-regression-9871bc9a25eb

Neural Networks
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html