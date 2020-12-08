import preprocessing as prep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network
import sklearn.svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

#Preprocessed data retrieval
X = prep.X_pd
y = prep.y_pd

X_np = X.to_numpy()
y_np = y.to_numpy()
    
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_np, y_np, test_size=0.30, random_state=0)

#Run Feature Selection
fs = SelectKBest(score_func=mutual_info_regression, k="all")
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)

for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

#Linear Regression Model
linear_model = sklearn.linear_model.LinearRegression()
linear_model.fit(X_train_fs,y_train)
lin_preds = linear_model.predict(X_train_fs)

print('Training Mean Squared Error for Linear Regression: %.2f' % mean_squared_error(y_train, lin_preds))
print('Training R-Squared Score for Linear Regression: %.3f' % linear_model.score(X_train_fs,y_train))

linear_model2 = sklearn.linear_model.LinearRegression()
linear_model2.fit(X_train_fs,y_train)
lin_preds2 = linear_model2.predict(X_test_fs)

print('Testing Mean Squared Error for Linear Regression: %.2f' % mean_squared_error(y_test, lin_preds2))
print('Testing R-Squared Score for Linear Regression: %.3f' % linear_model2.score(X_test_fs,y_test))

#Support Vector Regression Model . Note : Training takes about 2 mins. Please see comments after print statements for the results
svr_model = sklearn.svm.LinearSVR(random_state=0)
param_distribution = {'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'], 'dual': [True,False], 'max_iter': [100,500,1000,2000]}
randomized_search_svr = sklearn.model_selection.RandomizedSearchCV(svr_model, param_distribution, n_iter=22, verbose=2, cv=5, random_state=0)
randomized_search_svr.fit(X_train_fs, y_train)

best_estimator = randomized_search_svr.best_estimator_

y_pred_1 = best_estimator.predict(X_train_fs)
y_pred_2 = best_estimator.predict(X_test_fs)

training_score = best_estimator.score(X_train_fs, y_train)
testing_score = best_estimator.score(X_test_fs, y_test)

mse_train = sklearn.metrics.mean_squared_error(y_train, y_pred_1)
mse_test = sklearn.metrics.mean_squared_error(y_test, y_pred_2)

print("\nrandomized search best estimator")
print("best_params_: %s" % randomized_search_svr.best_params_)
print("Training Mean Squared Error: %.3f"  %(mse_train))
print("Training R-Squared Error: %.3f"  %(training_score))
print("Testing Mean Squared Error: %.3f"  %(mse_test))
print("Testing R-Squared Error: %.3f"  %(testing_score))

"""

best estimator
best_params_: {'loss': squared_epsilon_insensitive, 'dual': False, 'max_iter': 100}
Training Mean Squared Error: 4014.688
Training R-Squared Error: 0.315
Testing Mean Squared Error: 4002.641
Testing R-Squared Error: 0.336

"""

#RandomizedSearch with RandomForestRegressor . Note : Training takes about 7.5 mins. Please see comments after print statements for the results
rfr = sklearn.ensemble.RandomForestRegressor( random_state=0)
param_distribution = {'n_estimators': [50,100,200,300], 'max_features': ['auto','sqrt'], 'max_depth':[10,20,30,40,50]}
randomized_search_rfr = sklearn.model_selection.RandomizedSearchCV(rfr, param_distribution, n_iter=22, verbose=2, cv=5, random_state=0)
randomized_search_rfr.fit(X_train_fs, y_train)

best_estimator = randomized_search_rfr.best_estimator_

y_pred_1 = best_estimator.predict(X_train_fs)
y_pred_2 = best_estimator.predict(X_test_fs)

training_score = best_estimator.score(X_train_fs, y_train)
testing_score = best_estimator.score(X_test_fs, y_test)

mse_train = sklearn.metrics.mean_squared_error(y_train, y_pred_1)
mse_test = sklearn.metrics.mean_squared_error(y_test, y_pred_2)

print("\nrandomized search best estimator")
print("best_params_: %s" % randomized_search_rfr.best_params_)
print("Training Mean Squared Error: %.3f"  %(mse_train))
print("Training R-Squared Error: %.3f"  %(training_score))
print("Testing Mean Squared Error: %.3f"  %(mse_test))
print("Testing R-Squared Error: %.3f"  %(testing_score))

"""

best estimator
best_params_: {'n_estimators': 300, 'max_features': 'sqrt', 'max_depth': 50}
Training Mean Squared Error: 437.618
Training R-Squared Error: 0.925
Testing Mean Squared Error: 3125.089
Testing R-Squared Error: 0.481

"""

#RandomizedSearch with Multi-layer Perceptron regressor. Note : Training takes about 4 mins. Please see comments after print statements for the results

nn = sklearn.neural_network.MLPRegressor( random_state=0, momentum=0.9)
param_neural_net = {'solver': ['sgd','adam'], 'hidden_layer_sizes':[(),(100,),(100,57),(57,25),(100,57,25)], 'activation':['tanh','sgd'], 'batch_size':[100,200,300], 'max_iter':[10,50,100,200,500],'learning_rate_init':[0.001, 0.01, 0.1]}
randomized_search_nn = sklearn.model_selection.RandomizedSearchCV(nn, param_neural_net, n_iter=40, verbose=2, cv=5, random_state=0)
randomized_search_nn.fit(X_train_fs, y_train)

best_estimator = randomized_search_nn.best_estimator_

y_pred_1 = best_estimator.predict(X_train_fs)
y_pred_2 = best_estimator.predict(X_test_fs)

training_score = best_estimator.score(X_train_fs, y_train)
testing_score = best_estimator.score(X_test_fs, y_test)

mse_train = sklearn.metrics.mean_squared_error(y_train, y_pred_1)
mse_test = sklearn.metrics.mean_squared_error(y_test, y_pred_2)

print("\n\n neural net search best estimator")
print("best_params_: %s" % randomized_search_nn.best_params_)
print("Training Mean Squared Error: %.3f"  %(mse_train))
print("Training R-Squared Error: %.3f"  %(training_score))
print("Testing Mean Squared Error: %.3f"  %(mse_test))
print("Testing R-Squared Error: %.3f"  %(testing_score))

"""

best estimator
best_params_: {'solver': 'sgd', 'max_iter': 10, 'learning_rate_init': 0.001, 'hidden_layer_sizes': (100, 57), 'batch_size': 100, 'activation': 'tanh'}
Training Mean Squared Error: 3414.409
Training R-Squared Error: 0.418
Testing Mean Squared Error: 3673.520
Testing R-Squared Error: 0.390

"""

#Plotting 
plt.scatter(y_test, y_pred_2,  color='black')
plt.plot(y_pred_2, y_pred_2, color='blue', linewidth=3)
plt.title('Price predictions vs Actual Price for Best Estimator')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()