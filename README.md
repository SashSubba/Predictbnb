<h1> Predictbnb </h1>
<p>
Airbnb price predictor, for COMP432 Machine Learning @ Concordia University.
</p>

<h2> Goal </h2>
<p>
The purpose of this project is to investigate various models to predict Airbnb listings in the Montreal area.<br>
The investigated models are :
<ul>
  <li>Linear Regression</li>
  <li>Support Vector Regression</li>
  <li>Random Forest Regression</li>
  <li>Neural Netwrok</li>
</ul>
</p>


<h2>Data</h2>
<p>
The dataset was taken from : http://insideairbnb.com/get-the-data.html 
</p>

<h2>Setup</h2>
<p>
  To run this project, a GPU is not required but Python3 and pip3 is required.<br>
  The following packages with minimum version level must be installed via pip3:
  
<ul>
  <li>Scikit-learn 0.22.1</li>
  <li>Pandas 1.0.1</li>
  <li>Matplotlib 3.1</li>
  <li>Scipy 1.4.1</li>
</ul>

</p>

<h2>Running the project</h2>
<p>
To run the project, simply run the following command from your terminal and within the repository:
  
`python main.py`

or 

`python3 main.py`

depending on how you set up your path variable name for python3.

<b>Note</b> : It is not necessary to run `preprocessing.py` before running `main.py`, because it is automatically imported into `main.py`.
  
</p>
<p></p>


<h2>Sources</h2>

<p>
For preprocessing and training the models, we relied on the following documentations for each Model:
  

Random Forest Regression
<ul>
  <li>https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor</li>
  <li>https://scikit-learn.org/stable/modules/ensemble.html</li>
  <li>https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters</li>
  <li>https://medium.com/datadriveninvestor/random-forest-regression-9871bc9a25eb</li>
</ul>

Neural Networks
<ul>
<li>https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html</li>
</ul>


</p>
