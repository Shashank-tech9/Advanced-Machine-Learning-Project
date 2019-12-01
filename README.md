# Advanced-Machine-Learning-Project
Final Project for the course "Advanced Machine Learning".
In this project we aim to classify spam messages from 2 datasets of emails, using Four algorithms: 
1) Naive Bayes
2) Muli level perceptron 
3) Support Vector Machine and 
4) Relevance Vector Machine
# Databases 
The fopllowing the are the databases which we used :
data_email
The following the are the libraries we used:
Sklearn
quadprog
# Jupyter Notebook code
```
import numpy as np
#from decision_tree import calculate_information_gain
#from decision_tree import decision_tree_train
#from decision_tree import decision_tree_predict
#from decision_tree import recursive_tree_train
from scipy.io import loadmat
from naive_bayes import naive_bayes_train, naive_bayes_predict
from load_all_data import *
from manipulate_data_nb import *
from manipulate_data_mlp import *
from manipulate_data_svm import *
from manipulate_data_rvm import *
from crossval import cross_validate
from mlp import mlp_train, mlp_predict, logistic, nll
from kernelsvm import kernel_svm_train, kernel_svm_predict
from skrvm import RVC
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import copy
```
We test 6 values of feature size for the 4 algorithms
```
feature_size = [25, 50, 75, 100, 125, 150]
#feature_size = [25]

nb_test_accuracy1 = [None] * len(feature_size)
nb_train_accuracy1 = [None] * len(feature_size)

mlp_test_accuracy1 = [None] * len(feature_size)
mlp_train_accuracy1 = [None] * len(feature_size)

svm_test_accuracy1 = [None] * len(feature_size)
svm_train_accuracy1 = [None] * len(feature_size)

rvm_test_accuracy1 = [None] * len(feature_size)
rvm_train_accuracy1 = [None] * len(feature_size)



X = [None] * len(feature_size)
y = [None] * len(feature_size)

for i in range(len(feature_size)):
    X[i], y[i] = load_all_data(feature_size[i])
```
