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
Accuracy Test for Naive-Bayes
```
nb_params = { 'alpha': 1.0 }

for i in range(len(feature_size)):
    num_words_nb, num_training_nb, num_testing_nb, train_data_nb, test_data_nb, train_labels_nb, test_labels_nb = manipulate_data_nb(copy.deepcopy(X[i]), copy.deepcopy(y[i]))
    
    #print("test_labels", test_labels_nb);
    #print("teat_data", test_data_nb[0:20,0:20])                                                                                                                                 
    
    nb_model = naive_bayes_train(train_data_nb, train_labels_nb, nb_params)
    
    print("Feature size: %d" % feature_size[i])
    
    nb_train_predictions = naive_bayes_predict(train_data_nb, nb_model)
    nb_train_accuracy = np.mean(nb_train_predictions == train_labels_nb)
    print("Naive Bayes training accuracy: %f" % nb_train_accuracy)
    
    nb_test_predictions = naive_bayes_predict(test_data_nb, nb_model)
    nb_test_accuracy = np.mean(nb_test_predictions == test_labels_nb)
    print("Naive Bayes testing accuracy: %f" % nb_test_accuracy)
    
    nb_test_accuracy1[i] = nb_test_accuracy
    nb_train_accuracy1[i] = nb_train_accuracy
```
Accuracy test for Multi level Perceptron
```
num_folds = 4
structures = [[3]]
lambda_vals = [0.01, 0.1]
params = {
    'max_iter': 10000,
    #'max_iter': 100,
    'squash_function': logistic,
    'loss_function': nll
}
    
best_params = []
best_score = 0
for i in range(len(feature_size)):
    num_words_mlp, num_training_mlp, num_testing_mlp, train_data_mlp, test_data_mlp, train_labels_mlp, test_labels_mlp = manipulate_data_mlp(copy.deepcopy(X[i]), copy.deepcopy(y[i]))
    #print("test_labels", test_labels_mlp[0:20]);
    #print("teat_data", test_data_mlp[0:20,0:20]) 
            
    for j in range(len(structures)):
        for k in range(len(lambda_vals)):
            params['num_hidden_units']= structures[j]
            params['lambda'] = lambda_vals[k]
            #print("lambda", lambda_vals[k]);
            #print("structure", structures[j]);
            
            cv_score, models = cross_validate(mlp_train, mlp_predict, train_data_mlp, train_labels_mlp, num_folds, params)
        
            #print("cv_score", cv_score);
        
            if cv_score > best_score:
                best_score = cv_score
                best_params = copy.copy(params)
                
    print("Feature size: %d" % feature_size[i])
               
    mlp_model = mlp_train(train_data_mlp, train_labels_mlp, best_params)
    predictions, _, _, _ = mlp_predict(test_data_mlp, mlp_model)
    test_accuracy = np.mean(predictions == test_labels_mlp)
    
    print("MLP had test accuracy %f" % (test_accuracy))
    
    predictions, _, _, _ = mlp_predict(train_data_mlp, mlp_model)
    train_accuracy = np.mean(predictions == train_labels_mlp)
    print("MLP had train accuracy %f" % (train_accuracy))
    print("with structure %s and lambda = %f" % (repr(best_params['num_hidden_units']), best_params['lambda']))
    
    mlp_test_accuracy1[i] = test_accuracy
    mlp_train_accuracy1[i] = train_accuracy
 ```
 Accuracy test for Support Vector Machine
 ```
 num_folds = 4
c_vals = 10 ** np.linspace(-1, 3, 4)
sigmas = np.linspace(0.1, 7, 15)
#sigmas = [4.5]
best_params_svm = {
                    'kernel': 'rbf',
                    'C': c_vals[0],
                    'sigma': sigmas[0]
                  }
best_score = 0

for i in range(len(feature_size)):
    num_words_svm, num_training_svm, num_testing_svm, train_data_svm, test_data_svm, train_labels_svm, test_labels_svm = manipulate_data_svm(copy.deepcopy(X[i]), copy.deepcopy(y[i]))

    #print("test_labels", test_labels_svm[0:20]);
    #print("teat_data", test_data_svm[0:20,0:20]);                                                                                                                                          

    for j in range(len(c_vals)):
        for k in range(len(sigmas)):
            params = {
                'kernel': 'rbf',
                'C': c_vals[j],
                'sigma': sigmas[k]
            }
            
            cv_score, _ = cross_validate(kernel_svm_train, kernel_svm_predict, train_data_svm, train_labels_svm, num_folds, params)
        
            print("cv_score", cv_score);
            #print(j, " | ", k);
        
            if cv_score > best_score:
                best_score = cv_score
                best_params_svm['kernel'] = params['kernel']
                best_params_svm['C'] = params['C']
                best_params_svm['sigma'] = params['sigma']
    
    print("Feature size: %d" % feature_size[i])
                
    rbf_svm_model = kernel_svm_train(train_data_svm, train_labels_svm, best_params_svm)
    predictions, _ = kernel_svm_predict(test_data_svm, rbf_svm_model)
    test_accuracy = np.mean(predictions == test_labels_svm)
    
    print("RBF SVM had test accuracy %f" % (test_accuracy))
    
    predictions, _ = kernel_svm_predict(train_data_svm, rbf_svm_model)
    train_accuracy = np.mean(predictions == train_labels_svm)
    
    print("RBF SVM had train accuracy %f" % (train_accuracy))
    print("with C = %f, sigma = %f" % (best_params_svm['C'], best_params_svm['sigma']))
    
    svm_test_accuracy1[i] = test_accuracy
    svm_train_accuracy1[i] = train_accuracy
```
Accuracy Test for RVM
```
for i in range(len(feature_size)):
    #print("X[i]", X[i].shape)
    #print("Y[i]", y[i].shape)
    num_words_rvm, num_training_rvm, num_testing_rvm, train_data_rvm, test_data_rvm, train_labels_rvm, test_labels_rvm = manipulate_data_rvm(copy.deepcopy(X[i]), copy.deepcopy(y[i]))
    #print("train_data_rvm", train_data_rvm.shape)
    #print("train_label_rvm", train_labels_rvm.shape)
    clf = RVC(kernel='rbf', n_iter=10, n_iter_posterior=10, threshold_alpha=10000.0, verbose=False)
    clf.fit(train_data_rvm, train_labels_rvm)
    test_accuracy = clf.score(test_data_rvm, test_labels_rvm)
    print("Feature size: %d" % feature_size[i])
    print("RBF RVM had test accuracy %f" % (test_accuracy))
    train_accuracy = clf.score(train_data_rvm, train_labels_rvm)
    print("RBF RVM had train accuracy %f" % (train_accuracy))
    
    rvm_test_accuracy1[i] = test_accuracy
    rvm_train_accuracy1[i] = train_accuracy
  ```
