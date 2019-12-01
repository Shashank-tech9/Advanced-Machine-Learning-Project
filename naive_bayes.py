"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np
import math as math


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include an 'alpha' value
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """
    # TODO: INSERT YOUR CODE HERE TO LEARN THE PARAMETERS FOR NAIVE BAYES

    alpha = params['alpha']
    #alpha = 1e-5

    labels = np.unique(train_labels)

    d, n = train_data.shape
    num_classes = labels.size

    cond_prb = np.ones(shape =(num_classes, d+1))

    for i in range(0,num_classes):
        current_label = labels[i]
        num_current_label = np.sum(train_labels == labels[i])
        current_label_prb = (num_current_label + alpha) / (n + num_classes*alpha)
        cond_prb[i, 0] = current_label_prb

    for i in range(0,num_classes):
        num_y = np.sum(train_labels == labels[i])
        data_with_label = train_data[:, train_labels == labels[i]]
        num_y_and_x = data_with_label.dot(np.ones(data_with_label.shape[1]))

        cond_prb[i,1:] = (num_y_and_x + alpha) / (num_y + (2*alpha))

    model = cond_prb

    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE HERE FOR USING THE LEARNED NAIVE BAYES PARAMETERS
    # TO CLASSIFY THE DATA

    d,n = data.shape
    prediction = np.ones(n)
    num_classes = model.shape[0]

    cond_prb = model[:,1:];
    not_cond_prb = 1-cond_prb

    cond_prb = np.log(cond_prb)
    not_cond_prb = np.log(not_cond_prb)

    for i in range(0,n):
        feature_i = data[:,i]
        t1 = np.dot(cond_prb, feature_i)
        t2 = np.dot(not_cond_prb, 1-feature_i)
        t3 = (t1+t2)+np.log(model[:,0])
        prediction[i] = t3.argmax()

    return prediction

