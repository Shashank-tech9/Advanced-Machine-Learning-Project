import numpy as np
#import re
#import nltk
from sklearn.datasets import load_files
#import pickle
#from nltk.corpus import stopwords
import collections

def manipulate_data_mlp(X, y):
    #from sklearn.feature_extraction.text import TfidfTransformer
    #tfidfconverter = TfidfTransformer()
    #X = tfidfconverter.fit_transform(X).toarray()

    X = np.array(X)

    X = np.transpose(X)
    
    #print("X.shape", X.shape)
    #print("y.shape", y.shape)
    
    #y = np.invert(y)
    
    #np_ones = np.ones({y.shape});
    
    #y = (y*2) - (np_ones);

    for i in range(y.shape[0]):
        if(y[i]<=0):
            y[i] = -1;
        else:
            y[i] = 1;

    
    #collections_y = collections.Counter(y)
    
    #print("collections_y", collections_y)

    train_count = int(len(y)*0.667);

    train_data = X[:, 0:train_count]
    train_labels = y[0:train_count]

    test_data = X[:,(train_count+1):]
    test_labels = y[(train_count+1):]

    num_words = X.shape[1]
    num_training = train_labels.shape[0]
    num_testing = test_labels.shape[0]

    #print("train_data_shape", train_data.shape);
    #print("test_data_shape", test_data.shape);

    #print("train_labels_shape", train_labels.shape);
    #print("test_labels_shape", test_labels.shape);

    #print("num_words", num_words);
    #print("num_testing", num_testing);
    #print("num_training", num_training);

    return num_words, num_training, num_testing, train_data, test_data, train_labels, test_labels
