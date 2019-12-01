import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

def load_all_data(num_feature):

    movie_data = load_files(r"./data_email/email_train")
    X, y = movie_data.data, movie_data.target
    
    #print("X", len(X));
    #print("y", y.shape);
    #print("y", y[0:20]);
    #print("X[0] ", X[0]);
    
    documents = []
    
    #nltk.download('wordnet')
    #from nltk.stem import WordNetLemmatizer
    
    stemmer = WordNetLemmatizer()
    
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W+', ' ', str(X[sen]))
        
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        
        # Converting to Lowercase
        document = document.lower()
        
        # Lemmatization
        document = document.split()
    
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        
        documents.append(document)
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    #print("document[0] ", documents[0])
    
    vectorizer = CountVectorizer(max_features=num_feature, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(documents).toarray()
    
    #print("X", len(X));
    #print("X[0]", len(X[0]));
    #print("X[0]", X[0])
    #print("vectorizer", vectorizer)
    
    #from sklearn.feature_extraction.text import TfidfTransformer
    #tfidfconverter = TfidfTransformer()
    #X = tfidfconverter.fit_transform(X).toarray()
    
    #X = np.array(X)

    #X = np.transpose(X)
    #
    #print("X.shape", X.shape)
    #print("y.shape", y.shape)
    #
    ##y = np.invert(y)
    #
    ##np_ones = np.ones({y.shape});
    #
    ##y = (y*2) - (np_ones);
    #
    #collections_y = collections.Counter(y)
    #
    #print("collections_y", collections_y)

    #train_data = X[:, 0:4000]
    #train_labels = y[0:4000]

    #test_data = X[:,4001:]
    #test_labels = y[4001:]

    #num_words = X.shape[1]
    #num_training = train_labels.shape[0]
    #num_testing = test_labels.shape[0]

    #print("train_data_shape", train_data.shape);
    #print("test_data_shape", test_data.shape);

    #print("train_labels_shape", train_labels.shape);
    #print("test_labels_shape", test_labels.shape);

    #print("num_words", num_words);
    #print("num_testing", num_testing);
    #print("num_training", num_training);

    return X, y


    

#load_all_data(10)
    
    
    #from sklearn.feature_extraction.text import TfidfVectorizer
    #tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    #X = tfidfconverter.fit_transform(documents).toarray()

