#!/usr/bin/env python
# coding: utf-8

# In[1]:


#############################################################
## ASSIGNMENT 1 CODE SKELETON
## RELEASED: 2/6/2019
## DUE: 2/15/2019
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################


# In[2]:


from collections import defaultdict
import gzip

import numpy as np

training_file = "data/complex_words_training.txt"
development_file = "data/complex_words_development.txt"
test_file = "data/complex_words_test_unlabeled.txt"


# ## 1. Evaluation Metrics

# In[3]:


## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    ## YOUR CODE HERE...
    TP, FP = 0, 0
    
    for index in range(len(y_pred)):
        if y_pred[index] == 1:
            if y_true[index] == 1:
                TP += 1
            else:
                FP += 1

    precision = TP / (TP + FP)

    return precision
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    ## YOUR CODE HERE...
    TP, FN = 0, 0

    for index in range(len(y_pred)):
        if y_pred[index] == y_true[index] == 1:
            TP += 1
        elif y_pred[index] == 0 != y_true[index]:
            FN += 1
    
    recall = TP / (TP + FN)

    return recall

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    ## YOUR CODE HERE...
    P = get_precision(y_pred, y_true)
    R = get_recall(y_pred, y_true)

    fscore = 2 * P * R / (P + R)

    return fscore


# ## 2. Complex Word Identification

# In[4]:


## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


# ### 2.1: A very simple baseline

# In[5]:


## Makes feature matrix for all complex
def all_complex_feature(words):
    return [1 for x in range(len(words))]

## Labels every word complex
def all_complex(data_file):
    ## YOUR CODE HERE...
    words, y_true = load_file(data_file)

    y_pred = all_complex_feature(words)

    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    
    performance = {'precision': precision, 'recall': recall, 'fscore': fscore}
    return performance

# ------------------2.1 Test---------------------------------------
print('------------------2.1 Test---------------------------------------')

performance_training = all_complex(training_file)
print('training performance:', performance_training)

performance_development = all_complex(development_file)
print('development performance:', performance_development)

print('-----------------------------------------------------------------')
# -----------------------------------------------------------------


# ### 2.2: Word length thresholding

# In[6]:


## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    y_pred = []
    for word in words:
        if len(word) < threshold:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_pred

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    ## YOUR CODE HERE
    training_words, training_y_true = load_file(training_file)
    development_words, development_y_true = load_file(development_file)

    best_threshold = 0
    max_fscore = 0
    best_training_performance = None
    for threshold in range(1, 20):
        # training_performance = [tprecision, trecall, tfscore]
        training_y_pred = length_threshold_feature(training_words, threshold)
            
        training_performance = {'precision': get_precision(training_y_pred, training_y_true),
                                'recall': get_recall(training_y_pred, training_y_true),
                                'fscore': get_fscore(training_y_pred, training_y_true)}
        if max_fscore < training_performance['fscore']:
            max_fscore = training_performance['fscore']
            best_training_performance = training_performance
            best_threshold = threshold

    print('the best threshould for training data (using F1-score):', best_threshold)
    
    # development_performance = [dprecision, drecall, dfscore]
    development_y_pred = length_threshold_feature(development_words, best_threshold)
    development_performance = {'precision': get_precision(development_y_pred, development_y_true),
                            'recall': get_recall(development_y_pred, development_y_true),
                            'fscore': get_fscore(development_y_pred, development_y_true)}
    
    return best_training_performance, development_performance

# ------------------2.2 Test---------------------------------------
print('------------------2.2 Test---------------------------------------')

performance_training, performance_development = word_length_threshold(training_file, development_file)
print('training performance:', performance_training)
print('development performance:', performance_development)

print('-----------------------------------------------------------------')
# -----------------------------------------------------------------


# ### 2.3: Word frequency thresholding

# In[7]:


## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt', encoding='UTF-8') as f: 
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

ngram_counts_file = "ngram_counts.txt.gz"
counts = load_ngram_counts(ngram_counts_file)


# In[8]:


# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    y_pred = []
    for word in words:
        if counts[word] < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_words, training_y_true = load_file(training_file)
    development_words, development_y_true = load_file(development_file)
    
    def find_best_threshold(words, y_true):
        min_freq = np.min(list(counts.values()))
        max_freq = np.max(list(counts.values()))
        
        max_fscore = 0
        best_threshold = 0
        
        scope = 10000000000
        lower, upper = min_freq + 1, max_freq
        while(scope > 0):
            current_lower, current_upper = lower, upper
            flag = False
            for threshold in range(current_lower, current_upper, scope):
                y_pred = frequency_threshold_feature(words, threshold, counts)
                fscore = get_fscore(y_pred, y_true)
                if fscore >= max_fscore:
                    max_fscore = fscore
                    best_threshold = threshold
                    if flag == False:
                        lower = max(min_freq + 1, threshold - scope)
                        flag = True
                    upper = min(threshold + scope, max_freq - 1)
            
            scope = int(scope / 10)
            
            print('max_fscore_approx =', max_fscore, 'in threshold scope (', lower, '-', upper, ')')
            
        return best_threshold, fscore
    
    training_best_threshold, training_fscore = find_best_threshold(training_words, training_y_true)
    
    print('---------------------------')
    print("best threshold for training data:", training_best_threshold, ", best fscore:", training_fscore)
    print('---------------------------')
    
    training_y_pred = frequency_threshold_feature(training_words, training_best_threshold, counts)
    # training_performance = [tprecision, trecall, tfscore]
    training_performance = {'precision': get_precision(training_y_pred, training_y_true),
                                'recall': get_recall(training_y_pred, training_y_true),
                                'fscore': get_fscore(training_y_pred, training_y_true)}
    
    development_best_threshold, development_fscore = find_best_threshold(development_words, development_y_true)
    
    print('---------------------------')
    print("best threshold for development data:", development_best_threshold, ", best fscore:", development_fscore)
    print('---------------------------')
    
    development_y_pred = frequency_threshold_feature(development_words, development_best_threshold, counts)
    # development_performance = [dprecision, drecall, dfscore]
    development_performance = {'precision': get_precision(development_y_pred, development_y_true),
                                'recall': get_recall(development_y_pred, development_y_true),
                                'fscore': get_fscore(development_y_pred, development_y_true)}
    
    return training_performance, development_performance


# ------------------2.3 Test---------------------------------------
print('------------------2.3 Test---------------------------------------')

training_performance, development_performance = word_frequency_threshold(training_file, development_file, counts)
print('training performance:', training_performance)
print('development performance:', development_performance)

print('-----------------------------------------------------------------')
# -----------------------------------------------------------------


# ### 2.4: Naive Bayes

# In[9]:


def get_length_and_frequency(words, counts, mean=None, std=None):
        length_frequency = []
        for word in words:
            length_frequency.append([len(word), counts[word]])
        
        length_frequency = np.array(length_frequency)
        
        if mean is None:
            mean = np.mean(length_frequency, axis=0)
            std = np.std(length_frequency, axis=0)
        normalized_length_frequency = (length_frequency - mean) / std
        
        return normalized_length_frequency, mean, std


# In[10]:


## Trains a Naive Bayes classifier using length and frequency features
from sklearn.naive_bayes import GaussianNB

def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_words, training_y_true = load_file(training_file)
    development_words, development_y_true = load_file(development_file)
    
    X_train, training_mean, training_std = get_length_and_frequency(training_words, counts)
    Y_train = np.array(training_y_true)
    
    NB_classifier = GaussianNB()
    NB_classifier.fit(X_train, Y_train)
    
    X_development, _, _ = get_length_and_frequency(development_words, counts, training_mean, training_std)
    Y_development = np.array(development_y_true)
    development_y_pred = NB_classifier.predict(X_development)
    
    # development_performance = (dprecision, drecall, dfscore)
    development_performance = {'precision': get_precision(development_y_pred, development_y_true),
                                'recall': get_recall(development_y_pred, development_y_true),
                                'fscore': get_fscore(development_y_pred, development_y_true)}
    return development_performance

# ------------------2.4 Test---------------------------------------
print('------------------2.4 Test---------------------------------------')

training_performance = naive_bayes(training_file, training_file, counts)
development_performance = naive_bayes(training_file, development_file, counts)
print('training performance:', training_performance)
print('development performance:', development_performance)

print('-----------------------------------------------------------------')
# -----------------------------------------------------------------


# ### 2.5: Logistic Regression

# In[11]:


## Trains a Naive Bayes classifier using length and frequency features
from sklearn.linear_model import LogisticRegression

def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_words, training_y_true = load_file(training_file)
    development_words, development_y_true = load_file(development_file)
    
    X_train, training_mean, training_std = get_length_and_frequency(training_words, counts)
    Y_train = np.array(training_y_true)
    
    LR_classifier = LogisticRegression()
    LR_classifier.fit(X_train, Y_train)
    
    X_development, _, _ = get_length_and_frequency(development_words, counts, training_mean, training_std)
    Y_development = np.array(development_y_true)
    development_y_pred = LR_classifier.predict(X_development)
    
    # development_performance = (dprecision, drecall, dfscore)
    development_performance = {'precision': get_precision(development_y_pred, development_y_true),
                                'recall': get_recall(development_y_pred, development_y_true),
                                'fscore': get_fscore(development_y_pred, development_y_true)}
    return development_performance

# ------------------2.5 Test---------------------------------------
print('------------------2.5 Test---------------------------------------')

training_performance = logistic_regression(training_file, training_file, counts)
development_performance = logistic_regression(training_file, development_file, counts)
print('training performance:', training_performance)
print('development performance:', development_performance)

print('-----------------------------------------------------------------')
# -----------------------------------------------------------------


# ## 2.7: Build your own classifier
# 
# ### features used:
# 1. word length
# 2. word frequency
# 3. syllable number
# 4. synonym number

# In[1]:


import syllables
from nltk.corpus import wordnet 

# generate feature data
def get_features(words, counts, normalize_mean=None, normalize_std=None):
    length_frequency, normalize_mean, normalize_std = get_length_and_frequency(words, counts, normalize_mean, normalize_std)
    syllable_count = []
    synonym_count= []
    frequency_ratio = []
    
    for word in words:
        syllable_count.append([syllables.count_syllables(word)])
        
        synonym = []
        for syn in wordnet.synsets(word):
            for orginal in syn.lemmas():
                synonym.append(orginal.name())
        synonym = set(synonym)
        
        synonym_count.append([len(synonym)])
        
    features = np.concatenate((np.array(length_frequency), np.array(syllable_count), np.array(synonym_count)), axis=1)
    
    return features, normalize_mean, normalize_std


# In[13]:


# compute the performance (precision, recall, fscore)
def get_performance(y_pred, y_true):
    performance = {'precision': get_precision(y_pred, y_true),
                'recall': get_recall(y_pred, y_true),
                'fscore': get_fscore(y_pred, y_true)}
    return performance

training_file = "data/complex_words_training.txt"
development_file = "data/complex_words_development.txt"
test_file = "data/complex_words_test_unlabeled.txt"

training_words, training_labels = load_file(training_file)
development_words, development_labels = load_file(development_file)

training_words, training_labels = np.array(training_words), np.array(training_labels)
development_words, development_labels = np.array(development_words), np.array(development_labels)

training_development_words = np.concatenate((training_words, development_words), axis=0)
training_development_labels = np.concatenate((training_labels, development_labels), axis=0)


# ### model 1: Naive Bayes

# In[14]:


from sklearn.naive_bayes import GaussianNB

def NB_2(training_words, training_labels, predict_words, counts):
    training_features, mean, std = get_features(training_words, counts)
    predict_features, _, _ = get_features(predict_words, counts, mean, std)
    
    NB_classifier = GaussianNB()
    NB_classifier.fit(training_features, training_labels)
    
    y_pred = NB_classifier.predict(predict_features)
    
    return y_pred

training_development_y_pred = NB_2(training_development_words, training_development_labels, training_development_words, counts)
print('perofrmance:', get_performance(training_development_y_pred, training_development_labels))


# ### model 2: Support Vector Machine

# In[15]:


from sklearn.svm import SVC

def SVM(training_words, training_labels, predict_words, counts):
    training_features, mean, std = get_features(training_words, counts)
    predict_features, _, _ = get_features(predict_words, counts, mean, std)
    
    SVM_classifier = SVC()
    SVM_classifier.fit(training_features, training_labels)
    
    y_pred = SVM_classifier.predict(predict_features)
    
    return y_pred

training_development_y_pred = SVM(training_development_words, training_development_labels, training_development_words, counts)
print('perofrmance:', get_performance(training_development_y_pred, training_development_labels))


# ### model 3: K Nearest Neighbors

# In[16]:


from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def KNN(training_words, training_labels, predict_words, counts):
    training_features, mean, std = get_features(training_words, counts)
    predict_features, _, _ = get_features(predict_words, counts, mean, std)
    
    KNN_classifier = KNeighborsClassifier(n_neighbors=3)
    KNN_classifier.fit(training_features, training_labels)
        
    y_pred = KNN_classifier.predict(predict_features)
    
    return y_pred

training_development_y_pred = KNN(training_development_words, training_development_labels, training_development_words, counts)
print('perofrmance:', get_performance(training_development_y_pred, training_development_labels))


# ### model 4: Feed-Forward Neural Network

# In[21]:


import keras
from keras.layers.core import Activation, Dense
import h5py

def build_model(training_words, training_labels):
    training_features, mean, std = get_features(training_words, counts)
    NN_model = keras.models.Sequential()
    adam_opt = keras.optimizers.Adam(lr=0.001)
    
    NN_model.add(Dense(32, input_shape=(4,)))
    NN_model.add(Activation('relu'))
    NN_model.add(Dense(32))
    NN_model.add(Activation('relu'))
    NN_model.add(Dense(1))
    NN_model.add(Activation('sigmoid'))
    
    NN_model.compile(optimizer=adam_opt, loss='mse')
    NN_model.fit(training_features, training_labels, epochs=100)
    
    NN_model.save('neural_network.h5')

# build_model(training_development_words, training_development_labels)
def neural_network(NN_model, training_words, training_labels, predict_words):
    training_features, mean, std = get_features(training_words, counts)
    predict_features, _, _ = get_features(predict_words, counts, mean, std)
    
    y_pred = NN_model.predict(predict_features)
    
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    
    return y_pred

NN_model = keras.models.load_model('neural_network.h5')
training_development_y_pred = neural_network(NN_model, training_development_words, training_development_labels, training_development_words)
print('perofrmance:', get_performance(training_development_y_pred, training_development_labels))


# ### Combo model: decision maker (with neural network)

# In[22]:


def build_coef_net(training_words, training_labels, counts):
    from sklearn.model_selection import train_test_split
    
    NN_model = keras.models.load_model('neural_network.h5')
    
    classifier_model = keras.models.Sequential()
    adam_opt = keras.optimizers.Adam(lr=0.001)

    classifier_model.add(Dense(32, input_shape=(4,)))
    classifier_model.add(Activation('relu'))
    classifier_model.add(Dense(32))
    classifier_model.add(Activation('relu'))
    classifier_model.add(Dense(1))
    classifier_model.add(Activation('sigmoid'))

    classifier_model.compile(optimizer=adam_opt, loss='mse')
    
    for epoch in range(50):
        X_train, X_test, y_train, y_test = train_test_split(training_words, training_labels, test_size=0.2)
        
        y_pred_NB = NB_2(X_train, y_train, X_test, counts).reshape(-1, 1)
        y_pred_SVM = SVM(X_train, y_train, X_test, counts).reshape(-1, 1)
        y_pred_KNN = KNN(X_train, y_train, X_test, counts).reshape(-1, 1)
        y_pred_NN = neural_network(NN_model, X_train, y_train, X_test).reshape(-1, 1)

        predict_features = np.concatenate((y_pred_NB, y_pred_SVM, y_pred_KNN, y_pred_NN), axis=1)
        classifier_model.fit(predict_features, y_test)
    
    classifier_model.save('main_classifier_model.h5')

# build_coef_net(training_development_words, training_development_labels, counts)


# ## My complex word classifier
# 1. use multiple models to predict the labels
# 2. use decision maker model to judge and make prediction

# In[23]:


def decision_model(y_pred_NB, y_pred_SVM, y_pred_KNN, y_pred_NN):
    predict_features = np.concatenate((y_pred_NB, y_pred_SVM, y_pred_KNN, y_pred_NN), axis=1)
    
    classifier_model = keras.models.load_model('main_classifier_model.h5')
    y_pred = classifier_model.predict(predict_features)
    
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    
    return y_pred

def main_classifier(training_words, training_labels, predict_words, counts):
    training_features, mean, std = get_features(training_words, counts)
    predict_features, _, _ = get_features(predict_words, counts, mean, std)
    
    y_pred_NB = NB_2(training_words, training_labels, predict_words, counts).reshape(-1, 1)
    y_pred_SVM = SVM(training_words, training_labels, predict_words, counts).reshape(-1, 1)
    y_pred_KNN = KNN(training_words, training_labels, predict_words, counts).reshape(-1, 1)
    y_pred_NN = neural_network(NN_model, training_words, training_labels, predict_words).reshape(-1, 1)
    
    y_pred = decision_model(y_pred_NB, y_pred_SVM, y_pred_KNN, y_pred_NN)
    
    return y_pred

training_development_y_pred = main_classifier(training_development_words, training_development_labels, training_development_words, counts)
print('perofrmance:', get_performance(training_development_y_pred, training_development_labels))


# In[25]:


if __name__ == "__main__":
    test_file = "data/complex_words_test_unlabeled.txt"
    test_words = []
    with open(test_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                test_words.append(line_split[0].lower())
            i += 1
    
    test_words = np.array(test_words)
    
    test_y_pred = main_classifier(training_development_words, training_development_labels, test_words, counts)
    print('output size:', test_y_pred.shape)
    
    with open('test_labels.txt', 'w') as f:
        for label in test_y_pred:
            f.writelines(str(int(label[0])) + '\n')
    print('finished')


# In[ ]:




