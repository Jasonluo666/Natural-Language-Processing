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

from collections import defaultdict
import gzip

import numpy as np

training_file = "data/complex_words_training.txt"
development_file = "data/complex_words_development.txt"
test_file = "data/complex_words_test_unlabeled.txt"
ngram_counts_file = "ngram_counts.txt.gz"

#### 1. Evaluation Metrics ####

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

#### 2. Complex Word Identification ####

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
    
### 2.1: A very simple baseline

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

### 2.2: Word length thresholding

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
    development_words, development_true = load_file(development_file)

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
    development_performance = {'precision': get_precision(development_y_pred, development_true),
                            'recall': get_recall(development_y_pred, development_true),
                            'fscore': get_fscore(development_y_pred, development_true)}
    
    return best_training_performance, development_performance

# ------------------2.2 Test---------------------------------------
print('------------------2.2 Test---------------------------------------')

performance_training, performance_development = word_length_threshold(training_file, development_file)
print('training performance:', performance_training)
print('development performance:', performance_development)

print('-----------------------------------------------------------------')
# -----------------------------------------------------------------

### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt', encoding='UTF-8') as f: 
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    y_pred = []
    for word in words:
        if counts[word] < threshold:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_pred

def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_words, training_y_true = load_file(training_file)
    development_words, development_true = load_file(development_file)

    min_freq = np.min(list(counts.values()))
    max_freq = np.max(list(counts.values()))

    
    for threshold in range(min_freq, max_freq, 1000):
        training_y_pred = frequency_threshold_feature(training_words, threshold, counts)
        print(get_fscore(training_y_pred, training_y_true))

    # training_performance = [tprecision, trecall, tfscore]
    # development_performance = [dprecision, drecall, dfscore]
    # return training_performance, development_performance

counts = load_ngram_counts(ngram_counts_file)
word_frequency_threshold(training_file, development_file, counts)
### 2.4: Naive Bayes
        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE    
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)
    
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)
