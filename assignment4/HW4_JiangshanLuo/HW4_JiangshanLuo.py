
# coding: utf-8

# # Part 1: Constrainted Model

# In[22]:


from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support

# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    features = [
        (o + 'word', word),
        # TODO: add more features here.
        ('contain_digit', any(char.isdigit() for char in word)),
        ('all_digit', all(char.isdigit() for char in word)),
        ('contain_cap', any(char.isupper() for char in word)),
        ('all_cap', all(char.isupper() for char in word)),
        ('contain_specChar', any(not char.isalpha() and not char.isdigit() for char in word))
    ]
    #print(features)
    return features
    

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in range(-1, 2):
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            featlist = getfeats(word, o)
            features.extend(featlist)
    
    return dict(features)


# In[19]:


if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))
    
    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # TODO: play with other models
    #model = Perceptron(verbose=1)
    
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(learning_rate_init=0.005, verbose=1, max_iter=3)
    
#     from sklearn.tree import DecisionTreeClassifier
#     model = DecisionTreeClassifier(criterion='entropy', verbose=1)
    
#     from sklearn.ensemble import RandomForestClassifier
#     model = RandomForestClassifier(verbose=1)
    
    model.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in dev_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("constrained_results.txt", "w") as out:
        for sent in dev_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py constrained_results.txt")


# In[20]:


import conlleval

conlleval.main(["", "constrained_results.txt"])


# # Part 2: Unconstrainted Model

# In[1]:


import numpy as np
import re
import gc
import math
import random
import tensorflow as tf
import tflearn
from nltk.corpus import conll2002


# ## use google pretrained Word2Vec

# In[2]:


import gensim

# train model
Word2Vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  

print(Word2Vec_model['sentence'])


# In[3]:


def getfeats(word, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
#     features = [
#         (o + 'word', word),
#         # TODO: add more features here.
#         ('contain_digit', any(char.isdigit() for char in word)),
#         ('all_digit', all(char.isdigit() for char in word)),
#         ('contain_cap', any(char.isupper() for char in word)),
#         ('all_cap', all(char.isupper() for char in word)),
#         ('contain_specChar', any(not char.isalpha() and not char.isdigit() for char in word))
#     ]
    try:
        word_vec = Word2Vec_model[word]
    except:
        word_vec = Word2Vec_model['unknown']
    
    features = list(word_vec) + [any(char.isdigit() for char in word), all(char.isdigit() for char in word),
                any(char.isupper() for char in word), all(char.isupper() for char in word),
                any(not char.isalpha() and not char.isdigit() for char in word)]
    #print(features)
    return features
    

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in range(-1, 2):
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
        elif i+o < 0:
            word = '$'
        elif i+o >= len(sent):
            word = '#'
        featlist = getfeats(word, o)
        features.extend(featlist)
    
    return features


# ## FFN

# In[4]:


# parameters
training_epoch = 1
learning_rate = 0.005
batch_size = 100

# Network parameters
n_hidden_1 = 300
n_hidden_2 = 300
n_hidden_3 = 300


def mlp_ff(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['h3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # out_layer = tf.nn.softmax(out_layer)
    return out_layer


# In[8]:


# Load the training data
train_sents = list(conll2002.iob_sents('esp.train'))
dev_sents = list(conll2002.iob_sents('esp.testa'))
test_sents = list(conll2002.iob_sents('esp.testb'))

label_set = set()
for sent in train_sents:
    for i in range(len(sent)):
        label_set.add(sent[i][-1])

label_encode = dict([(list(label_set)[index], index) for index in range(len(label_set))])
label_decode = dict([(index, list(label_set)[index]) for index in range(len(label_set))])

# placeholder
n_input = len(word2features(['default'], 0))
n_classes = len(label_set)
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])

# weights
weights = {
    'h1': tf.Variable(
        tf.random_uniform([n_input, n_hidden_1], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
        name='w_h1'),
    'h2': tf.Variable(
        tf.random_uniform([n_hidden_1, n_hidden_2], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
        name='w_h2'),
    'h3': tf.Variable(
        tf.random_uniform([n_hidden_2, n_hidden_3], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
        name='w_h3'),
    'out': tf.Variable(
        tf.random_uniform([n_hidden_3, n_classes], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
        name='w_out')
}

biases = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1]), name='b_h1'),
    'h2': tf.Variable(tf.random_normal([n_hidden_2]), name='b_h2'),
    'h3': tf.Variable(tf.random_normal([n_hidden_3]), name='b_h3'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='b_out')
}

# tf.add_to_collection('vars', weights)
# tf.add_to_collection('vars', biases)
#
# saver = tf.train.Saver()

pred = mlp_ff(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# prediction
y_p = tf.argmax(pred, 1)

init = tf.global_variables_initializer()

save_path = ''
saver = tf.train.Saver()


# In[9]:


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epoch):
        avg_cost = 0
        # random.shuffle(batch_list)
        count = 0
        for sent in train_sents:
            count += 1
            avg_cost = 0
            for i in range(len(sent)):
                feats = word2features(sent,i)
                train_x = feats
                train_y = [x == label_encode[sent[i][-1]] for x in range(n_classes)]
                _, c = sess.run([optimizer, cost], feed_dict={x: [train_x], y: [train_y]})
                avg_cost += c
            if count % 1000 == 0:
                print('process:', count)

        print("Epoch:", '%04d' % (epoch + 1), "cost=",               "{:.9f}".format(avg_cost))

    save_path = saver.save(sess, './models/tf_mlp.ckpt')
    print(save_path)


# In[14]:


with tf.Session() as sess:
    sess.run(init)

    print(save_path)
    saver.restore(sess, './models/tf_mlp.ckpt')
    print("Model restored from file: %s" % save_path)

    with open("unconstrained_results.txt", "w") as out:
        for sent in dev_sents:
            for i in range(len(sent)):
                feats = word2features(sent,i)
                test_x = feats
                test_y = [x == label_encode[sent[i][-1]] for x in range(n_classes)]

                y_pred = sess.run(pred, feed_dict={x: [test_x], y: [test_y]})
                y_pred = label_decode[np.argmax(y_pred)]
                word = sent[i][0]
                gold = sent[i][-1]
                out.write("{}\t{}\t{}\n".format(word,gold,y_pred))
        out.write("\n")

print("Now run: python conlleval.py unconstrained_results.txt")


# In[23]:


import conlleval

conlleval.main(["", "unconstrained_results.txt"])

