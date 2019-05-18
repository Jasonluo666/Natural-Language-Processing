
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='word')

# Load the training data
train_data = []
with open("train/english_train.text", 'r', encoding='utf8') as f:
    for line in f.readlines():
#         train_data.append(tknzr.tokenize(line.lower()))
        train_data.append(line.lower())
train_label = []
with open("train/english_train.labels", 'r', encoding='utf8') as f:
    for line in f.readlines():
        train_label.append(int(line[:-1]))

# Load the testing data
test_data = []
with open("test/english_test.text", 'r', encoding='utf8') as f:
    for line in f.readlines():
#         test_data.append(tknzr.tokenize(line.lower()))
        test_data.append(line.lower())
test_label = []
with open("test/english_test.labels", 'r', encoding='utf8') as f:
    for line in f.readlines():
        test_label.append(int(line[:-1]))

vectorizer.fit(train_data)
vectorizer.vocabulary_

train_X = vectorizer.transform(train_data)
train_Y = train_label

train_X

from sklearn.svm import LinearSVC
model = LinearSVC(dual=False, C=0.1, verbose=0)

model.fit(train_X, train_Y)

test_X = vectorizer.transform(test_data)
test_Y = test_label

y_pred = model.predict(test_X)
y_pred

with open("output/gold_labels_english.txt", "w", encoding='utf8') as out:
    for i, sent in enumerate(test_data): 
        gold = test_Y[i]
        out.write("{}\n".format(gold))
    out.write("\n")

with open("output/predicted_labels_english.txt", "w", encoding='utf8') as out:
    for i, sent in enumerate(test_data): 
        pred = y_pred[i]
        out.write("{}\n".format(pred))
    out.write("\n")

# python scorer_semeval18.py gold_labels_file predicted_labels_file
import scorer_semeval18
scorer_semeval18.main("output/gold_labels_english.txt", "output/predicted_labels_english.txt")

tknzr = TweetTokenizer()
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Load the training data
train_data = []
with open("train/spanish_train.text", 'r', encoding='utf8') as f:
    for line in f.readlines():
#         train_data.append(tknzr.tokenize(line.lower()))
        train_data.append(line.lower())
train_label = []
with open("train/spanish_train.labels", 'r', encoding='utf8') as f:
    for line in f.readlines():
        train_label.append(int(line[:-1]))

# Load the testing data
test_data = []
with open("test/spanish_test.text", 'r', encoding='utf8') as f:
    for line in f.readlines():
#         test_data.append(tknzr.tokenize(line.lower()))
        test_data.append(line.lower())
test_label = []
with open("test/spanish_test.labels", 'r', encoding='utf8') as f:
    for line in f.readlines():
        test_label.append(int(line[:-1]))

vectorizer.fit(train_data)
print(vectorizer.transform([train_data[0]]).toarray())

train_X = vectorizer.transform(train_data)
train_Y = train_label

train_X

from sklearn.svm import LinearSVC
SVM_model = LinearSVC(dual=False, C=0.1, verbose=0)

from sklearn.neural_network import MLPClassifier
MLP_model = MLPClassifier(learning_rate_init=0.005, verbose=1)

from sklearn.linear_model import LogisticRegression
LogisticRegression_model = LogisticRegression(multi_class='multinomial', solver='saga')

from sklearn.ensemble import RandomForestClassifier
RandomForest_model = RandomForestClassifier()

def multi_model_train(train_X, train_Y):
    print('RandomForest_model')
    RandomForest_model.fit(train_X, train_Y)

    print('LogisticRegression_model')
    LogisticRegression_model.fit(train_X, train_Y)

    print('SVM_model')
    SVM_model.fit(train_X, train_Y)
    
    import numpy as np

    training_vec = []
    training_vec.append(SVM_model.predict(train_X))
    training_vec.append(RandomForest_model.predict(train_X))
    training_vec.append(LogisticRegression_model.predict(train_X))

    training_vec = np.array(training_vec)
    print(training_vec.shape)
    training_vec = np.rot90(training_vec)
    print(training_vec.shape)

    print('MLP_model')
    MLP_model.fit(training_vec, train_Y)

def multi_model_predict(X):
    training_vec = []
    training_vec.append(SVM_model.predict(X))
    training_vec.append(RandomForest_model.predict(X))
    training_vec.append(LogisticRegression_model.predict(X))

    X_vec = np.array(training_vec)
    print(X_vec.shape)
    X_vec = np.rot90(training_vec)
    print(X_vec.shape)
    print(X_vec[:10], test_Y[:10])

    return LogisticRegression_model.predict(X)

multi_model_train(train_X, train_Y)

test_X = vectorizer.transform(test_data)
test_Y = test_label

y_pred = multi_model_predict(test_X)
y_pred

with open("output/gold_labels_spanish.txt", "w", encoding='utf8') as out:
    for i, sent in enumerate(test_data): 
        gold = test_Y[i]
        out.write("{}\n".format(gold))
    out.write("\n")

with open("output/predicted_labels_spanish.txt", "w", encoding='utf8') as out:
    for i, sent in enumerate(test_data): 
        pred = y_pred[i]
        out.write("{}\n".format(pred))
    out.write("\n")

# python scorer_semeval18.py gold_labels_file predicted_labels_file
import scorer_semeval18
scorer_semeval18.main("output/gold_labels_spanish.txt", "output/predicted_labels_spanish.txt")

from translate import Translator
translator= Translator(to_lang="es")

translator.translate("hi")

en_hashmap = {}
with open("mapping/english_mapping.txt", 'r', encoding='utf8') as f:
    for line in f.readlines():
        line = line.split()
        en_hashmap[line[0]] = line[1]

es_hashmap = {}
with open("mapping/spanish_mapping.txt", 'r', encoding='utf8') as f:
    for line in f.readlines():
        line = line.split()
        es_hashmap[line[1]] = line[0]

en2es_hashmap = {}
for key in en_hashmap:
    if en_hashmap[key] in es_hashmap:
        en2es_hashmap[key] = es_hashmap[en_hashmap[key]]
en2es_hashmap

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='word')

# Load the training data
train_data = []
with open("train/spanish_train.text", 'r', encoding='utf8') as f:
    for line in f.readlines():
#         train_data.append(tknzr.tokenize(line.lower()))
        train_data.append(line.lower())
train_label = []
with open("train/spanish_train.labels", 'r', encoding='utf8') as f:
    for line in f.readlines():
        train_label.append(int(line[:-1]))

# load extra data
extra_label = []
missing_pos = []
with open("train/english_train.labels", 'r', encoding='utf8') as f:
    for index, line in enumerate(f.readlines()[:-10000]):
        if line[:-1] in en2es_hashmap:
            extra_label.append(int(line[:-1]))
        else:
            missing_pos.append(index)

extra_data = []
with open("train/extra_data.txt", 'r', encoding='utf8') as f:
    for index, line in enumerate(f.readlines()):
        if index not in missing_pos:
            extra_data.append(line.lower())

train_data += extra_data
train_label += extra_label

# Load the testing data
test_data = []
with open("test/spanish_test.text", 'r', encoding='utf8') as f:
    for line in f.readlines():
#         test_data.append(tknzr.tokenize(line.lower()))
        test_data.append(line.lower())
test_label = []
with open("test/spanish_test.labels", 'r', encoding='utf8') as f:
    for line in f.readlines():
        test_label.append(int(line[:-1]))

vectorizer.fit(train_data)
print(vectorizer.transform([train_data[0]]).toarray())

train_X = vectorizer.transform(train_data)
train_Y = train_label

train_X

from sklearn.linear_model import LogisticRegression
LogisticRegression_model = LogisticRegression(multi_class='multinomial', solver='saga')

LogisticRegression_model.fit(train_X, train_Y)

test_X = vectorizer.transform(test_data)
test_Y = test_label

y_pred = LogisticRegression_model.predict(test_X)
y_pred

with open("output/gold_labels_spanish_plus.txt", "w", encoding='utf8') as out:
    for i, sent in enumerate(test_data): 
        gold = test_Y[i]
        out.write("{}\n".format(gold))
    out.write("\n")

with open("output/predicted_labels_spanish_plus.txt", "w", encoding='utf8') as out:
    for i, sent in enumerate(test_data): 
        pred = y_pred[i]
        out.write("{}\n".format(pred))
    out.write("\n")

# python scorer_semeval18.py gold_labels_file predicted_labels_file
import scorer_semeval18
scorer_semeval18.main("output/gold_labels_spanish_plus.txt", "output/predicted_labels_spanish_plus.txt")
