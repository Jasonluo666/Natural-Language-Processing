
# coding: utf-8

# In[1]:


import os
import csv
import subprocess
import re
import random
import numpy as np


# In[2]:


def read_in_shakespeare():
    '''Reads in the Shakespeare dataset processesit into a list of tuples.
        Also reads in the vocab and play name lists from files.

        Each tuple consists of
        tuple[0]: The name of the play
        tuple[1] A line from the play as a list of tokenized words.

        Returns:
        tuples: A list of tuples in the above format.
        document_names: A list of the plays present in the corpus.
        vocab: A list of all tokens in the vocabulary.
    '''

    tuples = []
    
    with open('will_play_text.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            play_name = row[1]
            line = row[5]
            line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((play_name, line_tokens))

    with open('vocab.txt') as f:
        vocab =  [line.strip() for line in f]

    with open('play_names.txt') as f:
        document_names =  [line.strip() for line in f]

    return tuples, document_names, vocab

tuples, document_names, vocab = read_in_shakespeare()


# ### create_term_document_matrix

# In[3]:


def get_row_vector(matrix, row_id):
    return matrix[row_id, :]

def get_column_vector(matrix, col_id):
    return matrix[:, col_id]

def create_term_document_matrix(line_tuples, document_names, vocab):
    '''Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    document_names: A list of the document names
    vocab: A list of the tokens in the vocabulary

    Let m = len(vocab) and n = len(document_names).

    Returns:
    td_matrix: A mxn numpy array where the number of rows is the number of words
        and each column corresponds to a document. A_ij contains the
        frequency with which word i occurs in document j.
    '''

    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    docname_to_id = dict(zip(document_names, range(0, len(document_names))))

    # YOUR CODE HERE
    matrix = np.zeros((len(vocab), len(document_names)))
    
    for line in line_tuples:
        doc_name = line[0]
        words = line[1]
        for word in words:
            matrix[vocab_to_id[word]][docname_to_id[doc_name]] += 1
            
    return matrix


term_document_matrix = create_term_document_matrix(tuples, document_names, vocab)


# ### compute_cosine_similarity

# In[4]:


def compute_cosine_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

    Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

    Returns:
    A scalar similarity value.
    '''

    # YOUR CODE HERE
    inner = np.inner(vector1, vector2)
    length = (np.sqrt(np.sum(vector1 ** 2)) * np.sqrt(np.sum(vector2 ** 2)))
    
    if length == 0:
        return 0
    
    return inner / length


similarity_matrix_cos = np.zeros((len(document_names), len(document_names)))

for x in range(len(document_names)):
    for y in range(x + 1, len(document_names)):
        similarity_matrix_cos[x][y] = similarity_matrix_cos[y][x] = compute_cosine_similarity(get_column_vector(term_document_matrix, x), get_column_vector(term_document_matrix, y))

for index in range(len(similarity_matrix_cos)):
    closest_id = np.argmax(similarity_matrix_cos[index])
    print('the closest play to', document_names[index], '---is---', document_names[closest_id])


# ### create_term_context_matrix

# In[5]:


def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    '''Returns a numpy array containing the term context matrix for the input lines.

    Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    vocab: A list of the tokens in the vocabulary

    Let n = len(vocab).

    Returns:
    tc_matrix: A nxn numpy array where A_ij contains the frequency with which
        word j was found within context_window_size to the left or right of
        word i in any sentence in the tuples.
    '''

    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))

    # YOUR CODE HERE
    term_context_matrix = np.zeros((len(vocab), len(vocab)))
    
    for line in line_tuples:
        for word_id in range(len(line[1])):
            word = line[1][word_id]
            # term_context_matrix[vocab_to_id[word]][vocab_to_id[word]] += 1
            for shift in range(1, context_window_size + 1):
                left = word_id - shift
                right = word_id + shift
                if left >= 0:
                    term_context_matrix[vocab_to_id[word]][vocab_to_id[line[1][left]]] += 1
                if right < len(line[1]):
                    term_context_matrix[vocab_to_id[word]][vocab_to_id[line[1][right]]] += 1
            
    return term_context_matrix

term_context_matrix = create_term_context_matrix(tuples, vocab, 2)


# ### create_PPMI_matrix

# In[6]:


def create_PPMI_matrix(term_context_matrix):
    '''Given a term context matrix, output a PPMI matrix.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
    term_context_matrix: A nxn numpy array, where n is
    the numer of tokens in the vocab.

    Returns: A nxn numpy matrix, where A_ij is equal to the
    point-wise mutual information between the ith word
    and the jth word in the term_context_matrix.
    '''       

    # YOUR CODE HERE
    from scipy import sparse
    
    try:
        PPMI_matrix = sparse.load_npz("PPMI_matrix.npz").toarray()
    except:
        PPMI_matrix = np.zeros_like(term_context_matrix)

        total = np.sum(term_context_matrix)
        count_p_i = np.zeros(term_context_matrix.shape[0])
        count_p_j = np.zeros(term_context_matrix.shape[0])
        for j in range(term_context_matrix.shape[0]):
            count_p_i[j] = np.sum(term_context_matrix[:, j]) / total
        for i in range(term_context_matrix.shape[0]):
            count_p_j[i] = np.sum(term_context_matrix[i, :]) / total

        for i in range(term_context_matrix.shape[0]):
            for j in range(term_context_matrix.shape[1]):
                p_ij = term_context_matrix[i][j] / total
                p_i = count_p_i[j]
                p_j = count_p_j[i]

                if p_ij > 0:
                    PPMI_matrix[i][j] = max(np.log(p_ij / (p_i * p_j)), 0)
                else:
                    PPMI_matrix[i][j] = 0
        
        PPMI_matrix = sparse.csc_matrix(PPMI_matrix)
        sparse.save_npz("PPMI_matrix.npz",PPMI_matrix)
    return PPMI_matrix

PPMI_matrix = create_PPMI_matrix(term_context_matrix)


# ### create_tf_idf_matrix

# In[7]:


def create_tf_idf_matrix(term_document_matrix):
    '''Given the term document matrix, output a tf-idf weighted version.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
    term_document_matrix: Numpy array where each column represents a document 
    and each row, the frequency of a word in that document.

    Returns:
    A numpy array with the same dimension as term_document_matrix, where
    A_ij is weighted by the inverse document frequency of document h.
    '''

    # YOUR CODE HERE
    num_of_doc = term_document_matrix.shape[1]
    tf_idf_matrix = np.zeros_like(term_document_matrix)
    for word_id in range(len(term_document_matrix)):
        
        word_vector = term_document_matrix[word_id]
        doc_contain_word = np.sum(word_vector > 0)
        for doc_id in range(len(word_vector)):
            tf = 0 if word_vector[doc_id] == 0 else 1 + np.log10(word_vector[doc_id])
            idf = np.log(num_of_doc / doc_contain_word)
            tf_idf_matrix[word_id][doc_id] = tf * idf
    
    return tf_idf_matrix
create_tf_idf_matrix(term_document_matrix)


# ### compute_jaccard_similarity

# In[8]:


def compute_jaccard_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

    Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

    Returns:
    A scalar similarity value.
    '''

    # YOUR CODE HERE
    concatenated_vector = np.concatenate((vector1.reshape(-1, 1), vector2.reshape(-1, 1)), axis=1)
    return np.sum(np.min(concatenated_vector, axis=1)) / np.sum(np.max(concatenated_vector, axis=1))

similarity_matrix_jaccard = np.zeros((len(document_names), len(document_names)))

for x in range(len(document_names)):
    for y in range(x + 1, len(document_names)):
        similarity_matrix_jaccard[x][y] = similarity_matrix_jaccard[y][x] = compute_jaccard_similarity(get_column_vector(term_document_matrix, x), get_column_vector(term_document_matrix, y))

for index in range(len(similarity_matrix_jaccard)):
    closest_id = np.argmax(similarity_matrix_jaccard[index])
    print('the closest play to', document_names[index], '---is---', document_names[closest_id])


# ### compute_dice_similarity

# In[9]:


def compute_dice_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

    Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

    Returns:
    A scalar similarity value.
    '''

    # YOUR CODE HERE  2 * sum(min(a,b)) / sum(a+b)
    concatenated_vector = np.concatenate((vector1.reshape(-1, 1), vector2.reshape(-1, 1)), axis=1)
    return 2 * np.sum(np.min(concatenated_vector, axis=1)) / np.sum(vector1 + vector2)

similarity_matrix_dice = np.zeros((len(document_names), len(document_names)))

for x in range(len(document_names)):
    for y in range(x + 1, len(document_names)):
        similarity_matrix_dice[x][y] = similarity_matrix_dice[y][x] = compute_dice_similarity(get_column_vector(term_document_matrix, x), get_column_vector(term_document_matrix, y))

for index in range(len(similarity_matrix_dice)):
    closest_id = np.argmax(similarity_matrix_dice[index])
    print('the closest play to', document_names[index], '---is---', document_names[closest_id])


# ### rank_plays

# In[10]:


def rank_plays(target_play_index, term_document_matrix, similarity_fn):
    ''' Ranks the similarity of all of the plays to the target play.

    Inputs:
    target_play_index: The integer index of the play we want to compare all others against.
    term_document_matrix: The term-document matrix as a mxn numpy array.
    similarity_fn: Function that should be used to compared vectors for two
      documents. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

    Returns:
    A length-n list of integer indices corresponding to play names,
    ordered by decreasing similarity to the play indexed by target_play_index
    '''
    
    # YOUR CODE HERE
    similarity = []
    for index in range(term_document_matrix.shape[1]):
        similarity.append([similarity_fn(get_column_vector(term_document_matrix, target_play_index), get_column_vector(term_document_matrix, index)), index])
    
    sorted_list = np.array(sorted(similarity, key=lambda x:x[0], reverse=True))
    # print(sorted_list)
    return sorted_list[:, 1].astype(int)

rank_plays(0, term_document_matrix, compute_jaccard_similarity)


# ### rank_words

# In[11]:


def rank_words(target_word_index, matrix, similarity_fn):
    ''' Ranks the similarity of all of the words to the target word.

    Inputs:
    target_word_index: The index of the word we want to compare all others against.
    matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
    similarity_fn: Function that should be used to compared vectors for two word
      ebeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

    Returns:
    A length-n list of integer word indices, ordered by decreasing similarity to the 
    target word indexed by word_index
    '''

    # YOUR CODE HERE
    similarity = []
    for index in range(matrix.shape[0]):
        similarity.append([similarity_fn(get_row_vector(matrix, target_word_index), get_row_vector(matrix, index)), index])
    
    sorted_list = np.array(sorted(similarity, key=lambda x:x[0], reverse=True))
    # print(sorted_list)
    return sorted_list[:, 1].astype(int)

rank_words(0, term_context_matrix, compute_cosine_similarity)


# ### main program

# In[12]:


if __name__ == '__main__':
    tuples, document_names, vocab = read_in_shakespeare()

    print('Computing term document matrix...')
    td_matrix = create_term_document_matrix(tuples, document_names, vocab)

    print('Computing tf-idf matrix...')
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)

    print('Computing term context matrix...')
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

    print('Computing PPMI matrix...')
    PPMI_matrix = create_PPMI_matrix(tc_matrix)

    random_idx = random.randint(0, len(document_names)-1)
    similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
    for sim_fn in similarity_fns:
        print('\nThe top most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
        ranks = rank_plays(random_idx, td_matrix, sim_fn)
        for idx in range(0, 3):
            doc_id = ranks[idx]
            print('%d: %s' % (idx+1, document_names[doc_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on term-document frequency matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], td_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on tf_idf matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], tf_idf_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))

    word = 'juliet'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
        ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
        for idx in range(0, 10):
            word_id = ranks[idx]
            print('%d: %s' % (idx+1, vocab[word_id]))


# ### Discussion part
# ### cluster the plays

# In[13]:


from sklearn.cluster import KMeans

play_kmean = KMeans(n_clusters=3, random_state=0)
plays = []
for index in range(td_matrix.shape[1]):
    plays.append(get_column_vector(td_matrix, index))

play_kmean.fit(np.array(plays))

three_types = {0:[], 1:[], 2:[]}
for index in range(len(play_kmean.labels_)):
    three_types[play_kmean.labels_[index]].append(document_names[index])

three_types


# ### term_character_matrix

# In[14]:


name_line = []
character_names = {}
with open('will_play_text.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            character_name = row[4]
            if len(character_name) > 0:
                line = row[5]
                line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
                line_tokens = [token.lower() for token in line_tokens]
                character_names[character_name] = True

                name_line.append((character_name, line_tokens))

character_names = list(character_names.keys())
term_character_matrix = create_term_document_matrix(name_line, character_names, vocab)

print(term_character_matrix.shape)


# ### most/least similar characters

# In[15]:


from scipy import sparse

similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
for sim_fn in similarity_fns:
    print('using', sim_fn.__qualname__)
    try:
        similarity_matrix = sparse.load_npz("similarity_matrix_" + sim_fn.__qualname__ + ".npz").toarray()
    except:
        similarity_matrix = np.zeros((len(character_names), len(character_names)))
        
        for x in range(len(character_names)):
            for y in range(x + 1, len(character_names)):
                similarity_matrix[x][y] = similarity_matrix[y][x] = sim_fn(get_column_vector(term_character_matrix, x), get_column_vector(term_character_matrix, y))

        sparse.save_npz("similarity_matrix_" + sim_fn.__qualname__ + ".npz", sparse.csc_matrix(similarity_matrix))
    
    print('the most similar character pairs (A, B) are:')
    for index in range(len(similarity_matrix)):
        closest_id = np.argmax(similarity_matrix[index])
        print(character_names[index], '------', character_names[closest_id])


# ### cluster the characters

# In[16]:


character_kmean = KMeans(n_clusters=2, random_state=0)
characters = []
for index in range(term_character_matrix.shape[1]):
    characters.append(get_column_vector(term_character_matrix, index))

character_kmean.fit(np.array(plays))

two_types = {0:[], 1:[]}
for index in range(len(character_kmean.labels_)):
    two_types[character_kmean.labels_[index]].append(character_names[index])

two_types

