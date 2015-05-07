__author__ = 'haox1'
from nltk.corpus import wordnet as wn
import random
import pandas as pd
import numpy as np
import gensim
import cPickle
from sklearn.cross_validation import train_test_split

"""
This file is used to create list of synonyms, antonyms and irrelevant word pairs
"""

def generate_synonyms(file_name='synonyms_max.txt'):
    #generate 44k pairs of synonyms and save to file_name
    f = open(file_name, 'wb')
    count = 0
    for i in wn.all_synsets():
        if len(i.lemma_names()) > 1:
            """
            current_word = i.lemma_names()[0]
            if '_' in current_word:
                continue
            for word in i.lemma_names()[1:]:
                if '_' in word:
                    continue
            """
            l = i.lemma_names()
            for i in range(len(l)):
                for j in range(i+1, len(l)):
                    if '_' not in l[i] and '_' not in l[j]:
                        f.write(l[i]+', '+l[j]+'\n')
                        count+=1
    print count
    f.close()


def generate_antonyms(file_name='antonyms.txt'):
    # generate 3.2k pairs of antonyms and save to file_name
    antonym_list=[]
    for i in wn.all_synsets():
        for m in i.lemmas():
            if len(m.antonyms())>0:
                for j in m.antonyms():
                    if j.name()<m.name():
                        antonym_list.append((j.name(), m.name()))
                    else:
                        antonym_list.append((m.name(), j.name()))
                    print j.name(), m.name()
    antonym_list = set(antonym_list)
    f = open(file_name, 'wb')
    count = 0
    for i in antonym_list:
        if '_' in i[0] or '_' in i[1]:
            continue
        f.write(i[0]+', '+i[1]+'\n')
        count+=1
    print count
    f.close()

def generate_irrelevant(file_name='irrelevent_min.txt'):
    #generate irrelevant word pairs, will be used as negative label
    all_lemma = [i for i in wn.all_lemma_names() if '_' not in i]
    length = len(all_lemma)
    count = 0
    f = open(file_name, 'wb')
    for i in wn.all_synsets():
        m = len(i.lemma_names())/3+1
        for j in range(m):
            current_word = i.lemma_names()[random.randint(0, len(i.lemma_names())-1)]
            if '_' not in current_word:
                for j in range(2):
                    word = all_lemma[random.randint(0,length-10)]
                    if word not in i.lemma_names():
                        count+=1
                        f.write(current_word+', '+word+'\n')
    print count
    f.close()

def test_in_corpus(model):
    #test whether the two words are in the word2vec dict
    for file in ['synonyms_max.txt', 'synonyms.txt', 'antonyms.txt', 'irrelevent.txt']:
        f = open(file, 'r')
        j=[0,0]
        for i in f:
            j[0]+=1
            word1, word2 = i.strip().split(', ')
            if word1 in model and word2 in model:
                j[1]+=1
        f.close()
        print "In %r, %r of %r are in the corpus.\n" %(file, j[1], j[0])


def save_word2vec():
    word2vec_dict = {}
    with open('data\\vectors.6B.50d.txt') as f:
        for line in f.readlines():
            line = line.rstrip().split()
            word = line[0]
            vec = np.asarray([float(i) for i in line[1:]])
            word2vec_dict[word] = vec
    g = open('data\word2vec_dict.bin', 'wb')
    cPickle.dump(word2vec_dict, g)
    g.close()

def transform_to_vec(model,word1, word2):
    #get two words as input, and transform it to a feature vector
    vec1, vec2 = model[word1], model[word2]
    return np.concatenate((vec1, vec2, vec1*vec2, np.abs(vec1-vec2), vec1+vec2))

def transform_to_mat(model):
    """
    Load all the pairs in three txt files, transform it to matrix.
    X: features, shape of (num_instances, num_feat)
       features is a concatenation of x1, x2, (x1*x2), |x1 - x2|, (x1+x2),
    y: labels, shape of (num_instances)
    """
    X = []
    y = []
    for file in ['synonyms.txt', 'antonyms.txt', 'irrelevent_min.txt']:
        f = open(file, 'r')
        j=[0,0]
        for i in f:
            j[0]+=1
            word1, word2 = i.strip().split(', ')
            if word1 in model and word2 in model:
                j[1]+=1
                X.append(transform_to_vec(model, word1, word2))
                if file == 'synonyms.txt':
                    y.append(1)
                else:
                    y.append(-1)
        f.close()
        print "In %r, %r of %r are in the corpus.\n" %(file, j[1], j[0])
    X, y = np.array(X), np.array(y)
    print X.shape
    print y.shape
    return X, y


def main():
    #save_word2vec()
    #generate_antonyms()
    #generate_synonyms()
    generate_irrelevant()
    word2vec_dict = cPickle.load(open('data\word2vec_dict.bin', 'rb'))
    #test_in_corpus(word2vec_dict)

    X, y = transform_to_mat(word2vec_dict)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=3)
    cPickle.dump((X_train.astype(np.float64), X_test.astype(np.float64), y_train, y_test),open('word_mat_min.bin', 'wb'))

main()