__author__ = 'haox1'
import os
import pandas as pd
import nltk.data
from nltk.tokenize import RegexpTokenizer
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
#from mlp import *
#import climate
import theano
import theano.tensor as TT
import theanets
import cPickle

def sentence_iterator():
    #create a iterator over all the sentences in corpus
    file_list = [i for i in os.listdir(os.getcwd()+'\data') if i.startswith('wc_adj')]
    #create a sentence splitter
    sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
    word_tokenizer = RegexpTokenizer(u'(?u)[a-zA-Z0-9/\-]{2,}')

    for file in file_list:
        df = pd.read_csv('data/'+file, compression='gzip')
        df.note_txt = df.note_txt.astype('str')
        j=[0,0]
        for note in df.note_txt:
            para_list = [i for i in note.split('\r\n') if i.endswith('.')]
            if para_list:
                j[0]+=1
                para_list = [i[i.find(':')+1:].lstrip() for i in para_list]
                for para in para_list:
                    sent_list = sentence_splitter.tokenize(para)
                    for sent in sent_list:
                        j[1]+=1
                        #print sent.lower().lstrip().rstrip()
                        #split into a list of words
                        sent = word_tokenizer.tokenize(sent.lower())
                        #print sent
                        yield sent
        print "In file: %r, read %r sentences from %r notes" %(file, j[1], j[0])

def train_word2vec(sentences):
    model = Word2Vec(sentences=sentences, size=100)
    model.save('model_word2vec.bin')
    print model.similarity('computer', 'network')


def transform_to_vec(model,word1, word2):
    """
    get two words as input, and transform it to a feature vector
    """
    vec1, vec2 = model[word1], model[word2]
    return np.concatenate((vec1, vec2, vec1*vec2, np.abs(vec1-vec2), vec1+vec2))

def transform_to_mat():
    """
    Load all the pairs in three txt files, transform it to matrix.
    X: features, shape of (num_instances, num_feat)
       features is a concatenation of x1, x2, (x1*x2), |x1 - x2|, (x1+x2),
    y: labels, shape of (num_instances)
    """
    model = gensim.models.Word2Vec.load('model_word2vec.bin')
    X = []
    y = []
    for file in ['synonyms.txt', 'antonyms.txt', 'irrelevent.txt']:
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

def svm(X_train, X_test, y_train, y_test):
    """
    SVM training, optimal C=0.2, f1_score: 0.31
    """
    clf = LinearSVC(C=0.5,verbose=1, class_weight='auto')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print f1_score(y_test, pred, average='micro')



def main():
    theano.config.optimizer='fast_compile'
    theano.config.optimizer_verbose=True
    theano.config.exception_verbosity='high'
    #l = sentence_iterator()
    #train_word2vec(l)
    X, y = transform_to_mat()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=3)
    cPickle.dump((X_train.astype(np.float64), X_test.astype(np.float64), y_train, y_test),open('word_mat.bin', 'wb'))
    #X_train, X_test, y_train, y_test = cPickle.load(open('word_mat.bin', 'rb'))
    #print X_train.shape, X_test.shape, y_train.shape, y_test.shape
    #svm(X_train, X_test, y_train, y_test)
    #X_train, y_train = X_train[:200], y_train[:200]
    #mlp_train(X_train, X_test, y_train, y_test)


main()