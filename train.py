__author__ = 'Hao'

import cPickle
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix

def svm_train(X_train, X_test, y_train, y_test, C=1.0):
    #SVM training, optimal C=1.0, f1_score: 0.6834
    clf = LinearSVC(C=C, verbose=1, class_weight='auto')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print "Train f1 %.3f, Test f1: %.3f" %(f1_score(y_train, clf.predict(X_train)), f1_score(y_test, pred, average='micro'))
    print "Test Confusion Matrix: "
    print confusion_matrix(y_test, pred)
    return clf


def main():
    X_train, X_test, y_train, y_test = cPickle.load(open('word_mat_min.bin', 'rb'))
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    #for i in [0.1, 0.5, 1.0, 2.0, 5.0]:
    #    svm_train(X_train, X_test, y_train, y_test, C=i)

    clf = svm_train(X_train, X_test, y_train, y_test)
    #cPickle.dump(clf, open('svm.model', 'wb'))

main()