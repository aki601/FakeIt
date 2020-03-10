import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from nltk.corpus import stopwords
from sklearn.metrics import classification_report,roc_curve,auc, accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pylab as pl
import os
import warnings
from sklearn.model_selection import KFold
import scikitplot.plotters as skplt
from scipy.stats import f_oneway
warnings.filterwarnings('ignore')

AUC_NB = []
AUC_SVM = []
AUC_LR = []

def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)


def cleanup(text):
    text = textClean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def constructLabeledSentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences


def getEmbeddings(path,vector_dimension=300):
    data = pd.read_csv(path)

    missing_rows = []
    for i in range(len(data)):
        if data.loc[i, 'text'] != data.loc[i, 'text']:
            missing_rows.append(i)
    data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

    for i in range(len(data)):
        data.loc[i, 'text'] = cleanup(data.loc[i,'text'])

    x = constructLabeledSentences(data['text'])
    y = data['label'].values

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10,seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.iter)
    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size
    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        train_labels[i] = y[i]

    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        test_labels[j] = y[i]
        j = j + 1

    return text_train_arrays, text_test_arrays, train_labels, test_labels

#if not os.path.isfile('./xtrain.npy') or \
#    not os.path.isfile('./xtest.npy') or \
#    not os.path.isfile('./ytrain.npy') or \
#    not os.path.isfile('./ytest.npy'):
#    xtrain,xtest,ytrain,ytest = getEmbeddings("data.csv")
#    np.save('./xtrain', xtrain)
#    np.save('./xtest', xtest)
#    np.save('./ytrain', ytrain)
#    np.save('./ytest', ytest)
#
xtrain = np.load('./xtrain.npy')
xtest = np.load('./xtest.npy')
ytrain = np.load('./ytrain.npy')
ytest = np.load('./ytest.npy')

X = np.concatenate((xtrain, xtest))
Y = np.concatenate((ytrain,ytest))

def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte,ypred)
    plt.show()
    
#SVM
def SupportVectorC():
    kf = KFold(n_splits=5)
    clfsvm = SVC(probability=True)
    y_pred = 0
    for train_index, test_index in kf.split(X):
        X_train ,X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        clfsvm.fit(X_train, y_train)
        y_pred = clfsvm.predict(X_test)
        prob_svm = clfsvm.predict_proba(X_test)
        AUC_NB.append(roc_auc_score(y_test, prob_svm[:, 1]))
        print("AUC for SVM: ", AUC_NB)

    print("\nAccuracy:",accuracy_score(y_test,y_pred))
    print("\nClassification Report is:")
    print(classification_report(y_test,y_pred))
    
    fpr,tpr,threshold = roc_curve(y_test,prob_svm[:,1])

    pl.clf()
    pl.plot(fpr, tpr)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiverrating characteristic example')
    pl.legend(loc="lower right")
    pl.show()
    
    plot_cmat(y_test, y_pred) 

#    plot_cmat(ytest, y_pred)
    
def Logistic():
    LR = LogisticRegression()
    LR.fit(xtrain,ytrain)
    y_pred = LR.predict(xtest[1].reshape(1,-1))
    print(y_pred)

def NaiveBayes():
    kf = KFold(n_splits=5)
    clfgnb = GaussianNB()
    y_pred = 0
    for train_index, test_index in kf.split(X):
        X_train ,X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        clfgnb.fit(X_train, y_train)
        y_pred = clfgnb.predict(X_test)
        prob_nb = clfgnb.predict_proba(X_test)
        AUC_NB.append(roc_auc_score(y_test, prob_nb[:, 1]))
        print("AUC for NB: ", AUC_NB)

    print("\nAccuracy:",accuracy_score(y_test,y_pred))
    print("\nClassification Report is:")
    print(classification_report(y_test,y_pred))
    
    fpr,tpr,threshold = roc_curve(y_test,prob_nb[:,1])

    pl.clf()
    pl.plot(fpr, tpr)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiverrating characteristic example')
    pl.legend(loc="lower right")
    pl.show()
    
    plot_cmat(y_test, y_pred) 
    
def KLogistic():
    kf = KFold(n_splits=5)
    clfLR = LogisticRegression()
    y_pred = 0
    for train_index, test_index in kf.split(X):
        X_train ,X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        clfLR.fit(X_train, y_train)
        y_pred = clfLR.predict(X_test)
        prob_LR = clfLR.predict_proba(X_test)
        AUC_LR.append(roc_auc_score(y_test, prob_LR[:, 1]))
        print("AUC for Logistic: ", AUC_LR)
    
    print("\nAccuracy:",accuracy_score(y_test,y_pred))
    print("\nClassification Report is:")
    print(classification_report(y_test,y_pred))
    
    fpr,tpr,threshold = roc_curve(y_test,prob_LR[:,1])

    pl.clf()
    pl.plot(fpr, tpr)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiverrating characteristic example')
    pl.legend(loc="lower right")
    pl.show()
    plot_cmat(y_test, y_pred) 

    
if __name__ == "__main__":
    KLogistic()
#    NaiveBayes()
##    SupportVectorC()
#    ANOVA = f_oneway(AUC_LR, AUC_NB, AUC_SVM)
#    if ANOVA.statistic > 1:
#        print("Models are significantly different")
#    else:
#        print("Models are approximately same")