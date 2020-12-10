'''
import sklearn.svm

def make_SVM(C = 1, logGamma = -3 ):
    
    return sklearn.svm.SVC(C=C, gamma=10**logGamma)
'''
import os
import sys
from ML_model.base_ml import Machine_Learning

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import sklearn.svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score, roc_curve
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

class SVM(Machine_Learning):
    def __init__(self):
        pass

    def create_ml(self, C = 1, logGamma = -3, kernel = 'rbf'):
    #def create_ml(self, C = 25.57177734375, logGamma = 3.7646484375, kernel = 'rbf'):
    #def create_ml(self, **kwarg):
        #C = kwarg['c']
        #logGamma = kwarg['loggamma']
        while C < 0:
            C = C + 0.1
        self.model = sklearn.svm.SVC(C=C, gamma = 10**logGamma, kernel = kernel, probability=True)

    def train(self, X, Y):
        #self.model = OneVsRestClassifier(self.model, n_jobs=-1).fit(X, Y)
        self.model.fit(X, Y)

    def run(self, X):
        self.predictions = self.model.predict(X)

    def restore(self):
        pass
    
    def roc(self, y_test, x_test):
        
        '''
        self.fpr, self.tpr, thresholds = roc_curve(y_test, self.model.decision_function(x_test))
        plt.plot(self.fpr, self.tpr, '-', label="svm")
        plt.plot([0, 1], [0, 1], 'k--', label="random guess")
        #plt.plot([fallout], [recall], 'ro', ms=10)
        plt.xlabel('Fall-Out')
        plt.ylabel('Recall')
        plt.title('Receiver operating characteristic example')
        plt.show()
        '''
        
        '''
        self.fpr, self.tpr, thresholds = roc_curve(y_test, self.model.predict_proba(x_test)[:, 1])
        plt.plot(self.fpr, self.tpr, '-', label="svm")
        plt.plot([0, 1], [0, 1], 'k--', label="random guess")
        #plt.plot([fallout], [recall], 'ro', ms=10)
        plt.xlabel('Fall-Out')
        plt.ylabel('Recall')
        plt.title('Receiver operating characteristic example')
        plt.show()
        '''
        self.fpr, self.tpr, thresholds = precision_recall_curve(y_test, self.model.predict_proba(x_test)[:, 1])
        plt.plot(self.tpr, self.fpr, '-', label="svm")
        plt.plot([0, 1], [0, 1], 'k--', label="random guess")
        #plt.plot([fallout], [recall], 'ro', ms=10)
        plt.xlabel('Fall-Out')
        plt.ylabel('Recall')
        plt.title('Receiver operating characteristic example')
        plt.show()
        
    def roc_store(self):
        pd.DataFrame(self.fpr).to_csv("roc/prc/1day/svm_linear_precision.csv", mode = 'a', header=None, index=None)
        pd.DataFrame(self.tpr).to_csv("roc/prc/1day/svm_linear_recall.csv", mode = 'a', header=None, index=None)
        '''
        pd.DataFrame(self.fpr).to_csv("roc/1day/svm_rbf_roc_fpr.csv", mode = 'a', header=None, index=None)
        pd.DataFrame(self.tpr).to_csv("roc/1day/svm_rbf_roc_tpr.csv", mode = 'a', header=None, index=None)
        '''
        