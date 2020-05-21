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
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score, roc_curve
import matplotlib.pylab as plt

class SVM(Machine_Learning):
    def __init__(self):
        pass

    def create_ml(self, C = 1, logGamma = -3, kernel = 'rbf'):
    #def create_ml(self, **kwarg):
        #C = kwarg['c']
        #logGamma = kwarg['loggamma']
        self.model = sklearn.svm.SVC(C=C, gamma = 10**logGamma, kernel = kernel) #, probability=True)

    def train(self, X, Y):
        self.model.fit(X, Y)

    def run(self, X):
        self.predictions = self.model.predict(X)

    def restore(self):
        pass
    
    def roc(self, y_test, x_test):
        fpr, tpr, thresholds = roc_curve(y_test, self.model.decision_function(x_test))
        plt.plot(fpr, tpr, '-', label="svm")
        plt.plot([0, 1], [0, 1], 'k--', label="random guess")
        #plt.plot([fallout], [recall], 'ro', ms=10)
        plt.xlabel('Fall-Out')
        plt.ylabel('Recall')
        plt.title('Receiver operating characteristic example')
        plt.show()
