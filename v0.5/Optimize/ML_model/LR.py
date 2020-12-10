import os
import sys
from ML_model.base_ml import Machine_Learning

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import sklearn.ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

class LR(Machine_Learning):
    def __init__(self):
        pass

    '''
    def create_ml(self, log_var_smoothing = -9):
        #print('n_estimators = ' + str(n_estimators) + ' max_depth = ' + str(max_depth) + ' max_features = ' + str(max_features)
        self.model = GaussianNB(var_smoothing = 10**log_var_smoothing)
    '''
    def create_ml(self, log_tol = -4, C = 1.0):
        while(C < 0):
            C = 0
        self.model = LogisticRegression(tol = 10**log_tol, C = C, max_iter = 500)
    
    def train(self, X, Y):
        self.model.fit(X, Y)

    def run(self, X):
        self.predictions = self.model.predict(X)

    def restore(self):
        pass

    def roc(self, y_test, x_test):
        '''
        #fpr, tpr, thresholds = roc_curve(y_test, self.model.predict_proba(x_test)[:, 1])
        self.fpr, self.tpr, thresholds = roc_curve(y_test, self.model.decision_function(x_test))
        plt.plot(self.fpr, self.tpr, '-', label="lr")
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
        pd.DataFrame(self.fpr).to_csv("roc/prc/1day/lr_precision.csv", mode = 'a', header=None, index=None)
        pd.DataFrame(self.tpr).to_csv("roc/prc/1day/lr_recall.csv", mode = 'a', header=None, index=None)
        '''
        pd.DataFrame(self.fpr).to_csv("roc/1day/lr_roc_fpr.csv", mode = 'a', header=None, index=None)
        pd.DataFrame(self.tpr).to_csv("roc/1day/lr_roc_tpr.csv", mode = 'a', header=None, index=None)
        '''