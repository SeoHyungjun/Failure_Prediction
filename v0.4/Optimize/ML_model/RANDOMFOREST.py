import os
import sys
from ML_model.base_ml import Machine_Learning

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import sklearn.ensemble 

class RANDOMFOREST(Machine_Learning):
    def __init__(self):
        pass

    def create_ml(self, n_estimators = 10, max_depth = 1, max_features = 2):
        print('n_estimators = ' + str(n_estimators) + ' max_depth = ' + str(max_depth) + ' max_features = ' + str(max_features))
        self.model = sklearn.ensemble.RandomForestClassifier(n_estimators = int(n_estimators), max_depth = int(max_depth), max_features = int(max_features))

    def train(self, X, Y):
        self.model.fit(X, Y)

    def run(self, X):
        self.predictions = self.model.predict(X)

    def restore(self):
        pass
