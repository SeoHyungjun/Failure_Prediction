import sys
from Optimization_function import PSO

import optunity  
import Nested_CV

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_digits
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score, roc_curve
#import random

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import optunity
import optunity.metrics

import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ML_model import SVM, RANDOMFOREST

from sklearn.feature_selection import RFE
import sklearn.ensemble
import sklearn.svm

'''
digits = load_digits()
n = digits.data.shape[0]
positive_digit = 6
negative_digit = 8
positive_idx = [i for i in range(n) if digits.target[i] == positive_digit]
negative_idx = [i for i in range(n) if digits.target[i] == negative_digit]
# add some noise to the data to make it a little challenging
original_data = digits.data[positive_idx + negative_idx, ...]
data = original_data + 5 * np.random.randn(original_data.shape[0], original_data.shape[1])
labels = [True] * len(positive_idx) + [False] * len(negative_idx)
'''
'''
print(data)
print(labels)
'''
class Hyperparameter_Tuner:
    def __init__(self, Opt_algorithm, Opt_hyperparameter, ML_algorithm):
        self.Opt_algorithm = Opt_algorithm
        self.Opt_hyperparameter = Opt_hyperparameter
        self.ML_algorithm = ML_algorithm

        #self.iris_dataset = load_iris()
        #self.iris_x, self.iris_X_test, self.iris_y, self.iris_y_test = train_test_split(self.iris_dataset['data'], self.iris_dataset['target'], random_state=0)

    def load_data(self):
        '''
        failure_data = pd.read_csv("../data/h_failure/failure_96.csv", header = None, index_col=False)
        normal_data = pd.read_csv("../data/random_4/normal_4_data.csv", header = None, index_col=False)
        normal_data = (shuffle(normal_data)).iloc[:len(failure_data)]

        sum_data = pd.concat([failure_data, normal_data])
        sum_data = shuffle(sum_data)
        
        data = sum_data.iloc[:, 2:14]
        labels = sum_data.iloc[:, 1:2]
        
        self.data = data.values.tolist()
        self.labels = labels[1].tolist()
        print(len(self.data))
        print(len(self.labels))
        print('load end')
        '''
        
        #failure_data = pd.read_csv("../data/backblaze/backblaze_failure/failure_4days.csv",  index_col=False)
        failure_data = pd.read_csv("../data/backblaze/backblaze_failure/1819_12model_failure_4days.csv", index_col=False)
        #normal_data = pd.read_csv("../data/backblaze/backblaze_failure/failure_4days.csv", index_col=False)
        #normal_data = pd.read_csv("../data/backblaze/backblaze_normal/normal.csv", index_col=False)
        normal_data = pd.read_csv("../data/backblaze/backblaze_normal/12_select_1000000_normal.csv", index_col=False)
        normal_data = (shuffle(normal_data)).iloc[:len(failure_data)]

        sum_data = pd.concat([failure_data, normal_data])
        sum_data = shuffle(sum_data).fillna(0)
        sum_data.reset_index(inplace = True)
        #sum_data.to_csv('/Failure_Prediction/v0.4/data/backblaze/backblaze_failure/train.csv' , mode = 'w', index=False)
        
        test_list2 = ['smart_1_normalized', 'smart_1_raw', 
                     'smart_5_normalized', 'smart_5_raw', 
                     'smart_187_normalized', 'smart_187_raw',
                     'smart_189_normalized', 'smart_189_raw', 'smart_194_normalized', 'smart_194_raw',
                     'smart_197_normalized', 'smart_197_raw']
        
        test_list3 = ['smart_1_normalized', 'smart_5_raw', 'smart_9_normalized',
                     'smart_189_normalized',
                     'smart_197_normalized', 'smart_197_raw']
        
        test_list = ['smart_5_normalized', 'smart_5_raw',
                     'smart_187_normalized', 'smart_187_raw',
                     'smart_189_normalized', 'smart_189_raw',
                     'smart_197_normalized', 'smart_197_raw']
        plot_list = ['failure', 
                     'smart_5_normalized', 'smart_5_raw',
                     'smart_187_normalized', 'smart_187_raw',
                     'smart_189_normalized', 'smart_189_raw']
        
        '''
        test_list = ['smart_1_normalized', 'smart_1_raw', 'smart_3_normalized',
                     'smart_3_raw', 'smart_4_normalized', 'smart_4_raw',
                     'smart_5_normalized', 'smart_5_raw', 'smart_7_normalized',
                     'smart_7_raw', 'smart_9_normalized', 'smart_9_raw',
                     'smart_10_normalized', 'smart_10_raw', 'smart_12_normalized',
                     'smart_12_raw', 'smart_183_normalized', 'smart_183_raw',
                     'smart_184_normalized', 'smart_184_raw', 'smart_187_normalized',
                     'smart_187_raw', 'smart_188_normalized', 'smart_188_raw',
                     'smart_189_normalized', 'smart_189_raw', 'smart_190_normalized',
                     'smart_190_raw', 'smart_191_normalized', 'smart_191_raw'
                     'smart_192_normalized', 'smart_192_raw', 'smart_193_normalized',
                     'smart_193_raw', 'smart_194_normalized', 'smart_194_raw',
                     'smart_197_normalized', 'smart_197_raw', 'smart_198_normalized',
                     'smart_198_raw', 'smart_199_normalized', 'smart_199_raw',
                     'smart_240_normalized', 'smart_240_raw', 'smart_241_normalized',
                     'smart_241_raw', 'smart_242_normalized', 'smart_242_raw']
        '''
        
        '''
        test_list = [5, 13, 63, 79, 83,
                    14, 84,
                    9, 11, 15, 19, 21]
        '''
        
        sum_data = sum_data.iloc[:, 4:]
        #data = sum_data.iloc[:, 1:]
        data = sum_data.loc[:, test_list2]
        #print(data)
        '''
        df = data
        x = df.values.astype(float)
        x_scaled = MinMaxScaler().fit_transform(x)
        data = pd.DataFrame(x_scaled, columns = df.columns)
        print(data)
        '''

        print(data.columns)
        self.header = data.columns
        
        
        labels  = sum_data.loc[:, ['failure']]
        #labels = sum_data.iloc[:, 0:1]
        #labels.loc[labels['failure'] == 0, 'failure'] = -1
        #print(labels)
        
        self.data = data #.values.tolist()
        self.SVM_data = self.data.values.tolist()
        self.RF_data = self.data.values.tolist()
        self.labels = labels['failure'].tolist()
        print(len(self.data))
        print(len(self.labels))
        print('load end')
        
        #데이터 그림그리는부분
        '''
        plot_data = sum_data.loc[:, test_list]
        sns.set(style="ticks", color_codes=True)
        sns.pairplot(plot_data, hue="failure", palette="husl", hue_order= [1, 0])
        #sns.pairplot(plot_data, palette="husl")
        plt.show()
        '''
    
    def run_feature_selection(self):    
        print(list(self.ML_algorithm.keys()))
        selected_data = self.data
        next_data = selected_data.values.tolist()
        labels = self.labels
        
        def cv(next_data, labels, fold, ML_algo):
            #print(len(next_data))
            
            @optunity.cross_validated(x=next_data, y=labels, num_folds=fold)
            def nested_crossvalidation(x_train, y_train, x_test, y_test, ML_algo):
                model = SVM.SVM()
                if ML_algo == 'SVM':
                    model = SVM.SVM()
                elif ML_algo == 'RANDOMFOREST':
                    model = RANDOMFOREST.RANDOMFOREST()

                model.create_ml()
                model.train(x_train, y_train)
                model.run(x_test)
                return accuracy_score(model.predictions, y_test)
            
            return nested_crossvalidation(ML_algo)
        
        
        list_ML_algo = list(self.ML_algorithm.keys())
        for ML_algo in list_ML_algo:
            #score에 1개의 변수들을 사용했을때의 성능을 저장, 이유는 특정 ML에서는 feature가 너무 많으면 성능이 안나올때가 있어서
            #중요한 feature(장애 예측에 도움이 되는)가 제외되는것을 방지하기 위해서
            score_feature = {}
            list_feature = list(self.header)
            
            for feature in list_feature:
                next_data = selected_data.loc[:, [feature]]
                #print(next_data.columns)
                next_data = next_data.values.tolist()
                score_feature[feature] = cv(next_data, labels, 3, ML_algo)
                
            delete_feature = []
            for key, value in score_feature.items():
                print('key = {}, b = {}' .format(key, value) )
                if value < 0.51:
                    delete_feature.append(key)
           
            print('delete feature = ' + str(delete_feature))
            
            leave_feature = list_feature
            num_delete_feature = len(delete_feature)
            for i in range(num_delete_feature):
                score_feature = {}
                next_data = selected_data.loc[:, leave_feature].values.tolist()
                prev_acc = cv(next_data, labels, 3, ML_algo)
                #print('prev_acc = {}'.format(prev_acc))
                for feature in delete_feature:
                    leave_feature.remove(feature)
                    #print(leave_feature)
                    next_data = selected_data.loc[:, leave_feature].values.tolist()
                    next_acc = cv(next_data, labels, 3, ML_algo)
                    #print('drop feature = {} -> next_acc = {}'.format(feature, next_acc))

                    score_feature[feature] = next_acc
                    leave_feature.append(feature)
                
                #print(score_feature)
                max_key = max(score_feature.keys(), key=(lambda k: score_feature[k]))
                if( score_feature[max_key] > prev_acc):
                    leave_feature.remove(max_key)
                    delete_feature.remove(max_key)
                #print('\nleave feature = {}\n'.format(leave_feature))
            
            lenleave = len(leave_feature)
            for feature in range(lenleave):
                score_feature = {}
                next_data = selected_data.loc[:, leave_feature].values.tolist()
                next_acc = cv(next_data, labels, 3, ML_algo)
                for feature in leave_feature:
                    leave_feature.remove(feature)
                    next_data = selected_data.loc[:, leave_feature].values.tolist()
                    next_acc = cv(next_data, labels, 3, ML_algo)
                    
                    score_feature[feature] = next_acc
                    leave_feature.append(feature)
                    
                max_key = max(score_feature.keys(), key=(lambda k: score_feature[k])) 
                if( score_feature[max_key] > prev_acc):
                    leave_feature.remove(max_key)

            print(leave_feature)
            
            if ML_algo == 'SVM':
                self.SVM_data = self.data.loc[:, leave_feature].values.tolist()
            elif ML_algo == 'RF':
                self.RF_data = self.data.loc[:, leave_feature].values.tolist()
        
        '''
        for ML_algo in list(self.ML_algorithm.keys()):
            selected_data = self.data
            next_data = selected_data.values.tolist()
            
            for feature in self.header:
                next_data = selected_data.values.tolist()
                prev_acc = cv(next_data, labels, 3, ML_algo)
                
                next_data = selected_data.drop(feature, axis = 1).values.tolist()
                next_acc = cv(next_data, labels, 3, ML_algo)
                
                print('prev acc = ' + str(prev_acc) +' , next_acc = ' + str(next_acc))
                
                if next_acc >= prev_acc:
                    print(feature + ' drop')
                    selected_data = selected_data.drop(feature, axis = 1)
                    #print(type(selected_data))
            
            if ML_algo == 'SVM':
                self.SVM_data = selected_data.values.tolist()
            elif ML_algo == 'RANDOMFOREST':
                self.RF_data = selected_data.values.tolist()
         '''
    '''
    def run_feature_selection(self):
        for ML_algo in list(self.ML_algorithm.keys()):
            if ML_algo == 'SVM':
                model = sklearn.svm.SVC(kernel = 'linear')
            elif ML_algo == 'RF':
                model = sklearn.ensemble.RandomForestClassifier()
        
            model.fit(self.data.values.tolist(), self.labels)
            rfe = RFE(model, 8)
            fit = rfe.fit(self.data.values.tolist(), self.labels)
            print("Num Features: " + str(fit.n_features_))  
            print("Selected Features: " +  str(fit.support_))
            print("Feature Ranking: " + str(fit.ranking_))
    '''          
                
    def run_opt_algorithm(self):
        if self.Opt_algorithm == 'PSO':
            for ML_al in self.ML_algorithm.keys():
                kernel_list = []
                #self.opt_function = 
                print('\nPSO ' + ML_al + ' train')
                #print(self.ML_algorithm[ML_al])
                send_data = self.SVM_data
                if ML_al == 'SVM':
                    kernel_list = self.ML_algorithm[ML_al]['kernel']
                    self.ML_algorithm[ML_al].pop('kernel')
                    send_data = self.SVM_data
                elif ML_al == 'RANDOMFOREST':
                    send_data = self.RF_data
                    
                self.solver = PSO.make_PSO(self.Opt_hyperparameter, self.ML_algorithm[ML_al])
                print(self.ML_algorithm[ML_al])
                
                Opt_PSO_best = Nested_CV.nested_cv(self.solver, ML_al, self.ML_algorithm[ML_al], send_data, self.labels, kernel_list, self.Opt_hyperparameter)
                print('Opt_PSO_best_avg = ' + str(Opt_PSO_best))
                #self.nested_crossvalidation(self, ML_algorithm = ML_al, ML_hyperparameter = self.ML_algorithm[ML_al])

        elif self.Opt_algorithm == 'BO':
            for ML_al in self.ML_algorithm.keys():
                print('\nBO ' + ML_al + ' train')
                print(self.ML_algorithm[ML_al])
