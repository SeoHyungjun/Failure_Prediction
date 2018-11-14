import sys
Optimization_function_path = '/Failure_Prediction/v0.4/Optimize/Optimization_function'
sys.path.insert(0, Optimization_function_path)

import PSO

class Hyperparameter_Tuner:
    def __init__(self, Opt_algorithm, Opt_hyperparameter, ML_algorithm):
        self.Opt_algorithm = Opt_algorithm
        self.Opt_hyperparameter = Opt_hyperparameter
        self.ML_algorithm = ML_algorithm

    def run_opt_algorithm(self):
        if self.Opt_algorithm == 'PSO':
            for ML_al in self.ML_algorithm.keys():
                #self.opt_function = 
                print('\nPSO ' + ML_al + ' train')
                print(self.ML_algorithm[ML_al])
                solver = PSO.make_PSO(self.Opt_hyperparameter, self.ML_algorithm[ML_al])


        elif self.Opt_algorithm == 'BO':
            for ML_al in self.ML_algorithm.keys():
                print('\nBO ' + ML_al + ' train')
                print(self.ML_algorithm[ML_al])
        

