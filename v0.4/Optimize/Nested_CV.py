import optunity
import optunity.metrics
from ML_model import SVM
from Optimization_function import PSO

from sklearn.metrics import f1_score

def nested_cv(solver, ML_algorithm, ML_hyperparameter, data, labels):

    @optunity.cross_validated(x=data, y=labels, num_folds=5)
    def nested_crossvalidation(x_train, y_train, x_test, y_test, ML_algorithm, ML_hyperparameter, solver):

        @optunity.cross_validated(x=x_train, y=y_train, num_folds=2, num_iter=1)
        def inner_cv(x_train, y_train, x_test, y_test, c, loggamma):
            #model = sklearn.svm.SVC(C = ML_hyperparameter['c'], logGamma = ML_hyperparameter['loggamma'])
            if ML_algorithm == 'SVM':
                model = SVM.make_SVM(C = c, logGamma = loggamma).fit(x_train, y_train)
            #predictions = model.decision_function(x_test)
            predictions = model.predict(x_test)
            #return optunity.metrics.roc_auc(y_test, predictions)
            if ((True in predictions) and (False in predictions)):            
                inner_fscore = f1_score(predictions, y_test)
                #print(inner_fscore)
                return inner_fscore
            else:
                return 0.0


        # 'Crossvalidation.py' will using the solver that was made at HyperParameter_Tuner
        #solver = PSO.make_PSO(self.Opt_hyperparameter, self.ML_hyperparameter)
        hpars, info = solver.maximize(inner_cv)
        
        #model = SVM.make_SVM(C = hpars['c'], loggamma = hpars['loggamma']).fit(x_train, y_train)
       #print(hpars)
        model = SVM.make_SVM(C = hpars['c'], logGamma = hpars['loggamma']).fit(x_train, y_train)
        #predictions = model.decision_function(x_test)
        predictions = model.predict(x_test)
        #return optunity.metrics.roc_auc(y_test, predictions)
        nested_fscore = f1_score(y_test, predictions)
        print('f1-score = ' + str(nested_fscore))
        print(hpars)
        return nested_fscore

    best_f1_score = nested_crossvalidation(ML_algorithm = ML_algorithm, ML_hyperparameter = ML_hyperparameter, solver = solver)
    return best_f1_score
