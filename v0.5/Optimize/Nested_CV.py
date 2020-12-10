import os
import sys

import optunity
import optunity.metrics
from Optimization_function import PSO

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ML_model import SVM, RANDOMFOREST, GNB, LR

from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score, roc_curve
import matplotlib.pylab as plt

#kernel_list = []
#kernel = ''
#fp = open('result.txt', 'a')
#fp2 = open('result2.txt', 'a')

#out_fold= 5
#in_fold = 2
#particle = 4
#gen = 5

out_fold_count = 0
in_fold_count = 0
particle_count = 0
gen_count = 0

in_fold_sum = 0

#global result_gen = []
#global result_particle = []
#global result_fold = []

#res = [[[0 for i in range(particle)] for j in range(gen)] for k in range(out_fold)]
res = []

def nested_cv(solver, ML_algorithm, ML_hyperparameter, data, labels, kernel_list, Opt_hyperparameter):
    global out_fold_count
    global in_flod_count
    global particle_count
    global gen_count
    global in_fold_sum
    global res

    out_fold_count = 0
    in_fold_count = 0
    particle_count = 0
    gen_count = 0

    in_fold_sum = 0

    res = []
    fp = open('result.txt', 'a')
    fp2 = open('result2.txt', 'a')

    #kernel_list = kernel_list2
    print(Opt_hyperparameter)
    out_fold = int(Opt_hyperparameter['num_out_fold'])
    in_fold = int(Opt_hyperparameter['num_in_fold'])
    particle = int(Opt_hyperparameter['num_particles'])
    gen = int(Opt_hyperparameter['num_generations'])
    res = [[[0 for i in range(particle)] for j in range(gen)] for k in range(out_fold)]

    @optunity.cross_validated(x=data, y=labels, num_folds=out_fold)
    def nested_crossvalidation(x_train, y_train, x_test, y_test, ML_algorithm, ML_hyperparameter, solver, kernel_list):
        ker = ''
            
        @optunity.cross_validated(x=x_train, y=y_train, num_folds=in_fold, num_iter=1)
        def SVM_inner_cv(x_train, y_train, x_test, y_test, c , loggamma):
            #model = sklearn.svm.SVC(C = ML_hyperparameter['c'], logGamma = ML_hyperparameter['loggamma']
            model = SVM.SVM()
            #print('create_ml')
            model.create_ml(C = c, logGamma = loggamma, kernel = kernel)
            #print('train')
            model.train(x_train, y_train)
            #predictions = model.decision_function(x_test)
            #predictions = model.predict(x_test)
            #model.roc(y_test, x_test)
            #print('run')
            model.run(x_test)
            #return optunity.metrics.roc_auc(y_test, predictions)
            print('c : ' + str(c) + ', loggamma : ' + str(loggamma) + ', kernel : ' + str(kernel))
            fp.write('c : ' + str(c) + ', loggamma : ' + str(loggamma) + ', kernel : ' + str(kernel) + '\n')
            
            if ((True in model.predictions) and (False in model.predictions)) or ((1 in model.predictions) and (-1 in model.predictions)):
                inner_fscore = f1_score(model.predictions, y_test)
                
                print('accuracy', accuracy_score(model.predictions, y_test))
                #roc 그림 그리는 함수
                #model.roc(y_test, x_test)
                
                #model.decision_function(x_test)
                #print(confusion_matrix(y_test, model.predictions, labels=[-1, 1]))
                #return inner_fscore
            else:
                inner_fscore = 0.0
                #print('The model did not predict correctly.')
            print(inner_fscore)
            fp.write(str(inner_fscore) + '\n')
            
            global out_fold_count
            global in_fold_count
            global particle_count
            global gen_count
             
            global in_fold_sum

            if in_fold_count >= in_fold - 1:
                in_fold_sum = in_fold_sum + inner_fscore
                print(str(out_fold_count) + '  ' +  str(gen_count) + '  ' + str(particle_count))
                res[out_fold_count][gen_count][particle_count] = in_fold_sum/in_fold
                in_fold_sum = 0
                in_fold_count = 0
                particle_count = particle_count + 1
                if particle_count >= particle:
                    particle_count = 0
                    gen_count = gen_count + 1
                    if gen_count >= gen:
                        gen_count = 0
                        out_fold_count = out_fold_count + 1
            else:
                in_fold_sum = in_fold_sum + inner_fscore
                in_fold_count = in_fold_count + 1
                

            return inner_fscore

        #@optunity.cross_validated(x=x_train, y=y_train, num_folds=2, num_iter=1)
        @optunity.cross_validated(x=x_train, y=y_train, num_folds=in_fold, num_iter=1)
        def RANDOMFOREST_inner_cv(x_train, y_train, x_test, y_test, n_estimators, max_depth, max_features):
            if max_depth < 0:
                max_depth = 1
            #print('max_depth = ' + str( max_depth))
            model = RANDOMFOREST.RANDOMFOREST()
            model.create_ml(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
            model.train(x_train, y_train)
            
            #model.roc(y_test, x_test)
            
            model.run(x_test)

            if ((True in model.predictions) and (False in model.predictions)) or ((1 in model.predictions) and (-1 in model.predictions)):
                inner_fscore = f1_score(model.predictions, y_test)
                #print(inner_fscore)
                #return inner_fscore
            else:
                inner_fscore = 0.0
            print(inner_fscore)
            fp.write(str(inner_fscore) + '\n')

            global out_fold_count
            global in_fold_count
            global particle_count
            global gen_count
            global in_fold_sum

            if in_fold_count >= in_fold - 1:
                in_fold_sum = in_fold_sum + inner_fscore
                res[out_fold_count][gen_count][particle_count] = in_fold_sum/in_fold
                print(str(out_fold_count) + '  ' +  str(gen_count) + '  ' + str(particle_count))
                in_fold_sum = 0
                in_fold_count = 0
                particle_count = particle_count + 1
                if particle_count >= particle:
                    particle_count = 0
                    gen_count = gen_count + 1
                    if gen_count >= gen:
                        gen_count = 0
                        out_fold_count = out_fold_count + 1
            else:
                in_fold_sum = in_fold_sum + inner_fscore
                in_fold_count = in_fold_count + 1

            return inner_fscore

        @optunity.cross_validated(x=x_train, y=y_train, num_folds=in_fold, num_iter=1)
        def CNN_inner_cv(x_train, y_train, x_test, y_test):
            pass

        @optunity.cross_validated(x=x_train, y=y_train, num_folds=in_fold, num_iter=1)
        def GNB_inner_cv(x_train, y_train, x_test, y_test, log_var_smoothing):
            model = GNB.GNB()
            model.create_ml(log_var_smoothing = log_var_smoothing)
            model.train(x_train, y_train)
            #model.roc(y_test, x_test)
            model.run(x_test)

            if ((True in model.predictions) and (False in model.predictions)) or ((1 in model.predictions) and (-1 in model.predictions)):
               inner_fscore = f1_score(model.predictions, y_test)
            else:
                inner_fscore = 0.0
            print(inner_fscore)
            fp.write(str(inner_fscore) + '\n')

            global out_fold_count
            global in_fold_count
            global particle_count
            global gen_count
            global in_fold_sum

            if in_fold_count >= in_fold - 1:
                in_fold_sum = in_fold_sum + inner_fscore
                res[out_fold_count][gen_count][particle_count] = in_fold_sum/in_fold
                print(str(out_fold_count) + '  ' +  str(gen_count) + '  ' + str(particle_count))
                in_fold_sum = 0
                in_fold_count = 0
                particle_count = particle_count + 1
                if particle_count >= particle:
                    particle_count = 0
                    gen_count = gen_count + 1
                    if gen_count >= gen:
                        gen_count = 0
                        out_fold_count = out_fold_count + 1
            else:
                in_fold_sum = in_fold_sum + inner_fscore
                in_fold_count = in_fold_count + 1

            return inner_fscore

        @optunity.cross_validated(x=x_train, y=y_train, num_folds=in_fold, num_iter=1)
        def LR_inner_cv(x_train, y_train, x_test, y_test, log_tol, c):
            model = LR.LR()
            model.create_ml(log_tol = log_tol, C = c)
            model.train(x_train, y_train)
            #model.roc(y_test,x_test)
            model.run(x_test)

            if ((True in model.predictions) and (False in model.predictions)) or ((1 in model.predictions) and (-1 in model.predictions)):
                inner_fscore = f1_score(model.predictions, y_test)
            else:
                inner_fscore = 0.0
            print(inner_fscore)
            fp.write(str(inner_fscore) + '\n')

            global out_fold_count
            global in_fold_count
            global particle_count
            global gen_count
            global in_fold_sum

            if in_fold_count >= in_fold - 1:
                in_fold_sum = in_fold_sum + inner_fscore
                res[out_fold_count][gen_count][particle_count] = in_fold_sum/in_fold
                print(str(out_fold_count) + '  ' +  str(gen_count) + '  ' + str(particle_count))
                in_fold_sum = 0
                in_fold_count = 0
                particle_count = particle_count + 1
                if particle_count >= particle:
                    particle_count = 0
                    gen_count = gen_count + 1
                    if gen_count >= gen:
                        gen_count = 0
                        out_fold_count = out_fold_count + 1
            else:
                in_fold_sum = in_fold_sum + inner_fscore
                in_fold_count = in_fold_count + 1

            return inner_fscore


        # 'Crossvalidation.py' will using the solver that was made at HyperParameter_Tuner
        #solver = PSO.make_PSO(self.Opt_hyperparameter, self.ML_hyperparameter)
        if ML_algorithm == 'SVM':
            for kernel in kernel_list:
                ker = kernel
                hpars, info = solver.maximize(SVM_inner_cv)
                model = SVM.SVM()
                while(hpars['c'] < 0):
                    hpars['c'] = hpars['c']+0.1
                model.create_ml(C = hpars['c'], logGamma = hpars['loggamma'], kernel = ker)
        elif ML_algorithm == 'RANDOMFOREST' or ML_algorithm == 'RF':
            hpars, info = solver.maximize(RANDOMFOREST_inner_cv)
            model = RANDOMFOREST.RANDOMFOREST()
            model.create_ml(n_estimators = hpars['n_estimators'], max_depth = hpars['max_depth'], max_features = hpars['max_features'])
        elif ML_algorithm == 'CNN':
            pass
        elif ML_algorithm == 'GNB':
            hpars, info = solver.maximize(GNB_inner_cv)
            model = GNB.GNB()
            model.create_ml(log_var_smoothing = hpars['log_var_smoothing'])
        elif ML_algorithm == 'LR':
            hpars, info = solver.maximize(LR_inner_cv)
            model = LR.LR()
            model.create_ml(log_tol = hpars['log_tol'], C = hpars['c'])
        
        #model = SVM.make_SVM(C = hpars['c'], loggamma = hpars['loggamma']).fit(x_train, y_train)
        #print(hpars)
        #model = SVM.make_SVM(C = hpars['c'], logGamma = hpars['loggamma']).fit(x_train, y_train)
        #predictions = model.decision_function(x_test)
        model.train(x_train, y_train)

        #predictions = model.predict(x_test)
        model.run(x_test)
        #return optunity.metrics.roc_auc(y_test, predictions)
        nested_fscore = f1_score(y_test, model.predictions)
        model.roc(y_test, x_test)
        model.roc_store()

        if ((True in model.predictions) and (False in model.predictions)) or ((1 in model.predictions) and (-1 in model.predictions)):
            nested_fscore = f1_score(y_test, model.predictions)
            #model.roc(y_test, x_test)
            #print(model.predictions)
        else:
            print('The model did not predict correctly.')
            nested_fscore = 0.0
        print('f1-score = ' + str(nested_fscore) + ',\n when hyperparameters \'' + str(hpars) + '\' were entered.\n' )
        return nested_fscore

    best_f1_score = nested_crossvalidation(ML_algorithm = ML_algorithm, ML_hyperparameter = ML_hyperparameter, solver = solver, kernel_list = kernel_list)
    print(res)
    
    
    for i in res:
        fp2.write('\n\n')
        for j in i:
            fp2.write('\n')
            for k in j:
                fp2.write(str(k) + ', ')
    
    fp.close()
    fp2.close()
    return best_f1_score
