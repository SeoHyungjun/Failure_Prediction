import configparser
import Hyperparameter_Tuner as HPT
from multiprocessing import Process, Pool

#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split

class Optimizer:
    def __init__(self, config_file_name):
        self.config = config_file_name

        self.ML_algorithm = {}
        self.Opt_algorithm = {}
        #self.iris_dataset = load_iris()
        #self.iris_x, self.iris_X_test, self.iris_y, self.iris_y_test = train_test_split(self.iris_dataset['data'], self.iris_dataset['target'], random_state=0)

    def get_config(self):
        # read config file and save content as attribute
        self.configparser = configparser.ConfigParser()
        self.configparser.read('../' + self.config)
        
        # store ml_algorithm & opt_algorithm after change string to list
        # using dictionary in dictionary such as {BO : {A:1, B:2}, PSO : {C:3, D:4}}
        for opt in self.configparser['Opt_algorithm']['Opt_name'].split(','):
            hp_dict = {}
            opt_str = opt.strip().upper()
            for hp in self.configparser[opt_str + '_Hyperparameter']:
                hp_dict[hp] = self.configparser[opt_str + '_Hyperparameter'][hp]
            self.Opt_algorithm[opt_str] = hp_dict

        print(self.Opt_algorithm)

        for ml in self.configparser['ML_algorithm']['ML_name'].split(','):
            hp_dict = {}
            if ml.strip().upper() == 'RANDOMFOREST':
                ml_str = 'RF'
            else:
                ml_str = ml.strip().upper()
            for hp in self.configparser[ml_str + '_Hyperparameter']:
                hp_list = []
                if hp == 'kernel':
                    for ker in self.configparser[ml_str+'_Hyperparameter'][hp].split(','):
                        hp_list.append(ker.strip().strip("'").strip("'"))
                    hp_dict[hp] = hp_list

                else:
                    hp_num = self.configparser[ml_str+ '_Hyperparameter'][hp].split(',')
                    if hp_num[2].strip().strip('[').strip(']') == 'int':
                        hp_list.append(int(hp_num[0].strip().strip('[').strip(']')))
                        hp_list.append(int(hp_num[1].strip().strip('[').strip(']')))
                    elif hp_num[2].strip().strip('[').strip(']') == 'float':
                        hp_list.append(float(hp_num[0].strip().strip('[').strip(']')))
                        hp_list.append(float(hp_num[1].strip().strip('[').strip(']')))
                    hp_dict[hp] = hp_list
                self.ML_algorithm[ml_str] = hp_dict
 
            print(self.ML_algorithm)
 
    def run_Hyperparameter_Tuner(self, date):
        for opt_al in self.Opt_algorithm.keys():
            #with HPT.Hyperparameter_Tuner(opt_al, self.Opt_algorithm[opt_al], self.ML_algorithm) as Hyperparameter_Tuner_class:
                #Hyperparameter_Tuner_class.run_opt_algorithm()
            HPT_class = HPT.Hyperparameter_Tuner(opt_al, self.Opt_algorithm[opt_al], self.ML_algorithm)
            HPT_class.load_data(date)
            #HPT_class.load_baidu_data(date)
            HPT_class.run_opt_algorithm()

def test(c_list):
    opt = Optimizer('config')
    opt.get_config()
    opt.run_Hyperparameter_Tuner(c_list[0])

    fp2 = open('result2.txt', 'a')
    fp2.write('\nbackblaze LR nofs ' + str(c_list[0]*24) + ' ' + str(c_list[1]) + '\n')
    fp = open('result/svm_' +str(c_list[0]*24) + '_' + str(c_list[1]) + '.txt', 'a')
    fp.write('\nbackblaze LR nofs ' + str(c_list[0]*24) + ' ' + str(c_list[1]) + '\n')
    fp.close()

if __name__ == '__main__':
    time = [[1,1], [1,2], [1,3], [1,4], [1,5], 
            [2,1], [2,2], [2,3], [2,4], [2,5], 
            [4,1], [4,2], [4,3], [4,4], [4,5], 
            [8,1], [8,2], [8,3], [8,4], [8,5]]
    #time = [[8,1]]
    with Pool(32) as p:
        p.map(test, time)

'''
time = [8]

for date in time:
    for i in range(1, 2):
        opt = Optimizer('config')
        opt.get_config()
        opt.run_Hyperparameter_Tuner(date)

        fp = open('result2.txt', 'a')
        fp.write('\nbaidu svm rbf ' + str(date*24) + ' ' + str(i) + '\n')
        fp.close()
'''
