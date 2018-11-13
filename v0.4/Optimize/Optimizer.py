import configparser

class Optimizer:
    def __init__(self, config_file_name):
        self.config = config_file_name

        self.ML_algorithm = {}
        self.Opt_algorithm = {}

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
                hp_dict[hp] = self.configparser[ml_str + '_Hyperparameter'][hp]
            self.ML_algorithm[ml_str] = hp_dict
 
        print(self.ML_algorithm)
 
    def run_Hyperparameter_Tuner(self):
        for opt_al in self.Opt_algorithm.keys():
            print(opt_al)
            with Hyperparameter_Tuner(opt_al, self.ML_algorithm) as Hyperparameter_Tuner:
                d


opt = Optimizer('config')
opt.get_config()
#opt.run_Hyperparameter_Tuner()
