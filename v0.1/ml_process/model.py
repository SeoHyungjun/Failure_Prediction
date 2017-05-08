from abc import ABC, abstractmethod

# abstract class to be used by machine learning class
class Model(ABC):
    # db = Database()
    # input_x = Dataframe()
    # input_y = Dataframe()
    # ev_inf = Eval_info()

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def set_config(self, arg_dict):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def restore_all(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def print_model_config(model_name, arg_dict):
        print("Model [%s] configuration information" % model_name)
        for key in arg_dict.keys():
            print("[%s] : %s" % (key, arg_dict.get(key)))


    @staticmethod
    def print_config_all(model_dict) :
        print("Each model Configuration information")
        print("The Number of Models : %d" % len(model_dict))
        for model_order, model in model_dict.items() :
            model.print_model_config(model.model_name, model.arg_dict)

