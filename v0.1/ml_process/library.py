import model1, model2

class library :
    def __init__(self) :
        self.models = {'model1' : model1.model1(), 'model2' : model2.model2()}

    def insert_model(self, model_name, model_cls):
        self.models[model_name] = model_cls



# for test
if __name__ == '__main__' :
    lib = library()
    print(lib.models['model1'].model_name)
    print(lib.models['model2'].model_name)
