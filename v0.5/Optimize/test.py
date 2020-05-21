import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    

from ML_model.SVM import SVM
#from ML_model.base_ml import Machine_Learning
#from base_ml import Machine_Learning

a = 'SVM'

b = SVM()

#b = getattr(ML_model, a)
print(b)
