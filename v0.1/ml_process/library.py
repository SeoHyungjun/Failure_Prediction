#! /usr/bin/python3

import os
import pandas as pd

def load_mod(module_path) :
    cwd = os.getcwd()
    os.chdir(os.path.dirname(module_path))
    print(os.getcwd())
    print(os.path.basename(module_path))

    mod = __import__(os.path.basename(module_path))
    # try :
    #     mod = __import__(os.path.basename(module_path))
    # except ImportError as e:
    #     print("ImportError :", os.path.abspath(__file__))
    #     print(str(e))
    #     print("Failed Load library. Check the MODEL_TBL.csv")
    #     exit(0)

    os.chdir(cwd)
    return mod


SOURCE_ROOT = "/home/syseng/Failure_Prediction/v0.1"
model_tbl_path = SOURCE_ROOT + "/MODEL_TBL.csv"

pd_model_tbl = pd.read_csv(model_tbl_path)
class_obj_dict = dict()

for index, row in pd_model_tbl.iterrows():
   mod_path = SOURCE_ROOT + "/" +  row['MODULE_PATH']
   mod = load_mod(mod_path)
   class_obj_dict[row['MODEL_NAME']] = getattr(mod, row['CLASS_NAME'])

print(class_obj_dict)
