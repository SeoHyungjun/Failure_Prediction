#! /usr/bin/python3

import sys
import os
import pandas as pd

def load_mod(module_name) :
#    cwd = os.getcwd()
#    os.chdir(os.path.dirname(module_path))
    # print(os.getcwd())
    # print(os.path.basename(module_path))
    module = __import__(module_name)
    # try :
    #     mod = __import__(os.path.basename(module_path))
    # except ImportError as e:
    #     print("ImportError :", os.path.abspath(__file__))
    #     print(str(e))
    #     print("Failed Load library. Check the MODEL_TBL.csv")
    #     exit(0)

#    os.chdir(cwd)
    return module

SOURCE_ROOT = os.path.join("/", "root", "Failure_Prediction", "v0.1")
sys.path.insert(0, os.path.join(SOURCE_ROOT, "library", "data_transform"))
import data_transform as dt
model_tbl_path = os.path.join(SOURCE_ROOT, "MODEL_TBL.csv")
module_dirpaths = []

pd_model_tbl = pd.read_csv(model_tbl_path)
class_obj_dict = dict()
for index, row in pd_model_tbl.iterrows():
    module_path = os.path.join(SOURCE_ROOT, row['MODULE_PATH'])
    module_name = os.path.basename(module_path).split('.')[0]
    module_dirpath = os.path.dirname(module_path)
    if module_dirpath not in module_dirpaths:
        module_dirpaths.append(module_dirpath)
        sys.path.insert(0, module_dirpath)

    module = load_mod(module_name)
    class_obj_dict[row['MODEL_NAME']] = getattr(module, row['CLASS_NAME'])

print(class_obj_dict)
dt_func_cls = dt.Data_transform()
