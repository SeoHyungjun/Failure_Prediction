import os
import constant as ct

def make_dir(model_names):
    ct.STR_DIRECTORY_MODEL = list(model_names.split(","))
    if not os.path.exists(ct.STR_DERECTORY_ROOT):
        os.makedirs(ct.STR_DERECTORY_ROOT)
        for name in ct.STR_DIRECTORY_MODEL:
            model_path = os.path.join(ct.STR_DERECTORY_ROOT, name)
            os.makedirs(model_path)
            input_path = os.path.join(model_path, ct.STR_DERECTORY_INPUT)
            os.makedirs(input_path)
            saver_path = os.path.join(model_path, ct.STR_DERECTORY_SAVER)
            os.makedirs(saver_path)
            summary_path = os.path.join(model_path, ct.STR_DERECTORY_SUMMARY)
            os.makedirs(summary_path)
            train_path = os.path.join(summary_path, ct.STR_DERECTORY_TRAIN)
            os.makedirs(train_path)
            dev_path = os.path.join(summary_path, ct.STR_DERECTORY_DEV)
            os.makedirs(dev_path)
