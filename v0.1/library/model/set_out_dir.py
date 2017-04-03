import os

STR_DERECTORY_ROOT = "ML_data"
STR_DERECTORY_MODEL = ""
STR_DERECTORY_INPUT = "input"
STR_DERECTORY_SAVER = "saver"
STR_DERECTORY_SUMMARY = "summary"
STR_DERECTORY_TRAIN = "train"
STR_DERECTORY_DEV = "dev"

def make_dir(model_names):
    STR_DIRECTORY_MODEL = list(model_names.split(","))
    if not os.path.exists(STR_DERECTORY_ROOT):
        os.makedirs(STR_DERECTORY_ROOT)
        for name in STR_DIRECTORY_MODEL:
            model_path = os.path.join(STR_DERECTORY_ROOT, name)
            os.makedirs(model_path)
            input_path = os.path.join(model_path, STR_DERECTORY_INPUT)
            os.makedirs(input_path)
            saver_path = os.path.join(model_path, STR_DERECTORY_SAVER)
            os.makedirs(saver_path)
            summary_path = os.path.join(model_path, STR_DERECTORY_SUMMARY)
            os.makedirs(summary_path)
            train_path = os.path.join(summary_path, STR_DERECTORY_TRAIN)
            os.makedirs(train_path)
            dev_path = os.path.join(summary_path, STR_DERECTORY_DEV)
            os.makedirs(dev_path)
