import os
import constant as ct

def make_dir(algorithm_name):
    if not os.path.exists(ct.STR_DERECTORY_ROOT):
        os.makedirs(ct.STR_DERECTORY_ROOT)

    algorithm_path = os.path.join(ct.STR_DERECTORY_ROOT, algorithm_name)
    model_path = os.path.join(algorithm_path, ct.STR_DERECTORY_GRAPH)
    summary_path = os.path.join(algorithm_path, ct.STR_DERECTORY_SUMMARY)
    summary_train_path = os.path.join(summary_path, ct.STR_DERECTORY_SUMMARY_TRAIN)
    summary_dev_path = os.path.join(summary_path, ct.STR_DERECTORY_SUMMARY_DEV)

    if not os.path.exists(algorithm_path):
        os.makedirs(algorithm_path)
        os.makedirs(model_path)
        os.makedirs(summary_path)
        os.makedirs(summary_train_path)
        os.makedirs(summary_dev_path)

    return model_path, summary_train_path, summary_dev_path
