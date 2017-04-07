import os
import constant as ct

def make_dir(model_name):
    if not os.path.exists(ct.STR_DERECTORY_ROOT):
        os.makedirs(ct.STR_DERECTORY_ROOT)

    model_path = os.path.join(ct.STR_DERECTORY_ROOT, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        graph_path = os.path.join(model_path, ct.STR_DERECTORY_GRAPH)
        os.makedirs(graph_path)

        summary_path = os.path.join(model_path, ct.STR_DERECTORY_SUMMARY)
        os.makedirs(summary_path)
        summary_train_path = os.path.join(summary_path, "train")
        os.makedirs(summary_train_path)
        summary_dev_path = os.path.join(summary_path, "dev")
        os.makedirs(summary_dev_path)
