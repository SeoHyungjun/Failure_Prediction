import os
import constant as ct

def make_dir(model_tag, dir_output):
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    model_path = os.path.join(dir_output, model_tag)
    trained_model_path = os.path.join(model_path, ct.DIR_TRAINED_MODEL)
    summary_path = os.path.join(model_path, ct.DIR_SUMMARY)
    summary_train_path = os.path.join(summary_path, ct.DIR_SUMMARY_TRAIN)
    summary_validation_path = os.path.join(summary_path, ct.DIR_SUMMARY_VALIDATION)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(trained_model_path)
        os.makedirs(summary_path)
        os.makedirs(summary_train_path)
        os.makedirs(summary_validation_path)

    return trained_model_path, summary_train_path, summary_validation_path
