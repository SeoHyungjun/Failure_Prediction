import os
import constant as ct

def make_dir(model_tag, dir_project):
    if not os.path.exists(dir_project):
        os.makedirs(dir_project)

    model_path = os.path.join(dir_project, model_tag)
    input_path = os.path.join(model_path, ct.INPUT_DIR)
    trained_model_path = os.path.join(model_path, ct.TRAINED_MODEL_DIR)
    summary_path = os.path.join(model_path, ct.SUMMARY_DIR)
    summary_train_path = os.path.join(summary_path, ct.SUMMARY_TRAIN_DIR)
    summary_validation_path = os.path.join(summary_path, ct.SUMMARY_VALIDATION_DIR)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(input_path)
        os.makedirs(trained_model_path)
        os.makedirs(summary_path)
        os.makedirs(summary_train_path)
        os.makedirs(summary_validation_path)

    return trained_model_path, summary_train_path, summary_validation_path
