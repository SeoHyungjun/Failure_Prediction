# [output directory]
PROJECT_DIRPATH = "./FP_project" # only configurable in output directory config
INPUT_DIR = "input"
TRAINED_MODEL_DIR = "trained_model"
SUMMARY_DIR = "summary"
SUMMARY_TRAIN_DIR = "train"
SUMMARY_VALIDATION_DIR = "validation"
MODEL_SAVE_TAG = "model"


# [K_means]
KMEANS_MODEL_DIR = "K_Means"
KMEANS_MODEL_NAME = "K_Means"
KMEANS_CENTROID_NUM = 4
KMEANS_MAX_ITERS = 100
KMEANS_TRAINED_CENTROID_FILE = "centroid.csv"


# [ANN]
ANN_MODEL_DIR = "ANN"
ANN_MODEL_NAME = "ANN"
ANN_NODES_NUM = [8,5]
## regularization : dropout_keep_prob, l2_reg_lambda(when not applied, each value are 1.0, 0.0)
ANN_DROPOUT_KEEP_PROB = 0.5
ANN_L2_REG_LAMBDA = 0.0
ANN_VALIDATION_SAMPLE_PERCENTAGE = 0.1
ANN_BATCH_SIZE = 32
ANN_EPOCHS_NUM = 1
ANN_VALIDATION_INTERVAL = 2000
