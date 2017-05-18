# [output directory]
DIRPATH_PROJECT = "./FP_project" # only configurable in output directory config
DIR_INPUT = "input"
DIR_TRAINED_MODEL = "trained_model"
DIR_SUMMARY = "summary"
DIR_SUMMARY_TRAIN = "train"
DIR_SUMMARY_VALIDATION = "validation"
MODEL_SAVE_TAG = "model"   


# [K_means]
KMEANS_DIR_MODEL = "K_Means"
KMEANS_MODEL_NAME = "K_Means"
KMEANS_NUM_CENTROID = 4
KMEANS_MAX_ITERS = 500
KMEANS_TRAINED_CENTROID_FILE = "centroid.csv"


# [ANN]
ANN_DIR_MODEL = "ANN"
ANN_MODEL_NAME = "ANN"
ANN_NUM_NODES = [8,5,3]
## regularization : dropout_keep_prob, l2_reg_lambda(when not applied, each value are 1.0, 0.0)
ANN_DROPOUT_KEEP_PROB = 0.5
ANN_L2_REG_LAMBDA = 0.0
ANN_VALIDATION_SAMPLE_PERCENTAGE = 0.1
ANN_BATCH_SIZE = 2
ANN_NUM_EPOCHS = 2
ANN_VALIDATION_INTERVAL = 2000
