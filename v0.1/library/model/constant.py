# [output directory]
DIR_OUTPUT = "./ML_data" # only configurable in output directory config
DIR_TRAINED_MODEL = "model"
DIR_SUMMARY = "summary"
DIR_SUMMARY_TRAIN = "train"
DIR_SUMMARY_VALIDATION = "validation"
MODEL_SAVE_TAG = "model"   


# [K_means]
KMEANS_MODEL_NAME = "K_means"
KMEANS_NUM_CENTROID = 4
KMEANS_MAX_ITERS = 10
KMEANS_TRAINED_CENTROID_FILE = "learned_centroid.csv"
#x_train
#x_eval
#x_run


# [ANN]
ANN_MODEL_NAME = "ANN"
ANN_NUM_NODES = [2,4,3]
## regularization : dropout_keep_prob, l2_reg_lambda(when not applied, each value are 1.0, 0.0)
ANN_DROPOUT_KEEP_PROB = 0.5
ANN_L2_REG_LAMBDA = 0.0
ANN_VALIDATION_SAMPLE_PERCENTAGE = 0.1
ANN_BATCH_SIZE = 30
ANN_NUM_EPOCHS = 5
ANN_VALIDATION_INTERVAL = 50
#x_train
#x_eval
#x_run
