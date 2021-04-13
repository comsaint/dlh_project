# Paths
ROOT_PATH='../'

SRC_DIR='src/'
DATA_DIR='data/'
MODEL_DIR='models/'
PROCESSED_DATA_DIR='data/processed'

INDEX_FILE = 'Data_Entry_2017.csv'
TRAIN_VAL_FILE = 'train_val_list.txt'
TEST_FILE = 'test_list.txt'

# Hyperparamenters
NUM_CLASSES = 14  # 14 diseases (+1 if include 'No Finding')
USE_PRETRAIN = False
FEATURE_EXTRACT = False  # must be false if USE_PRETRAIN==False
MODEL_NAME = 'resnext'
VAL_SIZE = 0.1
NUM_EPOCHS = 50
LEARNING_RATE = 0.01
BATCH_SIZE = 64

# Utilities
NUM_WORKERS = 7
SEED = 42
WRITER_NAME = 'runs/experiment_reproduce_2'

# Other settings
SAMPLING = 0  # number of samples of input data, to reduce data size (for quick test). 0 to disable.
