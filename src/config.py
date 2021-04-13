import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
ROOT_PATH='../'

SRC_DIR='src/'
DATA_DIR='data/'
MODEL_DIR='models/'
PROCESSED_DATA_DIR='data/processed'
RAW_DATA_DIR='data/raw'

INDEX_FILE = 'Data_Entry_2017.csv'
TRAIN_VAL_FILE = 'train_val_list.txt'
TEST_FILE = 'test_list.txt'

# Hyperparamenters
NUM_CLASSES = 14  # 14 diseases (+1 if include 'No Finding')
USE_PRETRAIN = True
FEATURE_EXTRACT = True  # must be false if USE_PRETRAIN==False
MODEL_NAME = 'densenet'
VAL_SIZE = 0.1
NUM_EPOCHS = 30
LEARNING_RATE = 0.0005
BATCH_SIZE = 126

# Utilities
NUM_WORKERS = 7
SEED = 1

# Other settings
SAMPLING = 0  # number of samples of input data, to reduce data size (for quick test). 0 to disable.

