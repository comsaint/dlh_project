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
DISEASE = 'Atelectasis'
NUM_CLASSES = 2
FEATURE_EXTRACT = True
MODEL_NAME = 'alexnet'
SEED=1
VAL_SIZE=0.1
NUM_WORKERS=12

NUM_EPOCHS=30
LEARNING_RATE=0.005
BATCH_SIZE=64


# Other settings
SAMPLING = 0  # sample the input data to reduce data size (for quick test). 0 to disable.
# estimate of image mean and std
SAMPLE_MEAN = 129.76628483/255
SAMPLE_STD = 59.70063891/255
