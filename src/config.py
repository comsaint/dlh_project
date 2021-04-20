import torch
DEVICE = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
ROOT_PATH='../'

SRC_DIR='src/'
DATA_DIR='data/'
MODEL_DIR='models/'
CHECKPOINT_DIR = 'checkpoints/'
PROCESSED_DATA_DIR='data/processed'
RAW_DATA_DIR='data/raw'

INDEX_FILE = 'Data_Entry_2017.csv'
TRAIN_VAL_FILE = 'train_val_list.txt'
TEST_FILE = 'test_list.txt'

WRITER_NAME = 'runs/experiment_resnext101'

# Hyperparamenters
NUM_CLASSES = 15  # 14 diseases (+1 if include 'No Finding')
USE_PRETRAIN = True  # start with pretrained weights?
FEATURE_EXTRACT = False  # must be false if USE_PRETRAIN==False
MODEL_NAME = 'resnext101'
VAL_SIZE = 0.05
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 16

# just for convenience. Better be inferred from data.
if NUM_CLASSES == 14:
    TEXT_LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
elif NUM_CLASSES == 15:
    TEXT_LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding']

# Utilities
NUM_WORKERS = 7
SEED = 42

# Other settings
SAMPLING = 0  # number of samples of input data, to reduce data size (for quick test). 0 to disable.
VERBOSE=True
MODEL_LOSS=True
GREY_SCALE=False
