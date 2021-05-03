import torch
from datetime import datetime
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ##################################
# Paths - DO NOT EDIT!
ROOT_PATH = '../'

SRC_DIR = 'src/'
DATA_DIR = 'data/'
MODEL_DIR = 'models/'
CHECKPOINT_DIR = 'checkpoints/'
PROCESSED_DATA_DIR = 'data/processed'
RAW_DATA_DIR = 'data/raw'
OUT_DIR='out/'

INDEX_FILE = 'Data_Entry_2017.csv'
TRAIN_VAL_FILE = 'train_val_list.txt'
TEST_FILE = 'test_list.txt'
# ##################################

# Hyperparamenters
VAL_SIZE = 0.10
NUM_EPOCHS = 50
BATCH_SIZE = 24

USE_CLASS_WEIGHT = False  # weight class samples by prevalence
USE_EXTRA_INPUT = False  # TODO: concat age, gender and view position to features

# Uses model sizes unless undefined
GLOBAL_IMAGE_SIZE = 224
LOCAL_IMAGE_SIZE = 224

# Models
GLOBAL_MODEL_NAME = 'densenet'
LOCAL_MODEL_NAME = 'capsnet'
FUSION_MODEL_NAME = 'fusion'  # only for filename
FINE_TUNE = True  # if True, fine tune a pretrained model. Otherwise train from scratch.
FINE_TUNE_START_EPOCH = 5  # allow tuning of all parameters starting from this epoch. Ignore if FINE_TUNE==False.
EARLY_STOP_EPOCHS = 10  # stop training if no improvement compared to last best epoch
FINE_TUNE_STEP_WISE=True

# initial learning rates
GLOBAL_LEARNING_RATE = 1e-4
LOCAL_LEARNING_RATE  = 1e-3  #Note: For capsnet, initial lr should be 1e-3 or higher
FUSION_LEARNING_RATE = 1e-5
# TODO: settings for optimizer e.g. patience etc.

HEATMAP_THRESHOLD = 0.70

# Other settings
SAMPLING = 0  # samples the input data to reduce data size for quick test. 0 to disable (i.e. use all training set)
VERBOSE = True
MODEL_LOSS = False
GREY_SCALE = False
RECONSTRUCT= False

# Utilities
NUM_WORKERS = 4
SEED = 42

# just for convenience. Better be inferred from data.
NUM_CLASSES = 14  # 14 diseases (+1 if include 'No Finding')
if NUM_CLASSES == 14:
    TEXT_LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',  'Pleural_Thickening', 'Hernia']
elif NUM_CLASSES == 15:
    TEXT_LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',  'Pleural_Thickening', 'Hernia', 'No Finding']

# TensorBoard logs
current_time = datetime.now().strftime("%Y%m%d%H%M%S")  # not used due to multiple workers
WRITER_NAME = f"runs/experiment_fullmodel_{GLOBAL_MODEL_NAME}_{LOCAL_MODEL_NAME}_{int(VAL_SIZE*100)}_{BATCH_SIZE}_{int(HEATMAP_THRESHOLD*100)}"
if USE_CLASS_WEIGHT:
    WRITER_NAME += '_classw'
if USE_EXTRA_INPUT:
    WRITER_NAME += '_extra'
if FINE_TUNE:
    WRITER_NAME += '_tune'
