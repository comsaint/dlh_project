import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ##################################
# Paths - DO NOT EDIT!
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
# ##################################

# Hyperparamenters
WRITER_NAME = 'runs/experiment_fullmodel_resnet50_val010_noextra_noclassweight'  #
VAL_SIZE = 0.10
NUM_EPOCHS = 50
BATCH_SIZE = 32

USE_CLASS_WEIGHT = False  # weight class samples by prevalence
USE_EXTRA_INPUT = False  # TODO: concat age, gender and view position to features

GLOBAL_IMAGE_SIZE = 224
LOCAL_IMAGE_SIZE = 224

# Models
GLOBAL_MODEL_NAME = 'resnet50'
LOCAL_MODEL_NAME = 'resnet50'
FUSION_MODEL_NAME = 'fusion'  # only for filename
FINE_TUNE = True  # if True, fine tune a pretrained model. Otherwise train from scratch.
FINE_TUNE_START_EPOCH = 5  # allow tuning of all parameters starting from this epoch. Ignore if FINE_TUNE==False.
EARLY_STOP_EPOCHS = 10  # stop training if no improvement compared to last best epoch

# initial learning rates
GLOBAL_LEARNING_RATE = 1e-5
LOCAL_LEARNING_RATE = 1e-5
FUSION_LEARNING_RATE = 1e-5
# TODO: settings for optimizer e.g. patience etc.

HEATMAP_THRESHOLD = 0.70

# Other settings
SAMPLING = 0  # samples the input data to reduce data size for quick test. 0 to disable (i.e. use all training set)
VERBOSE = True
MODEL_LOSS = False
GREY_SCALE = False

# Utilities
NUM_WORKERS = 4
SEED = 42

# just for convenience. Better be inferred from data.
NUM_CLASSES = 14  # 14 diseases (+1 if include 'No Finding')
if NUM_CLASSES == 14:
    TEXT_LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',  'Pleural_Thickening', 'Hernia']
elif NUM_CLASSES == 15:
    TEXT_LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',  'Pleural_Thickening', 'Hernia', 'No Finding']
