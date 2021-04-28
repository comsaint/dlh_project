import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

WRITER_NAME = 'runs/experiment_densenet_val005_classweight_adamw_nopretrain_finetune_1'

# Hyperparamenters
NUM_CLASSES = 14  # 14 diseases (+1 if include 'No Finding')
FINE_TUNE = True  # if True, fine tune a pretrained model. Otherwise train from scratch
FINE_TUNE_START_EPOCH = 8  # allow tuning of all parameters starting from this epoch. Ignore if FINE_TUNE==False.
MODEL_NAME = 'densenet'
VAL_SIZE = 0.05
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
EARLY_STOP_EPOCHS = 10  # stop training if no improvement compared to last best epoch
USE_CLASS_WEIGHT = True  # weight class samples by prevalence

# Other settings
SAMPLING = 0  # number of samples of input data, to reduce data size (for quick test). 0 to disable.
VERBOSE = True
MODEL_LOSS = False
GREY_SCALE = False

# Utilities
NUM_WORKERS = 7
SEED = 42

# just for convenience. Better be inferred from data.
if NUM_CLASSES == 14:
    TEXT_LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',  'Pleural_Thickening', 'Hernia']
elif NUM_CLASSES == 15:
    TEXT_LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',  'Pleural_Thickening', 'Hernia', 'No Finding']
