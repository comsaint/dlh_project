import torch
#from ray import tune
import os

trial = 1
HYPERSEARCH = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ##################################
# Paths - DO NOT EDIT!
ROOT_PATH = os.path.join(os.path.dirname(__file__), '..')

SRC_DIR = 'src/'
DATA_DIR = 'data/'
MODEL_DIR = 'models/'
CHECKPOINT_DIR = 'checkpoints/'
PROCESSED_DATA_DIR = 'data/processed'
RAW_DATA_DIR = 'data/raw'
OUT_DIR='out/'

INDEX_FILE = 'Data_Entry_2017_v2020.csv'
TRAIN_VAL_FILE = 'train_val_list.txt'
TEST_FILE = 'test_list.txt'

# Uses model sizes unless undefined
GLOBAL_IMAGE_SIZE = 224
LOCAL_IMAGE_SIZE = 224

# Other settings
SAMPLING = 0  # samples the input data to reduce data size for quick test. 0 to disable (i.e. use all training set)
VERBOSE = True
MODEL_LOSS = False
GREY_SCALE = False
RECONSTRUCT = False

# Utilities
NUM_WORKERS = 7
SEED = 42

# just for convenience. Better be inferred from data.
NUM_CLASSES = 14  # 14 diseases (+1 if include 'No Finding')
if NUM_CLASSES == 14:
    TEXT_LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',  'Pleural_Thickening', 'Hernia']
elif NUM_CLASSES == 15:
    TEXT_LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',  'Pleural_Thickening', 'Hernia', 'No Finding']

'''
# search space of hyper-parameters
params_hypersearch = {
            "NUM_EPOCHS": 100,
            "BATCH_SIZE": 32,
            "FINE_TUNE": tune.choice([True, False]),
            "FINE_TUNE_START_EPOCH": 5,
            "EARLY_STOP_EPOCHS": 10,
            "VAL_SIZE": tune.grid_search([0.05, 0.01, 0.01]),
            "HEATMAP_THRESHOLD":  tune.uniform(0.6, 0.8),
            "GLOBAL_LEARNING_RATE": tune.loguniform(1e-6, 1e-3),
            "LOCAL_LEARNING_RATE": tune.loguniform(1e-6, 1e-3),
            "FUSION_LEARNING_RATE": tune.loguniform(1e-6, 1e-3),
            "USE_CLASS_WEIGHT": tune.choice([True, False]),
            "USE_EXTRA_INPUT": True,
            "GLOBAL_MODEL_NAME": tune.choice(['densenet', 'resnet50', 'resnext50', 'resnext101']),
            "LOCAL_MODEL_NAME": tune.choice(['densenet', 'resnet50', 'resnext50', 'resnext101']),
            "FUSION_MODEL_NAME": 'fusion',
            "FINE_TUNE_STEP_WISE": tune.choice([True, False]),
}
'''

# config of single pass
params = {
    "NUM_EPOCHS": 100,
    "BATCH_SIZE": 16,
    "FINE_TUNE": True,
    "FINE_TUNE_START_EPOCH": 4,
    "FINE_TUNE_STEP_WISE": True,
    "EARLY_STOP_EPOCHS": 10,
    "VAL_SIZE": 0.05,
    "HEATMAP_THRESHOLD":  0.7,
    "GLOBAL_LEARNING_RATE": 1e-4,
    "LOCAL_LEARNING_RATE": 1e-4,
    "FUSION_LEARNING_RATE": 1e-4,
    "USE_CLASS_WEIGHT": True,
    "USE_EXTRA_INPUT": True,
    "GLOBAL_MODEL_NAME": 'mobilenet',
    "LOCAL_MODEL_NAME": 'mobilenet',
    "FUSION_MODEL_NAME": 'fusion',
    "AUGMENTATIONS": ['rot', 'hflip'],  # see dataset.make_data_transform() for options
    "RESUME": True
}

WRITER_NAME = f"runs/experiment_fullmodel_{params['GLOBAL_MODEL_NAME']}_{params['LOCAL_MODEL_NAME']}_" \
              f"{float(params['VAL_SIZE'])*100}_{params['BATCH_SIZE']}_{float(params['HEATMAP_THRESHOLD'])*100}"
if params['USE_CLASS_WEIGHT']:
    WRITER_NAME += '_classw'
if params['USE_EXTRA_INPUT']:
    WRITER_NAME += '_extra'
if params['FINE_TUNE']:
    WRITER_NAME += '_tune'
if params['AUGMENTATIONS']:
    _s = '-'.join(params['AUGMENTATIONS'])
    WRITER_NAME += '_' + _s
WRITER_NAME += f"_trial{trial}"
