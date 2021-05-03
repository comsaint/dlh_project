import torch
from configparser import SafeConfigParser

parser = SafeConfigParser()
parser.optionxform = lambda option: option  # preserve case for letters
parser.read('params.ini')

params = dict()
for name in parser.options('ints'):
    params[name] = parser.getint('ints', name)
for name in parser.options('floats'):
    params[name] = parser.getfloat('floats', name)
for name in parser.options('booleans'):
    params[name] = parser.getboolean('booleans', name)
for name in parser.options('strings'):
    params[name] = parser.get('strings', name)

trial = 1
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

INDEX_FILE = 'Data_Entry_2017_v2020.csv'
TRAIN_VAL_FILE = 'train_val_list.txt'
TEST_FILE = 'test_list.txt'

GLOBAL_IMAGE_SIZE = 224
LOCAL_IMAGE_SIZE = 224

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

# TensorBoard logs
#current_time = datetime.now().strftime("%Y%m%d%H%M%S")  # not used due to multiple workers
WRITER_NAME = f"runs/experiment_fullmodel_{params['GLOBAL_MODEL_NAME']}_{params['LOCAL_MODEL_NAME']}_" \
              f"{float(params['VAL_SIZE'])*100}_{params['BATCH_SIZE']}_{float(params['HEATMAP_THRESHOLD'])*100}"
if params['USE_CLASS_WEIGHT']:
    WRITER_NAME += '_classw'
if params['USE_EXTRA_INPUT']:
    WRITER_NAME += '_extra'
if params['FINE_TUNE']:
    WRITER_NAME += '_tune'
WRITER_NAME += f"_trial{trial}"
