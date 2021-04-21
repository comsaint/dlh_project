"""

"""
import config
import pandas as pd
from data_processing import load_data_file

df_data, lst_labels = load_data_file(sampling=config.SAMPLING)

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# print(df_data[['Image Index',*CLASS_NAMES]].head())
df_data = df_data[['Image Index', *CLASS_NAMES]].sort_values('Image Index').reset_index(drop=True)

df_tr = pd.read_csv('D:\\agcnn\\labels\\train_list.txt', sep=' ', names=['Image Index', *CLASS_NAMES], header=None)
df_v = pd.read_csv('D:\\agcnn\\labels\\val_list.txt', sep=' ', names=['Image Index', *CLASS_NAMES], header=None)
df_te = pd.read_csv('D:\\agcnn\\labels\\test_list.txt', sep=' ', names=['Image Index', *CLASS_NAMES], header=None)
df_full = pd.concat([df_tr, df_v, df_te]).sort_values('Image Index').reset_index(drop=True)

print(df_data.equals(df_full))
print(df_data.head())
print(df_data.columns)
print(df_full.head())
print(df_full.columns)

df_cmp = df_data.compare(df_full)
df_cmp = df_cmp.join(df_data['Image Index'])
print(df_cmp)

df_cmp.to_csv('cmp.csv')

# write corrected train_list.txt etc.
df_tr_corrected = df_data[df_data['Image Index'].isin(df_tr['Image Index'].tolist())]
df_v_corrected = df_data[df_data['Image Index'].isin(df_v['Image Index'].tolist())]
df_te_corrected = df_data[df_data['Image Index'].isin(df_te['Image Index'].tolist())]

# list with labels
df_tr_corrected.to_csv('train_list_corrected.txt', sep=' ', header=False, index=False)
df_v_corrected.to_csv('val_list_corrected.txt', sep=' ', header=False, index=False)
df_te_corrected.to_csv('test_list_corrected.txt', sep=' ', header=False, index=False)

# list without labels
df_tr_corrected['Image Index'].to_csv('train_list_corrected_nolabel.txt', sep=' ', header=False, index=False)
df_v_corrected['Image Index'].to_csv('val_list_corrected_nolabel.txt', sep=' ', header=False, index=False)
df_te_corrected['Image Index'].to_csv('test_list_corrected_nolabel.txt', sep=' ', header=False, index=False)