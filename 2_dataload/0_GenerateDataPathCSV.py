# Data Structure
# dataset/
# |-- imgA
# |   |-- 0001.nii.gz
# |   |-- 0002.nii.gz
# |   |-- ...
# |-- imgB
# |   |-- 0001.nii.gz
# |   |-- 0002.nii.gz
# |   |-- ...
# |-- labels.csv
#       ↓↓↓
#     labels.csv contains the following content:
#       ↓↓↓
# idx,        label,        WBC,      NE,     D_D,      Lactic
# 0001        1             0.4       0.4     0.6       0.01
# 0002        1             0.3       0.3     0.2       0.07
# ...         ...           ...       ...     ...       ...
"""Data Structure"""


import os
import pandas as pd


folder_path_a = '1_dataset/imgA'
folder_path_b = '1_dataset/imgB'
df = pd.read_csv('1_dataset/labels.csv')

file_list_a = os.listdir(folder_path_a)
file_list_b = os.listdir(folder_path_b)

dfsa = []
dfsb = []
ids = df['idx'].tolist()
ids = [str(i).zfill(4) for i in ids]
print(ids)

for id in ids:
    a_path_name = os.path.join(folder_path_a, id + '.nii.gz')
    dfsa.append(a_path_name)
    print('a_name', a_path_name)

    b_path_name = os.path.join(folder_path_b, id + '.nii.gz')
    dfsb.append(b_path_name)
    print('b_name', b_path_name)


df['image_pathA'] = dfsa
df['image_pathB'] = dfsb

df.to_csv('', index=False)


