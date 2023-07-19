import pickle as pkl 
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import datetime as dt

    
#region 1 Load Files and initialize variables
io_params = {'parse_dates': ['epoch'], 'index_col': 'id'}
file_path = "predict-the-positions-and-speeds-of-600-satellites/jan_train.csv"
df_train = pd.read_csv(file_path, **io_params)
print(f"There are {len(df_train.sat_id.unique())} unique satellites in the train dataset, with {len(df_train)} total rows.")

##1.1 Combine test dataset and answer key
file_path_test1 = "predict-the-positions-and-speeds-of-600-satellites/answer_key.csv"
file_path_test2 = "predict-the-positions-and-speeds-of-600-satellites/jan_test.csv"
df_test1 = pd.read_csv(file_path_test1)
df_test2 = pd.read_csv(file_path_test2, **io_params).reset_index(drop=True) # drop=True throws away the old index
df_test = pd.concat([df_test2.iloc[:,0:2], df_test1, df_test2.iloc[:,2:]], axis=1)
print(f"There are {len(df_test.sat_id.unique())} unique satellites in the test dataset, with {len(df_test)} total rows.")

##1.2 Identify the number of satellites
num_sat = len(df_train.sat_id.unique())
#endregion

#region 2 Data Preprocessing
train = df_train
test = df_test

##2.1 Remove duplicate rows (epochs within a minute of each other)
dtypes = train.dtypes.to_dict()
cols_to_shift = train.columns.difference(['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim'])
train_intermediate = []
test_intermediate = []
for sat_id in tqdm(train['sat_id'].unique(), position=0):
    for dataset in (train, test):
        group = dataset.query('sat_id == @sat_id').copy()
        duplicates = group[group['epoch'].diff() < dt.timedelta(seconds=60)].index
    
        for i in reversed(duplicates):
            group.loc[i:, cols_to_shift] = group.loc[i:, cols_to_shift].shift(-1)
    
        group = group.drop(group[group['x'].isnull()].index)
        group['percent'] = np.arange(1, len(group)+1) / len(group)
    
        if dataset is train:
            train_intermediate.append(group)
        if dataset is test:
            test_intermediate.append(group)
    
train = pd.concat(train_intermediate).astype(dtypes)
train = train.assign(is_train = True)
test = pd.concat(test_intermediate).astype(dtypes)
test = test.assign(is_train = False)
print(f"Processed {len(train.sat_id.unique())} unique satellites, with {len(train)} total rows for training. Reduced by {len(df_train) - len(train)} rows.")
print(f"Processed {len(test.sat_id.unique())} unique satellites, with {len(test)} total rows for testing. Reduced by {len(df_test) - len(test)} rows.")
    
df = pd.concat([train, test], axis=0)
df = df.sort_values(['sat_id', 'epoch'])
#endregion
    
    
#region 3 Data Validation
for data_split, data in [('train', train), ('train', test)]:
    ##3.1 Check that the epoch is monotonically increasing and unique for each satellite
    assert all([sat_group['epoch'].is_monotonic_increasing for _, sat_group in data.groupby('sat_id')]), f"Epoch is not monotonic increasing in the {data_split} dataset."
    assert all([sat_group['epoch'].is_unique for _, sat_group in data.groupby('sat_id')]), f"Epoch is not unique in the {data_split} dataset."
    
    ##3.2 Check that the epoch is consistent for each satellite
    for sat_id, sat_group in data.groupby('sat_id'):
        intervals = sat_group['epoch'].diff().dropna().dt.total_seconds().unique().tolist()
        if len(intervals) > 3:
            print(f"There are {len(intervals)} unique intervals for sat_id {sat_id} in the {data_split} dataset.")
        cartesian_product = list(itertools.product(intervals, repeat = 2))
        assert all(list(map(lambda pair: (pair[0] - pair[1]) < 0.01, cartesian_product))), f'Intervals are not consistent: there are {len(intervals)} unique intervals for sat_id {sat_id} in the {data_split} dataset.'
#endregion

with open('validated_data.pkl', 'wb') as f:
    pkl.dump(df, f)
    