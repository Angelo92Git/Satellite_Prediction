import pickle as pkl
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from matplotlib import pyplot as plt
import my_metrics as mm
import my_utils as mu

with open('validated_data.pkl', 'rb') as f:
    df = pkl.load(f)

my_mod = mu.fit_AR_model(df, 310, 'x', lags=35)
mu.plot_AR_forecast(df, 310, 'x', my_mod)

# Declare tracking variables
var_index = {'x': 0, 'y': 1, 'z': 2, 'Vx': 3, 'Vy': 4, 'Vz': 5}
accuracy_dict = {}
accuracy_tensor = np.full((len(df['sat_id'].unique()), 6), np.nan)

for sat_id, sat_group in tqdm(df.groupby('sat_id'), position=0, leave=True, desc = 'Satellites'):
    sat_group = mu.set_epoch_as_index_and_freq(sat_group)
    accuracy_dict[sat_id] = {}

    train = sat_group.query('is_train == True')
    test = sat_group.query('is_train == False')
    for var in tqdm(['x', 'y', 'z', 'Vx', 'Vy', 'Vz'], position=1, leave=False, desc = 'Variables'):
        model = AutoReg(train[var], lags=35, trend = 'ct')
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))

        accuracy = 100*(1 - mm.smape(test[var].to_numpy(), forecast.to_numpy()))
        accuracy_dict[sat_id][var] = accuracy
        accuracy_tensor[sat_id][var_index[var]] = accuracy
    assert (not np.any(np.isnan(accuracy_tensor[sat_id,:]))), f"NaNs in accuracy_tensor for sat_id {sat_id}"

assert (not np.any(np.isnan(accuracy_tensor))), f"NaNs in accuracy_tensor"

#TODO vec_accuracy
#TODO: Cross-validation
#TODO: VAR_model
#TODO: DeepAR_model