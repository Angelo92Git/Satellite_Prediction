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

# my_mod = mu.fit_AR_model(df, 310, 'x', lags=35)
# mu.plot_AR_forecast(df, 310, 'x', my_mod)

cross_validate = False
if cross_validate:
#region Cross-validation
    #----------------------------------------------------------------------------------------
    # Cross-validation
    #----------------------------------------------------------------------------------------
    # Assume that the same lag should be used for all variables
    # id = 598 #DEBUG Remove
    # df = df.query(f'sat_id >= {id}')#DEBUG Remove
    def CV(df, cv_func, num_fits=4, num_lags=50):
        best_smape_cv_scores = np.full(len(df['sat_id'].unique()), -np.inf)
        best_vec_smape_cv_scores = np.full(len(df['sat_id'].unique()), -np.inf)
        smape_best_lags = np.full(len(df['sat_id'].unique()), np.nan)
        vec_smape_best_lags = np.full(len(df['sat_id'].unique()), np.nan)
        for sat_id, sat_group in tqdm(df.groupby('sat_id'), position=0, leave=True, desc='Cross-validation'):
            # sat_id = sat_id - id #DEBUG Remove
            sat_group = mu.set_epoch_as_index_and_freq(sat_group)
            train = sat_group.query('is_train == True')
            for lags in tqdm(np.arange(1, num_lags+1), position=1, leave=False, desc = 'Lags'):
                cv_smape_accuracy = []
                all_vec_targets = []
                all_vec_forecasts = []
                for cv_train, cv_val in cv_func(train, num_fits=num_fits):
                    if len(cv_train)-lags <= lags+2:
                        continue
                    all_vec_targets.append(cv_val[['x', 'y', 'z', 'Vx', 'Vy', 'Vz']].to_numpy().reshape(-1,6)) 
                    var_vec_forecasts = []
                    for var in tqdm(['x', 'y', 'z', 'Vx', 'Vy', 'Vz'], position=2, leave=False, desc = 'Variables'):
                        cv_model = AutoReg(cv_train[var], lags=lags, trend = 'ct')
                        cv_model_fit = cv_model.fit()
                        cv_forecast = cv_model_fit.forecast(steps=len(cv_val))
                        cv_smape_accuracy.append(mu.get_percent_accuracy(cv_val[var], cv_forecast, mm.smape))

                        var_vec_forecasts.append(cv_forecast.to_numpy().reshape(-1,1))

                    fold_vec_forecasts = np.concatenate(var_vec_forecasts, axis=1)
                    all_vec_forecasts.append(fold_vec_forecasts)

                if not cv_smape_accuracy or not all_vec_targets: # Check if empty
                    continue

                cv_smape_accuracy_lag = np.mean(cv_smape_accuracy)
                if  cv_smape_accuracy_lag > best_smape_cv_scores[sat_id]:
                    best_smape_cv_scores[sat_id] = cv_smape_accuracy_lag
                    smape_best_lags[sat_id] = lags

                all_vec_targets = np.concatenate(all_vec_targets, axis=0)
                all_vec_forecasts = np.concatenate(all_vec_forecasts, axis=0)
                cv_vec_smape_accuracy_lag = mu.get_percent_accuracy(all_vec_targets, all_vec_forecasts, mm.vector_smape)
                if cv_vec_smape_accuracy_lag > best_vec_smape_cv_scores[sat_id]:
                    best_vec_smape_cv_scores[sat_id] = cv_vec_smape_accuracy_lag
                    vec_smape_best_lags[sat_id] = lags   

        return smape_best_lags, vec_smape_best_lags


    ecv_smape_best_lags, ecv_vec_smape_best_lags = CV(df, mu.expanding_window_cv)
    # Save ECV
    with open('Results/expanding_window_smape_best_lags.pkl', 'wb') as f:
        pkl.dump(ecv_smape_best_lags, f)

    with open('Results/expanding_window_vec_smape_best_lags.pkl', 'wb') as f:
        pkl.dump(ecv_vec_smape_best_lags, f)

    scv_smape_best_lags, scv_vec_smape_best_lags = CV(df, mu.sliding_window_cv)
    # Save SCV

    with open('Results/sliding_window_smape_best_lags.pkl', 'wb') as f:
        pkl.dump(scv_smape_best_lags, f)

    with open('Results/sliding_window_vec_smape_best_lags.pkl', 'wb') as f:
        pkl.dump(scv_vec_smape_best_lags, f)
#endregion

with open('Results/expanding_window_smape_best_lags.pkl', 'rb') as f:
    ecv_vec_smape_best_lags = pkl.load(f)

# #--------------------------------------------------------------------------------------------------
# # Declare tracking variables
# #--------------------------------------------------------------------------------------------------
var_index = {'x': 0, 'y': 1, 'z': 2, 'Vx': 3, 'Vy': 4, 'Vz': 5}
vars = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
# SMAPE scores directly from AR model forecast
smape_accuracy_dict = {} # Used for easy readability

# dim 600 x 6
smape_accuracy_tensor = np.full((len(df['sat_id'].unique()), 6), np.nan) # Used for exploration, 6 univariate timeseries

# Vector SMAPE scores from AR model forecast
# dim 600 x 1
vec_smape_accuracy_tensor = np.full((len(df['sat_id'].unique()), 1), np.nan) #  Used for exploration, 1 multivariate timeseries
# Store variable forecasts in a dictionary for vector SMAPE
var_forecasts_dict = {}

#--------------------------------------------------------------------------------------------------
# Iterate over satellites and forecast
#--------------------------------------------------------------------------------------------------
for sat_id, sat_group in tqdm(df.groupby('sat_id'), position=0, leave=True, desc = 'Satellites'):
    # Prepare data and set store keys
    sat_group = mu.set_epoch_as_index_and_freq(sat_group)
    smape_accuracy_dict[sat_id] = {}
    var_forecasts_dict[sat_id] = {}
    best_lag = ecv_vec_smape_best_lags[sat_id]
    if best_lag < 35:
        best_lag = 35 # Due to the scarcity of data for some satellites, we weren't able to test lags > 35 for all satellites
        # When this happens, we use the maximum lag that was tested for all satellites 

    train = sat_group.query('is_train == True')
    test = sat_group.query('is_train == False')
    for var in tqdm(['x', 'y', 'z', 'Vx', 'Vy', 'Vz'], position=1, leave=False, desc = 'Variables'):
        model = AutoReg(train[var], lags=best_lag, trend = 'ct')
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        # Store forecasts for vector SMAPE
        var_forecasts_dict[sat_id][var] = forecast

        # Calculate SMAPE accuracy per satellite and per variable and store in dictionary and tensor
        smape_accuracy = mu.get_percent_accuracy(test[var], forecast, mm.smape)
        smape_accuracy_dict[sat_id][var] = smape_accuracy
        smape_accuracy_tensor[sat_id][var_index[var]] = smape_accuracy
    assert (not np.any(np.isnan(smape_accuracy_tensor[sat_id,:]))), f"NaNs in smape_accuracy_tensor for sat_id {sat_id}"

    forecasts_array = np.array([var_forecasts_dict[sat_id][var] for var in vars]).T
    vec_smape_accuracy_tensor[sat_id] = mu.get_percent_accuracy(test[vars].to_numpy(), forecasts_array, mm.vector_smape)
    assert (not np.isnan(vec_smape_accuracy_tensor[sat_id])), f"NaNs in vec_smape_accuracy_tensor for sat_id {sat_id}"

# Check for NaNs and negative values
assert (not np.any(np.isnan(smape_accuracy_tensor))), f"NaNs in smape_accuracy_tensor"
assert (not np.any(np.isnan(vec_smape_accuracy_tensor))), f"NaNs in vec_smape_accuracy_tensor"
assert (np.any(smape_accuracy_tensor >= 0)), f"Negative values in smape_accuracy_tensor"
assert (np.any(vec_smape_accuracy_tensor >= 0)), f"Negative values in vec_smape_accuracy_tensor"

# Save results
with open('Results/AR_ct_accuracy_best_lags.pkl', 'wb') as f:
    pkl.dump(smape_accuracy_tensor, f)

with open('Results/AR_ct_vec_accuracy_best_lags.pkl', 'wb') as f:
    pkl.dump(vec_smape_accuracy_tensor, f)

# TODO: VAR_model
# TODO: DeepAR_model