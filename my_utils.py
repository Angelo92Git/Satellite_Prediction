import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import my_metrics as mm
import pickle as pkl

def set_epoch_as_index_and_freq(df: pd.DataFrame) -> pd.DataFrame:
    assert df['epoch'].diff()[1:].unique().shape == (1,), f"intervals not consistent for sat_id {df['sat_id'][0]}"
    freq = df['epoch'].diff().mode()[0]
    df.set_index('epoch', inplace=True)
    df.index.freq = f"{round(freq.total_seconds()*1000)}L"
    return df

def fit_AR_model(df, sat_id, var, lags=35, trend='ct'):
    sat_group = df.query(f'sat_id=={sat_id}')
    sat_group = set_epoch_as_index_and_freq(sat_group)
    train = sat_group.query('is_train == True')
    model = AutoReg(train[var], lags=lags, trend = trend)
    model_fit = model.fit()
    return model_fit

def get_percent_accuracy(target, forecast, metric_func):
    accuracy = 100*(1 - metric_func(target, forecast))
    return accuracy

def expanding_window_cv(train, num_fits=4):
    foldsize = len(train) // (num_fits+1) # The first fold is for training only, the rest are for validation as the folds get subsumed unto training at each step
    horizon = foldsize
    for i in range(num_fits):
        split_train = train.iloc[:(i+1)*foldsize]
        split_val = train.iloc[(i+1)*foldsize:(i+1)*foldsize+horizon]
        yield (split_train, split_val)

def sliding_window_cv(train, num_fits=4):
    foldsize = len(train) // num_fits
    windowsize = (foldsize*3//4) # training window
    horizon = foldsize - windowsize # validation window
    for i in range(num_fits):
        split_train = train.iloc[i*foldsize:i*foldsize+windowsize]
        split_val = train.iloc[i*foldsize+windowsize:i*foldsize+windowsize+horizon]
        yield (split_train, split_val)

def plot_AR_forecast(df, sat_id, var, model_fit):
    fig = plt.figure()
    t_all = df.query(f'sat_id=={sat_id}')['epoch']
    var_all = df.query(f'sat_id=={sat_id}')[var]
    t_train = df.query(f'sat_id=={sat_id} and is_train==True')['epoch']
    t_test = df.query(f'sat_id=={sat_id} and is_train==False')['epoch']
    var_train = df.query(f'sat_id=={sat_id} and is_train==True')[var]
    var_test = df.query(f'sat_id=={sat_id} and is_train==False')[var]
    plt.plot(t_train, var_train, marker = 'x', markeredgecolor = 'blue', label='train')
    plt.plot(t_test, var_test, marker = 'x', markeredgecolor = 'orange', label='test')
    preds_obj = model_fit.get_prediction(start = t_train.iloc[0], end = t_test.iloc[-1])
    preds = preds_obj.predicted_mean
    preds_int = preds_obj.conf_int(alpha=0.05)
    plt.plot(t_all, preds, 'red', label='AR model')
    plt.fill_between(t_all, preds_int['lower'], preds_int['upper'], color='k', alpha=0.2)
    plt.xlabel('Time')
    plt.ylabel(var)
    plt.title(f'AR model for sat_id {sat_id} and variable {var}')
    plt.legend(loc='upper right')
    center = np.mean(var_all)
    height = np.max(var_all) - np.min(var_all)
    plt.xticks(rotation=45)
    plt.plot([t_train.iloc[-1], t_train.iloc[-1]],[-90000 , 90000], color = 'k')
    plt.ylim(center - 0.6*height, center + 0.6*height)
    plt.xlim(t_all.iloc[0], t_all.iloc[-1])
    plt.tight_layout()
    plt.show()
    return

def plot_CV(df, sat_id, cv_func, num_fits=4, num_lags=50):
    sat_group = df.query(f'sat_id=={sat_id}')
    sat_group = set_epoch_as_index_and_freq(sat_group)
    train = sat_group.query('is_train == True')
    cv_smape_accuracy_lag = []
    cv_vec_smape_accuracy_lag = []
    lags_used = []
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
                cv_smape_accuracy.append(mm.smape(cv_val[var], cv_forecast))

                var_vec_forecasts.append(cv_forecast.to_numpy().reshape(-1,1))

            fold_vec_forecasts = np.concatenate(var_vec_forecasts, axis=1)
            all_vec_forecasts.append(fold_vec_forecasts)
        
        if not cv_smape_accuracy or not all_vec_targets: # Check if empty
            continue
        
        lags_used.append(lags)
        cv_smape_accuracy_lag.append(np.mean(cv_smape_accuracy))
        
        all_vec_targets = np.concatenate(all_vec_targets, axis=0)
        all_vec_forecasts = np.concatenate(all_vec_forecasts, axis=0)
        cv_vec_smape_accuracy_lag.append(mm.vector_smape(all_vec_targets, all_vec_forecasts))
    
    x_plot = lags_used
    y1_plot = cv_smape_accuracy_lag
    y2_plot = cv_vec_smape_accuracy_lag
    best_lag = np.argmin(y1_plot)+1
    plt.plot(x_plot, y1_plot, label='SMAPE')
    plt.plot(x_plot, y2_plot, label='Vector SMAPE')
    plt.xlabel('Lags')
    plt.ylabel('CV score')
    plt.title(f'Cross-validation score for sat_id {sat_id}')
    plt.legend(loc='upper right')
    plt.xlim(1, np.max(lags_used))
    plt.ylim(0, 1)
    plt.plot([best_lag, best_lag],[0 , 1], color = 'r')
    plt.tight_layout()
    plt.savefig(f'Results/CV_accuracy_{sat_id}.png')
    plt.close('all')
    return


# with open('validated_data.pkl', 'rb') as f:
#     df = pkl.load(f)

# plot_CV(df, 310, sliding_window_cv)