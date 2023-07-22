import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from matplotlib import pyplot as plt
import numpy as np

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
    accuracy = 100*(1 - metric_func(target.to_numpy(), forecast.to_numpy()))
    return accuracy

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
    plt.ylim(center - 0.6*height, center + 0.6*height)
    plt.xlim(t_all.iloc[0], t_all.iloc[-1])
    plt.plot([t_train.iloc[-1], t_train.iloc[-1]],[-70000 , 70000], color = 'k')
    plt.tight_layout()
    plt.show()
    return