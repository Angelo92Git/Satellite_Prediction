# Things to try

1. Darts and statsmodels are useful
1. AR models, DeepAR
1. Grid Search for ARmodel
1. VAR model
1. Grid Search for VAR model
1. Hold-out CV
1. Expanding window CV
1. Rolling window CV (Blocked CV)
1. LSTM or some complicated model
1. Justify loss functions using <https://lavinei.github.io/pybats/loss_functions.html>
1. Kalman filter is a filtering model for smoothing the series, not a forecasting model
1. read model_fit.summary()

## Notes

1. AR fits using Least squares so the model is naive in this case, the final evaluation is with SMAPE
1. Error accumulation over time
1. Satellite 252 has very few data points
    - The maximum number of lags that work is 35
1. Add a constant and time-trend when using AR (satellies have linear trends)
1. Use the same lag for all variables
1. The worst smape loss satellites with lag 45 are:
   - 249 (high freqeuency, some linear trend)
   - 37 (consistant across methods at 74%, looks like there is some linear trend with time, high frequency)
   - 301 (pinched, very low fequency), very low accuracy across the board
1. No model variance since the data generation process is deterministic, i.e. no error term.
1. Histogram of scores
1. Histogram of scores after differencing velocities
1. minimum size of train set = num lags*2+1
1. SMAPE is used to evaluate the model and used in CV, but not to fit the model.
1. No smoothing is done on the data i.e. no moving average, no exponential smoothing, no Kalman filter
1. Differencing could be implemented to gett better results for the velocities
1. In cases of data scarcity expanding window CV is better than sliding window CV