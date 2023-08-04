import pickle as pkl
import my_utils as mu
from matplotlib import pyplot as plt

with open('validated_data.pkl', 'rb') as f:
    df = pkl.load(f) 

with open('Results/expanding_window_vec_smape_best_lags.pkl', 'rb') as f:
    best_lags = pkl.load(f)

model_fitted50 = mu.fit_AR_model(df, 0, 'x', 50)
model_fitted75 = mu.fit_AR_model(df, 0, 'x', 75)

mu.plot_AR_forecast(df, 0, 'x', model_fitted50, 'lags50')
mu.plot_AR_forecast(df, 0, 'x', model_fitted75, 'lags75')

# mu.acf(df, 0, 'x', 160)
# mu.pcf(df, 310, 'x', 82)

# mu.plot_CV_fig(df, 310, mu.holdout_cv, method="OOS", method_c = "blue", num_fits=0.75)
# mu.plot_CV_fig(df, 310, mu.sliding_window_cv, method="Sliding window", method_c="orange", num_fits=4)
# mu.plot_CV_fig(df, 310, mu.expanding_window_cv, method="Expanding window", method_c="green", num_fits=4)
# plt.savefig('Results/CV_fig_310.png')

# plt.plot(best_lags)
# plt.show()

