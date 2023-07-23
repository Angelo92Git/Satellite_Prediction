import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------------------
# SMAPE
#----------------------------------------------------------------------------------------

models = ['AR_ct']
vars = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
trend_accuracy = {}
for trend in models:
    with open(f'Results/{trend}_accuracy_best_lags.pkl', 'rb') as f:
        trend_accuracy[trend] = pkl.load(f)

fig, axs = plt.subplots(len(models), len(vars), figsize=(20, 7), sharey=True)
fig.suptitle('SMAPE Accuracy Histograms for AR (Larger is better)')
axs[0].set_ylabel(f'Log satellite counts')
plt.setp(axs[1:], ylabel='')
for i, trend in enumerate(models):
    for j, var in enumerate(vars):
        axs[j].hist(trend_accuracy[trend][:,i], bins=50)
        axs[j].set_yscale('log')
        axs[j].set_xlabel('Accuracy %')
        axs[j].set_title(f'{var}')
        plt.tight_layout()

plt.savefig('Results/SMAPE_AR_accuracy_histograms_best_lags.png')
plt.close('all')

fig, axs = plt.subplots(len(vars), 1, figsize=(6, 8), sharex=True)
axs[len(vars)-1].set_xlabel('Sat_id')

for i, var in enumerate(vars):
    axs[i].plot(trend_accuracy['AR_ct'][:,i], linestyle=None, marker='x', markersize=10, color='black')
    axs[i].set_ylabel(f'{var}')
    axs[i].set_title(f'{var}')
    plt.tight_layout()

# plt.show()
# plt.savefig('Results/sat_id_with_scores_best_lags.png')
plt.close('all')

#----------------------------------------------------------------------------------------
# Vector SMAPE
#----------------------------------------------------------------------------------------

with open(f'Results/AR_ct_vec_accuracy_best_lags.pkl', 'rb') as f:
    vec_accuracy = pkl.load(f)

plt.hist(vec_accuracy, bins=50)
plt.yscale('log')
plt.ylabel(f'Log satellite counts')
plt.xlabel('Accuracy %')
plt.title(f'Vector SMAPE Accuracy Histograms for AR (Larger is better)')
plt.tight_layout()
plt.savefig('Results/Vector_SMAPE_AR_accuracy_histograms_best_lags.png')