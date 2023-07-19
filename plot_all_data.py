import pickle as pkl 
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime as dt
import numpy as np

with open('validated_data.pkl', 'rb') as f:
    df = pkl.load(f)

print('\nData head entries:')
print(df.head(), '\n')
print(f"Columns: {(', ').join(list(df.columns))}\n")

def vars_plotting_subplot(var, axs, sat_id, sat_group, plot_group_size):
    sub_plot_index = sat_id % plot_group_size # Equal to zero here (first subplot)
    train_group = sat_group.query('is_train == True')
    test_group = sat_group.query('is_train == False')
    axs[sub_plot_index].plot(train_group['epoch'], train_group[var], label = 'train')
    axs[sub_plot_index].plot(test_group['epoch'], test_group[var], label = 'test')
    axs[sub_plot_index].set_xlabel('Epoch')
    axs[sub_plot_index].set_ylabel(var)
    axs[sub_plot_index].tick_params(axis='x', rotation=0)
    if sub_plot_index == 0:
        axs[sub_plot_index].legend(loc='upper right')
    return


def plot_by_variables(df):
    for var in tqdm(['x', 'y', 'z', 'Vx', 'Vy', 'Vz'], position=0):
        for sat_id, sat_group in tqdm(df.groupby('sat_id'), position=1):
            # Create subplots every <plot_group> satellites
            plot_group_size = 10
            if sat_id % plot_group_size == 0:
                # Set Variables for top-level plot
                main_plot_index = sat_id // plot_group_size
                fig, axs = plt.subplots(plot_group_size, 1, figsize=(20,20))
                fig.suptitle(f"State: {var} Satellites: {main_plot_index} to {main_plot_index + plot_group_size - 1}", fontsize=20)

                # Plot the data for the first subplot
                vars_plotting_subplot(var, axs, sat_id, sat_group, plot_group_size)
            else:
                vars_plotting_subplot(var, axs, sat_id, sat_group, plot_group_size)
            if sat_id == (main_plot_index*plot_group_size + plot_group_size - 1):
                plt.tight_layout()
                plt.subplots_adjust(top=0.95)
                plt.savefig(f'Plots/var_{var}/sat_id_{main_plot_index}_to_{main_plot_index + plot_group_size - 1}_var_{var}.png')
                plt.close('all')
    return

def plot_by_satellite(df):
    num_states = 6
    vars = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
    for sat_id, sat_group in tqdm(df.groupby('sat_id'), position=0):
        fig, axs = plt.subplots(num_states, 1, figsize=(20,20), sharex=True)
        fig.suptitle(f"Satellite: {sat_id}", fontsize=20)
        train_group = sat_group.query('is_train == True')
        test_group = sat_group.query('is_train == False')
        for i in range(len(axs)):
            axs[i].plot(train_group['epoch'], train_group[vars[i]], label = 'train')
            axs[i].plot(test_group['epoch'], test_group[vars[i]], label = 'test')
            axs[i].set_xlabel('Epoch')
            axs[i].tick_params(axis='x', rotation=0)
            axs[i].set_ylabel(vars[i])
            if i == 0:
                axs[i].legend(loc='upper right')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(f'Plots/sat_plots/sat_id_{sat_id}.png')
        plt.close('all')

def plot_custom_select(df):
    sat_id = input("Enter satellite id to plot: ")
    assert int(sat_id) in df['sat_id'].unique(), f"Satellite id {sat_id} not in dataset."
    var = input("Enter variable to plot: ")
    assert var in ['x', 'y', 'z', 'Vx', 'Vy', 'Vz'], f"Variable {var} not in dataset."
    sat_group = df.query(f'sat_id == {sat_id}')
    train = sat_group.query('is_train == True')
    test = sat_group.query('is_train == False')
    plt.figure(figsize=(20,15))
    plt.plot(train['epoch'], train[var], label = 'train')
    plt.plot(test['epoch'], test[var], label = 'test')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel(var, fontsize=20)
    plt.tick_params(axis='x', rotation=45)
    plt.legend(loc='upper right', fontsize=20)
    plt.title(f"Satellite: {sat_id}, Variable: {var}", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'Plots/sat_id_{sat_id}_var_{var}.png')
    plt.close('all')
    return

def plot_3d(df):
    sat_id = int(input("Enter satellite id to plot: "))
    assert sat_id in df['sat_id'].unique(), f"Satellite id {sat_id} not in dataset."
    sat_group = df.query(f'sat_id == {sat_id}')
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    ref_time = sat_group['epoch'].min()
    c_grad = np.arange(len(sat_group['epoch']))
    ax.scatter(sat_group['x'], sat_group['y'], sat_group['z'], c=c_grad, marker='o', s=10**2, cmap='jet')
    ax.plot(sat_group['x'], sat_group['y'], sat_group['z'], color = 'grey')
    ax.set_xlabel('x', fontsize=30)
    ax.set_ylabel('y', fontsize=30)
    ax.set_zlabel('z', fontsize=30)
    ax.set_title(f"Satellite: {sat_id} orbit", fontsize=40)
    plt.tight_layout()
    plt.savefig(f'Plots/sat_id_{sat_id}_3d.png')
    plt.close('all')
    return

def plot_matrix(df):
    vars = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
    sat_id = int(input("Enter satellite id to plot: "))
    assert sat_id in df['sat_id'].unique(), f"Satellite id {sat_id} not in dataset."
    sat_group = df.query(f'sat_id == {sat_id}')
    pd.plotting.scatter_matrix(sat_group[vars], alpha=0.2, figsize=(9, 9), diagonal='kde')
    plt.suptitle(f"Satellite: {sat_id} orbit pair plot", fontsize=15)
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"Plots/sat_id_{sat_id}_matrix.png")
    plt.close('all')
    pass

# Main execution
plot_by = input("Plot by variables or by satellite or custom_select or 3D plot or pair plot? (v/s/c/3d/p): ")
if plot_by == 'v':
    plot_by_variables(df)
elif plot_by == 's':
    plot_by_satellite(df)
elif plot_by == 'c':
    plot_custom_select(df)
elif plot_by == '3d':
    plot_3d(df)
elif plot_by == 'p':
    plot_matrix(df)
else:
    print("Invalid input. Exiting.")




