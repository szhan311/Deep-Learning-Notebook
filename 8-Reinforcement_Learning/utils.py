'''
Author: Shaorong Zhang
Email: szhan311@ucr.edu

'''
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rewards(rewards, ma_rewards, plot_cfg, tag='train'):
    '''
    rewards: rewards
    ma_rewards: moving average rewards
    '''
    sns.set()
    plt.figure(dpi=100)  
    plt.title("learning curve on {} of {} for {}".format(
        plot_cfg.device, plot_cfg.algo, plot_cfg.env))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path+"{}_rewards_curve".format(tag))
    plt.show()

def save_results(rewards, ma_rewards, tag='train', path='./results'):
    ''' Save rewards
    '''
    np.save(path+'{}_rewards.npy'.format(tag), rewards)
    np.save(path+'{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Result saved')

def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path+"losses_curve")
    plt.show()

def make_dir(*paths):
    ''' Make directory
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def del_empty_dir(*paths):
    ''' Delete all the empty directories
    '''
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))