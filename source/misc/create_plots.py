# create plots from logs after training for presentation
# internal, non-cleaned up code

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# seaborn style is nice
plt.style.use('seaborn')
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('axes', labelsize=16)
matplotlib.rc('legend', fontsize=12)

img_type = "png"


def load_result(config, agent, online, run):
    on_off = "online" if online else "offline"
    path = os.path.join("data", config, "logs", f"{agent}_{on_off}_{run}.csv")
    return np.genfromtxt(path, delimiter=';', skip_header=1)

def get_info(config):
    path = os.path.join("data", config, "logs", f"DQN_online_1_info.csv")

    with open(path, "r") as f:
        for line in f:
            splits = line.split(";")
            eval, behavioral = float(splits[2].strip()), float(splits[3].strip())
            break
    return (eval, behavioral)


experiments = ["breakout_agarwal", "breakout_fujimoto_2500000", "breakout_fujimoto_25000", "breakout_fujimoto_1"]
algos = ["DQN", "QRDQN", "REM", "BCQ"]


"""
load online results
"""

online = []
for experiment in experiments:
    online.append(load_result(experiment, "DQN", True, 1))
online = np.stack(online, axis=0)


"""
First, compare all experiments in breakout_agarwal and fujimoto_2500000
"""
for experiment in experiments:

    results = []
    for algo in algos:
        rslt = []
        for run in range(1,4):
            rslt.append(load_result(experiment, algo, False, run)[:,:2])
        results.append(np.stack(rslt, axis=0))
    results = np.stack(results, axis=0)

    (eval, behavioral) = get_info(experiment)

    plt.figure(figsize=(8,6))
    plt.hlines(y=online[:,:,1].mean(axis=0).max() , xmin=-100000, xmax=10100000, color="black")
    est = np.mean(online[:, :, 1], axis=0)
    sd = np.std(online[:, :, 1], axis=0)
    cis = (est - sd, est + sd)
    plt.fill_between(online[0, :, 0], cis[0], cis[1], alpha=0.2, color="black")
    plt.plot(online[:, :, 0].mean(axis=0), online[:, :, 1].mean(axis=0), color="black", label="Online DQN")
    for algo in range(len(algos)):
        est = np.mean(results[algo,:,:,1], axis=0)
        sd = np.std(results[algo,:,:,1], axis=0)
        cis = (est - sd, est + sd)
        plt.fill_between(results[algo,0,:,0], cis[0], cis[1], alpha=0.2)
        plt.plot(results[algo,0,:,0], est, label=algos[algo])
    plt.xticks(range(0, 10010000, 1000000))
    plt.xlim(left=-100000, right=10100000)
    ax = plt.gca()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(6,6))
    plt.ylabel("Reward")
    plt.xlabel("Training steps")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), fancybox=True,
               shadow=True, ncol=5, frameon=True, handlelength=1)
    plt.savefig(os.path.join("results", "presentation", f"{experiment}.{img_type}"), bbox_inches='tight')
    plt.show()


"""
Value estimates
"""

for experiment in experiments:

    results = []
    for algo in algos:
        rslt = []
        for run in range(1,4):
            rslt.append(load_result(experiment, algo, False, run))
        results.append(np.stack(rslt, axis=0))
    results = np.stack(results, axis=0)

    plt.figure(figsize=(8,6))
    for algo in range(-1, len(algos)):
        if algo == -1:
            est = np.mean(online[:, :, 4], axis=0)
            sd = np.std(online[:, :, 4], axis=0)
            cis = (est - sd, est + sd)
            plt.fill_between(online[0, :, 0], cis[0], cis[1], alpha=0.2, color="black")
            plt.plot(online[0, :, 0], est, label="Online DQN", color="black")
        else:
            est = np.mean(results[algo,:,:,4], axis=0)
            sd = np.std(results[algo,:,:,4], axis=0)
            cis = (est - sd, est + sd)
            plt.fill_between(results[algo,0,:,0], cis[0], cis[1], alpha=0.2)
            plt.plot(results[algo,0,:,0], est, label=algos[algo])
    plt.xticks(range(0, 10010000, 1000000))
    plt.xlim(left=-100000, right=10100000)
    plt.ylim(bottom=-0.1, top=10.1)
    ax = plt.gca()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(6,6))
    plt.ylabel("Value estimate")
    plt.xlabel("Training steps")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), fancybox=True,
               shadow=True, ncol=5, frameon=True, handlelength=1)
    plt.savefig(os.path.join("results", "presentation", f"{experiment}_value.{img_type}"))
    plt.show()

"""
Ablation study
"""
labels = ["ER buffer", "2.500.000 policies", "25.000 policies", "1 policy"]

for algo in range(len(algos)):

    results = []
    for experiment in experiments:
        rslt = []
        for run in range(1,4):
            rslt.append(load_result(experiment, algos[algo], False, run))
        results.append(np.stack(rslt, axis=0))
    results = np.stack(results, axis=0)

    plt.figure(figsize=(8,6))

    plt.hlines(y=online[:, :, 1].mean(axis=0).max(), xmin=-100000, xmax=10100000, color="black")
    """
    est = np.mean(online[:, :, 1], axis=0)
    sd = np.std(online[:, :, 1], axis=0)
    cis = (est - sd, est + sd)
    plt.fill_between(online[0, :, 0], cis[0], cis[1], alpha=0.2, color="black")
    plt.plot(online[:, :, 0].mean(axis=0), online[:, :, 1].mean(axis=0), color="black", label="Online DQN")
    """
    for i, experiment in enumerate(experiments):
            est = np.mean(results[i,:,:,1], axis=0)
            sd = np.std(results[i,:,:,1], axis=0)
            cis = (est - sd, est + sd)
            plt.fill_between(results[i,0,:,0], cis[0], cis[1], alpha=0.2)
            plt.plot(results[i,0,:,0], est, label=labels[i])
    plt.xticks(range(0, 10010000, 1000000))
    plt.xlim(left=-100000, right=10100000)
    plt.ylim(bottom=-2, top=46)
    #plt.title(algos[algo], loc='center', fontsize=18)
    ax = plt.gca()
    ax.set_title(algos[algo], y=1.0, pad=-30, fontsize=20, fontweight="bold")
    ax.ticklabel_format(axis='x', style='sci', scilimits=(6,6))
    plt.ylabel("Reward")
    plt.xlabel("Training steps")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), fancybox=True,
               shadow=True, ncol=5, frameon=True, handlelength=1)
    plt.savefig(os.path.join("results", "presentation", f"{algos[algo]}_ablation.{img_type}"))
    plt.show()