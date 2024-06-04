import numpy as np
import json

from matplotlib import pyplot as plt

from algorithms.btm import BTM
from algorithms.if1 import IF
from algorithms.rcs import RCS
from algorithms.doubler import Doubler
from algorithms.rucb import RUCB
from algorithms.rmed import RMED


T = 200000
samples = 10
best_arm = 0

def regret_fn(i: int, j: int) -> float:
    return pref[best_arm][i] + pref[best_arm][j] - 1


def compare_fn(i: int, j: int):
    i = int(i)
    j = int(j)
    return np.random.binomial(n=1, p=pref[i][j], size=1)

def run(ds_name):
    results = {"BTM": [], "Doubler": [], "IF": [], "RCS": [], "RMED1": [], "RMED2": [], "RUCB": []}

    regrets = np.zeros(T)
    for i in range(samples):
        x = BTM(len(pref), T, compare_fn, regret_fn)
        best_arm_BTM, reg_BTM = x.run()
        print("BTM done, best arm : ", best_arm_BTM)
        regrets = regrets + np.array(np.around(reg_BTM, 3))[:T]
    results["BTM"] = regrets / samples

    regrets = np.zeros(T)
    for i in range(samples):
        x = IF(len(pref), T, compare_fn, regret_fn)
        b_hat_IF, reg_IF = x.run()
        print("IF done, best arm : ", b_hat_IF)
        regrets = regrets + np.array(np.around(reg_IF, 3))[:T]
    results["IF"] = regrets / samples

    regrets = np.zeros(T)
    for i in range(samples):
        x = Doubler(T, pref, regret_fn)
        best_arm_Doubler, reg_Doubler = x.run()
        print("Doubler done, best arm : ", best_arm_Doubler)
        regrets = regrets + np.array(np.around(reg_Doubler, 3))[:T]
    results["Doubler"] = regrets / samples

    regrets = np.zeros(T)
    for i in range(samples):
        x = RUCB(len(pref), T, 0.5, compare_fn, regret_fn)
        best_arm_RUCB, reg_RUCB = x.run()
        print("RUCB done, best arm : ", best_arm_RUCB)
        regrets = regrets + np.array(np.around(reg_RUCB, 3))[:T]
    results["RUCB"] = regrets / samples

    regrets = np.zeros(T)
    for i in range(samples):
        x = RCS(len(pref), T, 0.5, pref, compare_fn, regret_fn)
        best_arm_RCS, reg_RCS = x.run()
        print("RCS done, best arm : ", best_arm_RCS)
        regrets = regrets + np.array(np.around(reg_RCS, 3))[:T]
    results["RCS"] = regrets / samples

    regrets = np.zeros(T)
    for i in range(samples):
        x = RMED('RMED1', len(pref), T, compare_fn, regret_fn)
        best_arm_RMED1, reg_RMED1, _ = x.run()
        print(len(reg_RMED1))
        print("RMED1 done, best arm : ", best_arm_RMED1)
        regrets = regrets + np.array(np.around(reg_RMED1, 3))[:T]
    results["RMED1"] = regrets / samples

    regrets = np.zeros(T)
    for i in range(samples):
        x = RMED('RMED2', len(pref), T, compare_fn, regret_fn)
        best_arm_RMED2, reg_RMED2, _ = x.run()
        print("RMED2 done, best arm : ", best_arm_RMED2)
        regrets = regrets + np.array(np.around(reg_RMED2, 3))[:T]
    results["RMED2"] = regrets / samples

    np.savez(f'./results/{ds_name}.npz', **results)

    plot(ds_name)
    # print(results)


# for ds_name in ['sushi16', 'mslr5']:
#     pref = np.load('./datasets/real/' + ds_name + '.npy')
#     run(ds_name)

# for ds_name in ['arithmetic', 'arxiv']:
#     pref = np.load('./datasets/synthetic/' + ds_name + '.npy')
#     run(ds_name)


def plot(ds_name):
    regrets = np.load('./results/' + ds_name + '.npz')
    print(type(regrets['BTM']))

    time_steps = np.arange(1, T + 1)

    keys = ["BTM", "IF", "Doubler", "RCS", "RMED1", "RMED2", "RUCB"]

    # Plotting regrets vs. time for each key

    for key in keys:
        plt.plot(time_steps, regrets[key], label=key)

    plt.xlabel('Time')
    plt.ylabel('Regrets')
    plt.title(f'Regrets vs. Time for Dataset {ds_name}')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


def plot_per_algorithm():
    arith = np.load('./results/arithmetic.npz')
    arxiv = np.load('./results/arxiv.npz')
    mslr5 = np.load('./results/mslr5.npz')
    sushi16 = np.load('./results/sushi16.npz')

    keys = ["BTM", "IF", "Doubler", "RCS", "RMED1", "RMED2", "RUCB"]

    for key in keys:
        time_steps = np.arange(1, T + 1)
        plt.plot(time_steps, arith[key], label='arithmetic')
        plt.plot(time_steps, arxiv[key], label='arxiv')
        plt.plot(time_steps, mslr5[key], label='mslr5')
        plt.plot(time_steps, sushi16[key], label='sushi16')
        plt.xlabel('Time')
        plt.ylabel('Regrets')
        plt.title(f'Regrets vs. Time for Algorithm {key}')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()


plot_per_algorithm()

# ds_names = ['mslr5', 'sushi16', 'arithmetic', 'arxiv']
# for ds in ds_names:
#     plot(ds)

