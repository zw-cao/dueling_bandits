# Implementation for relative upper confidence bound algorithm
import numpy as np
from math import log
import random

class RUCB():
    def __init__(self, K, T, alpha, compare_func, regret_func):
        self.T = T
        self.K = K
        self.W = np.arange(K)
        self.U = np.zeros((K, K))
        self.wins = np.zeros((K, K))  # i beats j
        self.compare_fn = compare_func
        self.regrets = []
        self.regret = 0
        self.regret_fn = regret_func
        self.alpha = alpha
        self.C = []
        self.B = []

    def update_U(self, t):
        for i in range(self.K):
            for j in range(self.K):
                wwT = float(self.wins[i][j] + self.wins.T[i][j])
                if wwT == 0:
                    left = 1
                    right = 1
                else:
                    left = float(self.wins[i][j]) / wwT
                    right = np.sqrt(float(self.alpha * log(t))/wwT)

                self.U[i][j] = left + right
        for i in range(self.K):
            self.U[i][i] = 0.5

    def create_C(self):
        return np.where(np.all(self.U >= 0.5, axis=1))[0]

    def draw_from_C(self):
        size_C = len(self.C)
        size_B = len(self.B)
        p = np.zeros(size_C)

        for idx, element in enumerate(self.C):
            if element in self.B:
                p[idx] = 0.5
            else:
                p[idx] = 1 / (2 ** size_B * (size_C - size_B))

        p /= np.sum(p)
        return np.random.choice(self.C, p=p)

    def get_best_arm(self) -> int:
        return np.argmax(np.sum(self.wins, axis=1))

    def select_arms(self, t):
        self.update_U(t)
        self.create_C()

        # choose left arm i
        if len(self.C) == 0:
            i = random.choice(range(self.K))
            self.B = np.append(self.B, self.C)
        elif len(self.C) == 1:
            i = self.C[0]
            self.B = self.C
        else:
            i = self.draw_from_C()

        # choose right arm j
        max_value = np.max(self.U[:, i])
        indices_of_max = np.where(self.U[:, i] == max_value)[0]

        # if there is a tie, i is not allowed to be equal to j.
        if i in indices_of_max and len(indices_of_max) > 1:
            j = i
            while j == i:
                j = random.choice(indices_of_max)
        else:
            j = random.choice(indices_of_max)

        return i, j

    def update_state(self, i, j, res):
        if res == 1:
            self.wins[i, j] += 1
        elif res == 0:
            self.wins[j, i] += 1

    def run(self):
        for t in range(self.T):
            i, j = self.select_arms(t)
            res = self.compare_fn(i, j)
            self.update_state(i, j, res)
            self.regret += self.regret_fn(i, j)
            self.regrets.append(self.regret)
        return self.get_best_arm(), self.regrets

