# Implementation for relative upper confidence bound algorithm

import numpy as np
from math import log, sqrt
import random


class IF():
    def __init__(self, K, T, compare_func, regret_func):
        self.T = T
        self.K = K
        self.W = np.arange(K)
        self.plays = np.zeros((K, K))  # n_b
        self.wins = np.zeros((K, K))  # w_b
        self.compare_fn = compare_func
        self.regrets = []
        self.regret_fn = regret_func
        self.regret = 0
        self.res = np.zeros((K, K))
        self.delta = 1 / (self.T * (self.K**2))

    def estimate_p_hat(self, i, j):
        if self.res[i][j] + self.res[j][i] == 0:
            return 0.5
        return self.res[i][j] / (self.res[i][j] + self.res[j][i])

    def is_confident(self, p_hat, i, j, eps=10**-12):
        t = self.res[i][j] + self.res[j][i] + eps
        c = np.sqrt(np.log(1 / self.delta) / t)
        return 0.5 < p_hat - c or 0.5 > p_hat + c

    def run(self):
        t = 0
        b_hat = random.choice(list(self.W))
        self.W = np.setdiff1d(self.W, np.array([b_hat]))
        p_hat = {}
        while len(self.W) and t < self.T:
            for b in self.W:
                # compare b_hat and b
                result = self.compare_fn(b_hat, b)
                self.res[b_hat][b] += result
                self.res[b][b_hat] += (1 - result)
                # update p_hat
                p_hat[b] = self.estimate_p_hat(b_hat, b)
                self.regret += self.regret_fn(b_hat, b)
                self.regrets.append(self.regret)
                t += 1

            to_remove = [b for b, p_hat in p_hat.items()
                         if p_hat > 0.5 and self.is_confident(p_hat, b_hat, b)]
            for rem in to_remove:
                p_hat.pop(rem)
                self.W = np.setdiff1d(self.W, np.array([rem]))

            candidates = [b for b, p_hat in p_hat.items()
                          if p_hat < 0.5 and self.is_confident(p_hat, b_hat, b)]

            if len(candidates):
                b_hat = candidates[0]
                self.W = np.setdiff1d(self.W, np.array([b_hat]))
                # reset p_hat
                p_hat = {}

        for _ in range(t, self.T):
            self.regret += self.regret_fn(b_hat, b_hat)
            self.regrets.append(self.regret)

        return b_hat, self.regrets[:self.T]

