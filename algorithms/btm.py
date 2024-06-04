# Implementation for beat the mean

import numpy as np
from math import log, sqrt
import random


class BTM:
    def __init__(self, K, T, compare_func, regret_func):
        self.T = T
        self.K = K
        self.W = np.arange(K)
        self.plays = np.zeros((K, K))  # n_b
        self.wins = np.zeros((K, K))  # w_b
        self.compare_fn = compare_func
        self.regrets = []
        self.regret_fn = regret_func

    def get_confidence_bound(self, n, gamma):
        if n == 0:
            c_star = 1
        else:
            c_star = self.c_del_gamma(n, gamma)

        return c_star

    def c_del_gamma(self, n, gamma):
        delta = 1 / (2 * self.T * self.K)
        return 3 * gamma * gamma * sqrt(log(1 / delta) / n)

    def get_p_hat(self):
        w = np.sum(self.wins, axis=1)
        n = np.sum(self.plays, axis=1)
        p_hat = {i: (w[i] / n[i] if n[i] > 0 else 0.5) for i in self.W}
        return p_hat

    def run(self, gamma=1.2):
        regret = 0
        n_star = np.min(np.sum(self.plays, axis=1))
        p_hat = self.get_p_hat()
        N = 10**15

        t = 0
        while len(self.W) > 1 and t < self.T and n_star < N:
            n_star = np.min(np.sum(self.plays, axis=1))
            c_star = self.get_confidence_bound(n_star, gamma)

            # get next pair
            b = np.argmin(np.sum(self.plays, axis=1))
            b_prime = np.setdiff1d(self.W, np.array([b]))[
                np.random.randint(len(self.W) - 1)]

            # compare
            result = self.compare_fn(b, b_prime)

            # update
            self.wins[b][b_prime] += result
            self.plays[b][b_prime] += 1

            t += 1
            regret += self.regret_fn(b, b_prime)
            self.regrets.append(regret)

            p_hat = self.get_p_hat()
            p_min_index = min(p_hat, key=p_hat.get)
            p_max_index = max(p_hat, key=p_hat.get)

            # remove bad arm
            if p_hat[p_min_index] + c_star < p_hat[p_max_index] - c_star:
                b_prime = p_min_index
                self.wins[:, b_prime] = 0
                self.plays[:, b_prime] = 0
                self.W = np.setdiff1d(self.W, np.array(b_prime))

        best_arm = max(p_hat, key=p_hat.get)
        return best_arm, self.regrets
