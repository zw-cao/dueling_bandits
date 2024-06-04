# Implementation for relative confidence sampling algorithm

import numpy as np
from math import log, sqrt
import random


class RCS:

    def __init__(self, K, T, alpha, pref, compare_func, regret_func):
        self.T = T  # total time
        self.K = K  # number of bandits
        self.W = np.arange(K)  # a set of bandits
        self.U = np.zeros((K, K))  # utility
        self.wins = np.zeros((K, K))  # i beats j
        self.plays = np.zeros((K, K))
        self.compare_fn = compare_func
        self.regrets = []
        self.regret = 0
        self.regret_fn = regret_func
        self.alpha = alpha
        self.theta = np.zeros((K, K))
        self.p = pref
        self.C = []
        self.B = []
        self.winner = None

    def update_U(self, t):
        for i in range(self.K):
            for j in range(self.K):
                wwT = float(self.wins[i][j] + self.wins.T[i][j])
                if wwT == 0:
                    left = 1
                    right = 1
                else:
                    left = float(self.wins[i][j]) / wwT
                    right = np.sqrt(float(self.alpha * log(t)) / wwT)

                self.U[i][j] = left + right
        for i in range(self.K):
            self.U[i][i] = 0.5

    def update_theta(self):
        for i in range(self.K):
            for j in range(self.K):
                if i < j:
                    self.theta[i][j] = random.betavariate(
                        alpha=(self.wins[i][j] + 1),
                        beta=(self.wins[j][i] + 1)
                    )
                    self.theta[j][i] = 1 - self.theta[i][j]

    def create_C(self):
        return np.where(np.all(self.U >= 0.5, axis=1))[0]

    def draw(self, i, j):
        p_value = self.p[i][j]

        if random.random() > p_value:
            return [0, 1]
        else:
            return [1, 0]

    def set_winner(self):
        candidates = np.where(np.all(self.U >= 0.5, axis=1))[0]
        if len(candidates) > 0:
            self.winner = random.choice(candidates)
        else:
            self.winner = self.get_least_played_arm()

    def choose_right_arm(self, i):
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
        return j

    def get_least_played_arm(self):
        min_value = np.min(np.sum(self.plays, axis=1))
        arms_least_played = np.where(self.plays == min_value)[0]
        return random.choice(arms_least_played)

    def select_arms(self, t):
        self.update_U(t)
        self.update_theta()

        self.set_winner()

        i = self.winner
        j = self.choose_right_arm(i)

        return i, j

    def update_state(self, i, j, res):
        if res == 1:
            self.wins[i, j] += 1
        elif res == 0:
            self.wins[j, i] += 1

    def get_best_arm(self) -> int:
        return np.argmax(np.sum(self.wins, axis=1))

    def run(self):
        for i in range(self.K):
            self.U[i][i] = 0.5

        for t in range(self.T):
            i, j = self.select_arms(t)
            result = self.compare_fn(i, j)

            self.wins[i][j] += result
            self.wins[j][i] += 1 - result
            self.plays[i][j] += 1
            self.plays[j][i] += 1

            # Assigning the regret
            self.regret += self.regret_fn(i, j)
            self.regrets.append(self.regret)

        # Returning the cumulative regret.
        return self.get_best_arm(), self.regrets