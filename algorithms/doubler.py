import numpy as np
import math
import json
import matplotlib.pyplot as plt


class sbm():

    def __init__(self, n_arms):

        self.alpha = 1
        self.k = n_arms

    def reset(self):

        self.averages = np.ones(self.k) * math.inf
        self.pulls = np.ones(self.k)
        self.time = 1

    def advance(self):

        bound = self.averages + np.sqrt((self.alpha + 2) * np.log(self.time) / (2 * self.pulls))
        choice = np.argmax(bound)
        return choice

    def feedback(self, arm, reward):

        self.time += 1

        if self.averages[arm] == math.inf:
            self.averages[arm] = reward
        else:
            self.averages[arm] = (self.averages[arm] * self.pulls[arm] + reward) / (self.pulls[arm] + 1)

        self.pulls[arm] += 1


class Doubler:

    def __init__(self, T, pref, regret_func):

        self.pref_matrix = np.array(pref)
        n_arms = len(pref[0])

        self.sbm = sbm(n_arms)

        self.l = np.random.randint(n_arms, size=1)
        self.i, self.t = 1, 1

        self.T = T
        self.regret_func = regret_func

    def run(self):

        regret = [0]

        while self.t < self.T:

            self.sbm.reset()
            new_l = set()

            for j in range(2 ** self.i):
                xt = np.random.choice(self.l)
                yt = self.sbm.advance()
                new_l.add(yt)
                bt = np.random.binomial(1, self.pref_matrix[xt][yt], 1)

                if bt == 1:
                    self.sbm.feedback(yt, 0)
                    self.sbm.feedback(xt, 1)
                else:
                    self.sbm.feedback(yt, 1)
                    self.sbm.feedback(xt, 0)

                regret.append(regret[-1] + self.regret_func(xt, yt))

                self.t += 1

                if self.t >= self.T:
                    break

            self.l = np.array(list(new_l))
            self.i += 1

        return  np.argmax(self.sbm.averages), list(np.around(regret, 3))