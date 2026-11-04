import math
from dataclasses import dataclass

import numpy as np

from rcpsp import serial_sgs


@dataclass(frozen=True)
class Params:
    ants: int = 20
    iterations: int = 100
    alpha: float = 1.0
    beta: float = 1.0
    rho: float = 0.1
    heuristic: str = "combined"
    stagnation: int = 30
    seed: int = 1


def heuristic(project, kind):
    short = 1 / np.maximum(project.durations, 1)
    critical = project.tail.astype(float)
    load = (project.demand / project.capacity).sum(axis=1)
    thrift = 1 / np.maximum(load, 0.05)

    if kind == "spt":
        return short
    if kind == "crit":
        return critical
    if kind == "res":
        return thrift
    return short * critical * thrift


class MMAS:
    def __init__(self, project, params):
        self.project = project
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self.eta = heuristic(project, params.heuristic)
        self.lower_bound = project.lower_bound()

        self.best_order = None
        self.best_start = None
        self.best_makespan = math.inf
        self.best_iteration = 0
        self.restarts = []
        self.best_curve = []
        self.iter_best = []
        self.iter_mean = []

        # tau[i, j]: how much the colony wants activity i to start before j.
        # The first ant runs on equal trails to estimate tau_max.
        self.tau = np.ones((project.n, project.n))
        _, makespan = serial_sgs(project, self.build_order())
        self.set_bounds(makespan)
        self.tau[:] = self.tau_max

    def set_bounds(self, makespan):
        self.tau_max = 1 / (self.params.rho * makespan)
        self.tau_min = 0.05 * self.tau_max

    def choose(self, eligible):
        if len(eligible) == 1:
            return eligible[0]

        idx = np.array(eligible)
        tau = self.tau[np.ix_(idx, idx)]
        score = tau.sum(axis=1) - tau.diagonal()
        alpha = self.params.alpha
        beta = self.params.beta
        weights = score ** alpha * self.eta[idx] ** beta
        probs = weights / weights.sum()
        return int(self.rng.choice(idx, p=probs))

    def build_order(self):
        waiting = [len(p) for p in self.project.preds]
        eligible = [i for i in range(self.project.n) if waiting[i] == 0]
        order = []
        while eligible:
            j = self.choose(eligible)
            order.append(j)
            eligible.remove(j)
            for s in self.project.succs[j]:
                waiting[s] -= 1
                if waiting[s] == 0:
                    eligible.append(s)
        return order

    def deposit(self, order, makespan):
        pos = np.empty(self.project.n, dtype=int)
        pos[order] = np.arange(self.project.n)
        # every pair (i, j) where i is earlier in the order gets 1 / makespan
        earlier = pos[:, None] < pos[None, :]
        self.tau[earlier] += 1 / makespan
        np.clip(self.tau, self.tau_min, self.tau_max, out=self.tau)

    def run(self, on_iteration=None):
        since_best = 0
        for it in range(1, self.params.iterations + 1):
            ants = []
            for _ in range(self.params.ants):
                order = self.build_order()
                start, makespan = serial_sgs(self.project, order)
                ants.append((makespan, order, start))
            makespan, order, start = min(ants, key=lambda ant: ant[0])

            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_order = order
                self.best_start = start
                self.best_iteration = it
                self.set_bounds(makespan)
                since_best = 0
            else:
                since_best += 1

            # only the iteration-best ant deposits
            self.tau *= 1 - self.params.rho
            self.deposit(order, makespan)

            if since_best >= self.params.stagnation:
                self.tau[:] = self.tau_max
                self.restarts.append(it)
                since_best = 0

            self.best_curve.append(self.best_makespan)
            self.iter_best.append(makespan)
            self.iter_mean.append(np.mean([ant[0] for ant in ants]))

            optimal = self.best_makespan == self.lower_bound
            last = optimal or it == self.params.iterations
            if on_iteration:
                on_iteration(self, it, last)
            if optimal:
                break

        return self.best_start, self.best_makespan
