from dataclasses import dataclass

import numpy as np

from problem import HIGH, LOW, himmelblau, penalty_coef, violation


@dataclass(frozen=True)
class Params:
    particles: int = 30
    iterations: int = 100
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    vmax: float = 0.2
    penalty: str = "adapt"
    lam: float = 100.0
    beta: float = 2.0
    init: str = "all"
    seed: int = 1


class PSO:
    def __init__(self, params):
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self.vmax = params.vmax * (HIGH - LOW)
        n = params.particles

        self.pos = self.start_positions(n)
        self.vel = self.rng.uniform(-self.vmax, self.vmax, (n, 2)) * 0.1
        self.f, self.v = self.evaluate(self.pos)
        self.evaluations = n

        self.pbest = self.pos.copy()
        self.pbest_f = self.f.copy()
        self.pbest_v = self.v.copy()

        self.history = []
        self.record(0)

    def start_positions(self, n):
        pos = self.rng.uniform(LOW, HIGH, (n, 2))
        if self.params.init != "infeasible":
            return pos
        # keep only points with x + y < 0, draw again the rest
        while True:
            bad = pos.sum(axis=1) >= 0
            if not bad.any():
                return pos
            pos[bad] = self.rng.uniform(LOW, HIGH, (bad.sum(), 2))

    def evaluate(self, pos):
        x = pos[:, 0]
        y = pos[:, 1]
        return himmelblau(x, y), violation(x, y)

    def lam_at(self, t):
        p = self.params
        return penalty_coef(p.penalty, p.lam, p.beta, t, p.iterations)

    def gbest(self, lam):
        # pbest is compared with the current lambda, it grows in adaptive mode
        fitness = self.pbest_f + lam * self.pbest_v ** 2
        return int(fitness.argmin())

    def record(self, t):
        lam = self.lam_at(t)
        g = self.gbest(lam)
        f = self.pbest_f[g]
        v = self.pbest_v[g]
        self.history.append({
            "t": t,
            "lam": lam,
            "pos": self.pos.copy(),
            "gbest": self.pbest[g].copy(),
            "f": f,
            "v": v,
            "F": f + lam * v ** 2,
            "outside": np.mean(self.v > 0),
        })

    def step(self, t):
        p = self.params
        lam = self.lam_at(t)
        g = self.pbest[self.gbest(lam)]

        r1 = self.rng.random(self.pos.shape)
        r2 = self.rng.random(self.pos.shape)
        self.vel = (p.w * self.vel
                    + p.c1 * r1 * (self.pbest - self.pos)
                    + p.c2 * r2 * (g - self.pos))
        np.clip(self.vel, -self.vmax, self.vmax, out=self.vel)
        self.pos += self.vel

        # a particle that left the box stops on its border
        out = (self.pos < LOW) | (self.pos > HIGH)
        np.clip(self.pos, LOW, HIGH, out=self.pos)
        self.vel[out] = 0

        self.f, self.v = self.evaluate(self.pos)
        self.evaluations += p.particles

        new_fit = self.f + lam * self.v ** 2
        old_fit = self.pbest_f + lam * self.pbest_v ** 2
        better = new_fit < old_fit
        self.pbest[better] = self.pos[better]
        self.pbest_f[better] = self.f[better]
        self.pbest_v[better] = self.v[better]

    def run(self, on_iteration=None):
        for t in range(1, self.params.iterations + 1):
            self.step(t)
            self.record(t)
            if on_iteration:
                on_iteration(self, t)
        return self.history[-1]

    def hit_iteration(self, eps=1e-4):
        # first iteration where gbest is feasible and f < eps
        for h in self.history:
            if h["v"] == 0 and h["f"] < eps:
                return h["t"]
        return None
