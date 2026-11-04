import sys
import time
from dataclasses import replace

import numpy as np

from mmas import MMAS, Params
from rcpsp import Project

SEEDS = range(1, 6)
EXPERIMENTS = [
    ("alpha", [0.0, 1.0, 3.0]),
    ("beta", [0.0, 1.0, 3.0]),
    ("rho", [0.02, 0.1, 0.5]),
    ("ants", [5, 20, 40]),
    ("heuristic", ["spt", "crit", "res", "combined"]),
]


def run_series(project, params):
    spans = []
    found = []
    t0 = time.perf_counter()
    for seed in SEEDS:
        aco = MMAS(project, replace(params, seed=seed))
        aco.run()
        spans.append(aco.best_makespan)
        found.append(aco.best_iteration)
    seconds = (time.perf_counter() - t0) / len(SEEDS)
    return spans, found, seconds


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "instances/j309_1.json"
    project = Project(path)
    base = Params()
    print(f"Задача {project.name}, {len(SEEDS)} запусків на кожне значення, "
          f"решта параметрів: m={base.ants}, α={base.alpha}, β={base.beta}, "
          f"ρ={base.rho}, η={base.heuristic}")
    print(f"{'параметр':>10} {'значення':>9} {'кращий':>7} {'середній':>9} "
          f"{'гірший':>7} {'ітерація':>9} {'час, с':>7}")

    # the base configuration appears in every group, run it once
    done = {}
    for name, values in EXPERIMENTS:
        for value in values:
            params = replace(base, **{name: value})
            if params not in done:
                done[params] = run_series(project, params)
            spans, found, seconds = done[params]
            print(f"{name:>10} {value:>9} {min(spans):>7} "
                  f"{np.mean(spans):>9.1f} {max(spans):>7} "
                  f"{np.mean(found):>9.0f} {seconds:>7.1f}", flush=True)
        print()


if __name__ == "__main__":
    main()
