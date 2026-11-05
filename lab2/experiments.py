import time
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from problem import nearest_minimum
from pso import PSO, Params

SEEDS = range(1, 21)
EXPERIMENTS = [
    ("quad100", dict(penalty="quad", lam=100.0)),
    ("adapt", dict(penalty="adapt", lam=100.0, beta=2.0)),
    ("quad0.01", dict(penalty="quad", lam=0.01)),
    ("quad100-inf", dict(penalty="quad", lam=100.0, init="infeasible")),
    ("adapt-inf", dict(penalty="adapt", lam=100.0, beta=2.0,
                       init="infeasible")),
]
COLORS = {"quad100": "#2a78d6", "adapt": "#eb6834", "quad0.01": "#1baf7a",
          "quad100-inf": "#2a78d6", "adapt-inf": "#eb6834"}


def run_series(params):
    finals = []
    hits = []
    minima = [0, 0, 0, 0]
    curves_f = []
    curves_big_f = []
    seconds = []
    for seed in SEEDS:
        pso = PSO(replace(params, seed=seed))
        t0 = time.perf_counter()
        best = pso.run()
        seconds.append(time.perf_counter() - t0)

        finals.append(best)
        hits.append(pso.hit_iteration())
        minima[nearest_minimum(best["gbest"])[0]] += 1
        curves_f.append([h["f"] for h in pso.history])
        curves_big_f.append([h["F"] for h in pso.history])
    return finals, hits, minima, curves_f, curves_big_f, np.mean(seconds)


def main():
    base = Params()
    print(f"{len(SEEDS)} прогонів (seed 1-{len(SEEDS)}), частинок "
          f"{base.particles}, ітерацій {base.iterations}, ω={base.w}, "
          f"c1={base.c1}, c2={base.c2}, v_max={base.vmax}·(ub-lb), gbest")
    print(f"{'налаштування':<12} {'best f':>9} {'mean f ± std':>21} "
          f"{'v>0':>4} {'f<1e-4':>6} {'ітер.':>5} {'оцінок':>6} "
          f"{'час, мс':>7}  M1 M2 M3 M4")

    curves = {}
    for name, change in EXPERIMENTS:
        finals, hits, minima, cf, cbf, sec = run_series(replace(base,
                                                                **change))
        f = np.array([b["f"] for b in finals])
        outside = sum(b["v"] > 0 for b in finals)
        found = [h for h in hits if h is not None]
        it = np.mean(found)
        evals = base.particles * (it + 1)
        print(f"{name:<12} {f.min():>9.2e} {f.mean():>10.2e} ± "
              f"{f.std():<8.2e} {outside:>4} {len(found):>6} {it:>5.1f} "
              f"{evals:>6.0f} {1000 * sec:>7.1f}  "
              + " ".join(f"{m:>2}" for m in minima), flush=True)
        curves[name] = (np.median(cf, axis=0), np.median(cbf, axis=0))

    # median best-so-far over the runs, F dashed
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    fig.canvas.manager.set_window_title("PSO experiments")
    groups = [("Старт у всій області", ["quad100", "adapt", "quad0.01"]),
              ("Старт лише в x + y < 0", ["quad100-inf", "adapt-inf"])]
    for ax, (title, names) in zip(axes, groups):
        for name in names:
            color = COLORS[name]
            f, big_f = curves[name]
            ax.plot(np.maximum(big_f, 1e-16), color=color, linewidth=2,
                    linestyle="--", label=f"{name}: F")
            ax.plot(np.maximum(f, 1e-16), color=color, label=f"{name}: f")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("ітерація")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("медіана gbest за 20 прогонами")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
