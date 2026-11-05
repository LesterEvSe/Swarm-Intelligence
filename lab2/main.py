import argparse
import time

from problem import MINIMA, nearest_minimum
from pso import PSO, Params

SHOW = [0, 1, 2, 5, 10, 20, 30, 50, 100]


def main():
    default = Params()
    parser = argparse.ArgumentParser(
        description="PSO, Himmelblau function with x + y >= 0")
    parser.add_argument("--particles", type=int, default=default.particles)
    parser.add_argument("--iters", type=int, default=default.iterations)
    parser.add_argument("--w", type=float, default=default.w)
    parser.add_argument("--c1", type=float, default=default.c1)
    parser.add_argument("--c2", type=float, default=default.c2)
    parser.add_argument("--vmax", type=float, default=default.vmax)
    parser.add_argument("--penalty", default=default.penalty,
                        choices=["quad", "adapt"])
    parser.add_argument("--lam", type=float, default=default.lam)
    parser.add_argument("--beta", type=float, default=default.beta)
    parser.add_argument("--init", default=default.init,
                        choices=["all", "infeasible"])
    parser.add_argument("--seed", type=int, default=default.seed)
    parser.add_argument("--live", action="store_true",
                        help="show the swarm while it runs")
    args = parser.parse_args()

    params = Params(args.particles, args.iters, args.w, args.c1, args.c2,
                    args.vmax, args.penalty, args.lam, args.beta, args.init,
                    args.seed)
    print("Гіммельблау: f = (x^2 + y - 11)^2 + (x + y^2 - 7)^2, "
          "x, y in [-6, 6], обмеження x + y >= 0")
    print(params)

    pso = PSO(params)
    dashboard = None
    if args.live:
        from dashboard import Dashboard
        dashboard = Dashboard(params)

    t0 = time.perf_counter()
    best = pso.run(dashboard)
    seconds = time.perf_counter() - t0

    print(f"\n{'ітер.':>5} {'λ':>8} {'x':>9} {'y':>9} {'f':>10} "
          f"{'F':>10} {'поза, %':>8}")
    for h in pso.history:
        if h["t"] not in SHOW and h["t"] != params.iterations:
            continue
        x, y = h["gbest"]
        print(f"{h['t']:>5} {h['lam']:>8.3f} {x:>9.5f} {y:>9.5f} "
              f"{h['f']:>10.3e} {h['F']:>10.3e} {100 * h['outside']:>8.0f}")

    x, y = best["gbest"]
    k, dist = nearest_minimum(best["gbest"])
    mx, my = MINIMA[k]
    print(f"\nРозв'язок: x = {x:.6f}, y = {y:.6f}")
    print(f"f(x, y) = {best['f']:.3e}, g(x, y) = x + y = {x + y:.6f}, "
          f"порушення v = {best['v']:.3f}")
    print(f"Найближчий мінімум M{k + 1} = ({mx:.6f}, {my:.6f}), "
          f"відстань {dist:.2e}")

    hit = pso.hit_iteration()
    if hit is None:
        print("Допустимої точки з f < 1e-4 не знайдено")
    else:
        print(f"f < 1e-4 у допустимій точці з ітерації {hit}, "
              f"оцінок f: {params.particles * (hit + 1)}")
    print(f"Усього оцінок f: {pso.evaluations}, час {1000 * seconds:.1f} мс")

    if dashboard:
        dashboard.keep_open()


if __name__ == "__main__":
    main()
