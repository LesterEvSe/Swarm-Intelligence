import argparse
import time

from mmas import MMAS, Params
from rcpsp import Project, check_schedule


def print_schedule(project, start):
    print(f"{'акт.':>5}  {'назва':<30} {'p':>3} {'старт':>6} {'кінець':>7}")
    for i in sorted(range(project.n), key=lambda i: start[i]):
        finish = start[i] + project.durations[i]
        print(f"{project.ids[i]:>5}  {project.names[i]:<30} "
              f"{project.durations[i]:>3} {start[i]:>6} {finish:>7}")


def main():
    default = Params()
    parser = argparse.ArgumentParser(description="RCPSP, Max-Min Ant System")
    parser.add_argument("instance")
    parser.add_argument("--ants", type=int, default=default.ants)
    parser.add_argument("--iters", type=int, default=default.iterations)
    parser.add_argument("--alpha", type=float, default=default.alpha)
    parser.add_argument("--beta", type=float, default=default.beta)
    parser.add_argument("--rho", type=float, default=default.rho)
    parser.add_argument("--heuristic", default=default.heuristic,
                        choices=["spt", "res", "crit", "combined"])
    parser.add_argument("--seed", type=int, default=default.seed)
    parser.add_argument("--live", action="store_true",
                        help="show the algorithm while it runs")
    args = parser.parse_args()

    try:
        project = Project(args.instance)
    except ValueError as e:
        print("Некоректні вхідні дані:", e)
        return

    resources = zip(project.resources, project.capacity)
    print(f"Задача {project.name}, активностей: {project.n}, ресурси: "
          + ", ".join(f"{r}={c}" for r, c in resources))
    if project.description:
        print(project.description)
    print(f"Критичний шлях = {project.tail.max()}, "
          f"нижня межа makespan = {project.lower_bound()}")

    params = Params(args.ants, args.iters, args.alpha, args.beta, args.rho,
                    args.heuristic, seed=args.seed)
    print(params)

    aco = MMAS(project, params)
    dashboard = None
    if args.live:
        from dashboard import Dashboard
        dashboard = Dashboard()

    t0 = time.perf_counter()
    start, makespan = aco.run(dashboard)
    seconds = time.perf_counter() - t0

    print(f"\nmakespan = {makespan}, "
          f"знайдено на ітерації {aco.best_iteration}, "
          f"час {seconds:.1f} с, скидань феромону: {len(aco.restarts)}")
    if makespan == aco.lower_bound:
        print("makespan дорівнює нижній межі, розклад оптимальний")
    errors = check_schedule(project, start)
    if errors:
        print("Перевірка розкладу:", errors)
    else:
        print("Перевірка розкладу: усі обмеження виконано")
    print()
    print_schedule(project, start)

    if dashboard:
        dashboard.keep_open()


if __name__ == "__main__":
    main()
