import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
GRAY = "#8a8984"


class Dashboard:
    """Live window: best schedule, convergence, resource load and pheromone."""

    def __init__(self, every=5):
        self.every = every
        plt.ion()
        self.fig, axes = plt.subplots(2, 2, figsize=(14, 8.5),
                                      gridspec_kw={"width_ratios": [1.4, 1]})
        self.fig.canvas.manager.set_window_title("MMAS RCPSP")
        self.ax_gantt, self.ax_conv = axes[0]
        self.ax_load, self.ax_tau = axes[1]
        self.colorbar = None

    def __call__(self, aco, it, last):
        if it % self.every != 0 and it != 1 and not last:
            return
        self.draw_gantt(aco)
        self.draw_convergence(aco)
        self.draw_load(aco)
        self.draw_pheromone(aco)

        p = aco.params
        title = (f"MMAS · {aco.project.name} · ітерація {it} · "
                 f"мурах {p.ants} · α={p.alpha} β={p.beta} ρ={p.rho} · "
                 f"η={p.heuristic}")
        self.fig.suptitle(title, fontweight="bold")
        self.fig.tight_layout()
        plt.pause(0.01)

    def keep_open(self):
        plt.ioff()
        plt.show()

    def draw_gantt(self, aco):
        project = aco.project
        start = aco.best_start
        ax = self.ax_gantt
        ax.clear()
        for i in range(project.n):
            ax.barh(i, project.durations[i], left=start[i],
                    color=COLORS[0], edgecolor="white")
            ax.text(start[i] + project.durations[i] / 2, i, project.ids[i],
                    ha="center", va="center", color="white", fontsize=7)

        labels = [f"{a} {name}" for a, name in zip(project.ids, project.names)]
        ax.set_yticks(range(project.n), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, aco.best_makespan)
        ax.set_xlabel("час")
        ax.set_title(f"Найкращий розклад: makespan = {aco.best_makespan}")

    def draw_convergence(self, aco):
        ax = self.ax_conv
        ax.clear()
        x = range(1, len(aco.best_curve) + 1)
        ax.plot(x, aco.iter_mean, color=COLORS[2], label="середній в ітерації")
        ax.plot(x, aco.iter_best, color=COLORS[1],
                label="найкращий в ітерації")
        ax.plot(x, aco.best_curve, color=COLORS[0], linewidth=2,
                label="найкращий за весь час")
        ax.axhline(aco.lower_bound, color=GRAY, linestyle=":",
                   label=f"нижня межа = {aco.lower_bound}")
        for r in aco.restarts:
            ax.axvline(r, color=GRAY, linewidth=0.7)

        ax.set_xlim(1, aco.params.iterations)
        ax.set_xlabel("ітерація")
        ax.set_ylabel("makespan")
        ax.legend(fontsize=8)
        ax.set_title("Збіжність (вертикальні лінії: скидання феромону)")

    def draw_load(self, aco):
        project = aco.project
        start = aco.best_start
        finish = start + project.durations
        ax = self.ax_load
        ax.clear()
        times = np.arange(aco.best_makespan)
        for k, name in enumerate(project.resources):
            used = []
            for t in times:
                running = (start <= t) & (t < finish)
                used.append(project.demand[running, k].sum())
            percent = 100 * np.array(used) / project.capacity[k]
            color = COLORS[k % len(COLORS)]
            ax.step(times, percent, where="post", color=color, label=name)

        ax.axhline(100, color=GRAY, linestyle="--")
        ax.set_ylim(0, 110)
        ax.set_xlim(0, aco.best_makespan)
        ax.set_xlabel("час")
        ax.set_ylabel("% від наявного")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Завантаження ресурсів")

    def draw_pheromone(self, aco):
        # rows and columns follow the best order,
        # so a learned order shows up as a dark upper triangle
        order = aco.best_order
        ids = [aco.project.ids[i] for i in order]
        ax = self.ax_tau
        ax.clear()
        im = ax.imshow(aco.tau[np.ix_(order, order)], cmap="Blues",
                       vmin=0, vmax=aco.tau_max)
        ax.set_xticks(range(len(ids)), ids, fontsize=6, rotation=90)
        ax.set_yticks(range(len(ids)), ids, fontsize=6)
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.set_title("Феромон τ[i][j]: «i раніше за j»")
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(im, ax=ax)
        else:
            self.colorbar.update_normal(im)
