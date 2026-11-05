import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider

from problem import HIGH, LOW, MINIMA, himmelblau

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
GRAY = "#8a8984"
TRAIL = 6


class Dashboard:
    """Live window: particles on the heat map, convergence, share outside."""

    def __init__(self, params, delay=0.05):
        self.params = params
        self.delay = delay
        plt.ion()
        self.fig = plt.figure(figsize=(14, 7.5))
        self.fig.canvas.manager.set_window_title("PSO Himmelblau")
        grid = self.fig.add_gridspec(2, 2, width_ratios=[1.1, 1],
                                     left=0.05, right=0.95, bottom=0.14,
                                     top=0.9, wspace=0.2, hspace=0.35)
        self.ax_map = self.fig.add_subplot(grid[:, 0])
        self.ax_conv = self.fig.add_subplot(grid[0, 1])
        self.ax_out = self.fig.add_subplot(grid[1, 1])
        self.draw_map_background()
        self.history = []

    def draw_map_background(self):
        ax = self.ax_map
        xs = np.linspace(LOW, HIGH, 400)
        x, y = np.meshgrid(xs, xs)
        heat = ax.contourf(x, y, np.log10(1 + himmelblau(x, y)), levels=30,
                           cmap="viridis")
        self.fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.02,
                          label="lg(1 + f)")

        # infeasible half-plane x + y < 0
        ax.fill([LOW, HIGH, LOW], [HIGH, LOW, LOW], hatch="//",
                facecolor="none", edgecolor="white", alpha=0.35)
        ax.plot([LOW, HIGH], [HIGH, LOW], color="white", linewidth=1.5)
        ax.text(-5.7, -5.5, "x + y < 0", color="white", fontsize=11)

        feasible = MINIMA.sum(axis=1) >= 0
        ax.scatter(*MINIMA[feasible].T, marker="*", s=180, color="white",
                   edgecolor="black", zorder=3, label="мінімуми f = 0")
        ax.scatter(*MINIMA[~feasible].T, marker="X", s=110, color=GRAY,
                   edgecolor="black", zorder=3,
                   label="мінімум поза допустимою областю")
        for k, (mx, my) in enumerate(MINIMA, start=1):
            ax.annotate(f"M{k}", (mx, my), xytext=(8, 8),
                        textcoords="offset points", color="white",
                        fontweight="bold")

        self.trails = LineCollection([], colors="white", linewidths=0.8,
                                     alpha=0.6)
        ax.add_collection(self.trails)
        self.dots = ax.scatter([], [], s=22, color=ORANGE, edgecolor="black",
                               linewidth=0.5, zorder=4, label="частинки")
        self.best_dot = ax.scatter([], [], s=140, marker="o",
                                   facecolor="none", edgecolor="red",
                                   linewidth=2, zorder=5, label="gbest")
        ax.set_xlim(LOW, HIGH)
        ax.set_ylim(LOW, HIGH)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    def __call__(self, pso, t):
        self.history = pso.history
        self.draw(t)
        plt.pause(self.delay)

    def draw(self, t):
        h = self.history[t]
        start = max(0, t - TRAIL)
        path = np.stack([s["pos"] for s in self.history[start:t + 1]], axis=1)
        self.trails.set_segments(list(path))
        self.dots.set_offsets(h["pos"])
        self.best_dot.set_offsets([h["gbest"]])
        self.ax_map.set_title(
            f"Ітерація {t}: gbest = ({h['gbest'][0]:.4f}, "
            f"{h['gbest'][1]:.4f}), f = {h['f']:.2e}")
        self.draw_convergence(t)
        self.draw_outside(t)

        p = self.params
        if p.penalty == "quad":
            penalty = f"квадратичний штраф λ = {p.lam:g}"
        else:
            penalty = f"адаптивний штраф λ0 = {p.lam:g}, β = {p.beta:g}"
        self.fig.suptitle(
            f"gbest-PSO · {penalty} · частинок {p.particles} · "
            f"ω = {p.w} c1 = {p.c1} c2 = {p.c2} · "
            f"v_max = {p.vmax}·(ub − lb) · seed {p.seed}",
            fontweight="bold")
        self.fig.canvas.draw_idle()

    def draw_convergence(self, t):
        ax = self.ax_conv
        ax.clear()
        hist = self.history[:t + 1]
        its = [h["t"] for h in hist]
        # log scale, exact zeros are drawn at 1e-16
        f = np.maximum([h["f"] for h in hist], 1e-16)
        big_f = np.maximum([h["F"] for h in hist], 1e-16)
        ax.plot(its, big_f, color=ORANGE, linewidth=2.5,
                label="F(x) = f(x) + P(x)")
        ax.plot(its, f, color=BLUE, linewidth=1.5, label="f(x) без штрафу")
        ax.set_yscale("log")
        ax.set_xlim(0, self.params.iterations)
        ax.set_ylim(1e-14, 1e3)
        ax.set_xlabel("ітерація")
        ax.set_title("Збіжність gbest (best-so-far)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def draw_outside(self, t):
        ax = self.ax_out
        ax.clear()
        hist = self.history[:t + 1]
        its = [h["t"] for h in hist]
        ax.plot(its, [100 * h["outside"] for h in hist], color=GREEN,
                label="частинок з x + y < 0, %")
        ax.plot(its, [h["lam"] for h in hist], color=GRAY, linestyle="--",
                label="коефіцієнт штрафу λ(t)")
        ax.set_xlim(0, self.params.iterations)
        ax.set_ylim(0, 105)
        ax.set_xlabel("ітерація")
        ax.set_title("Вихід у допустиму область")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def keep_open(self):
        # after the run any iteration can be shown again with the slider
        # or with the arrow keys
        last = len(self.history) - 1
        place = self.fig.add_axes([0.2, 0.03, 0.6, 0.03])
        self.slider = Slider(place, "ітерація", 0, last, valinit=last,
                             valstep=1)
        self.slider.on_changed(lambda value: self.draw(int(value)))

        def on_key(event):
            step = {"right": 1, "left": -1, "up": 10, "down": -10}
            if event.key == "home":
                self.slider.set_val(0)
            elif event.key in step:
                value = self.slider.val + step[event.key]
                self.slider.set_val(min(max(value, 0), last))

        # arrows and Home are matplotlib navigation keys by default
        for name in ["keymap.back", "keymap.forward", "keymap.home"]:
            plt.rcParams[name] = []
        self.fig.canvas.mpl_connect("key_press_event", on_key)
        plt.ioff()
        plt.show()
