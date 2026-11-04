import json
import math

import numpy as np


class Project:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.name = data["name"]
        self.description = data.get("description", "")
        self.resources = list(data["resources"])
        self.capacity = np.array(
            [data["resources"][r] for r in self.resources])

        acts = data["activities"]
        self.ids = list(acts)
        self.n = len(self.ids)
        index = {a: i for i, a in enumerate(self.ids)}

        self.names = []
        self.durations = np.zeros(self.n, dtype=int)
        self.demand = np.zeros((self.n, len(self.resources)), dtype=int)
        self.preds = []
        for i, a in enumerate(self.ids):
            act = acts[a]
            self.names.append(act.get("name", ""))
            self.durations[i] = act["p"]
            usage = act.get("usage", {})
            for k, r in enumerate(self.resources):
                self.demand[i, k] = usage.get(r, 0)

            preds = []
            for q in act.get("pred", []):
                if str(q) not in index:
                    raise ValueError(
                        f"активність {a}: невідомий попередник {q}")
                preds.append(index[str(q)])
            self.preds.append(preds)

        self.succs = [[] for _ in range(self.n)]
        for j in range(self.n):
            for i in self.preds[j]:
                self.succs[i].append(j)

        for i in range(self.n):
            if np.any(self.demand[i] > self.capacity):
                raise ValueError(f"активності {self.ids[i]} бракує ресурсу, "
                                 "навіть якщо вона виконується сама")

        # tail[i]: longest path from the start of i to the project end
        self.tail = np.zeros(self.n, dtype=int)
        for i in reversed(self.topological_order()):
            longest = max((self.tail[j] for j in self.succs[i]), default=0)
            self.tail[i] = self.durations[i] + longest

    def topological_order(self):
        # Kahn's algorithm: activities that never become free are on a cycle
        waiting = [len(p) for p in self.preds]
        queue = [i for i in range(self.n) if waiting[i] == 0]
        order = []
        while queue:
            i = queue.pop(0)
            order.append(i)
            for j in self.succs[i]:
                waiting[j] -= 1
                if waiting[j] == 0:
                    queue.append(j)

        if len(order) < self.n:
            stuck = [self.ids[i] for i in range(self.n) if waiting[i] > 0]
            names = ", ".join(stuck)
            raise ValueError("цикл у залежностях між активностями " + names)
        return order

    def lower_bound(self):
        bound = self.tail.max()
        for k in range(len(self.resources)):
            work = (self.durations * self.demand[:, k]).sum()
            bound = max(bound, math.ceil(work / self.capacity[k]))
        return int(bound)


def serial_sgs(project, order):
    """Puts every activity of the order at the earliest feasible time."""
    horizon = project.durations.sum()
    used = np.zeros((horizon + 1, len(project.resources)), dtype=int)
    start = np.zeros(project.n, dtype=int)

    for j in order:
        p = project.durations[j]
        ready = [start[i] + project.durations[i] for i in project.preds[j]]
        t = max(ready, default=0)
        while True:
            need = used[t:t + p] + project.demand[j]
            over = np.any(need > project.capacity, axis=1)
            if not over.any():
                break
            # jump right after the first moment with not enough resources
            t += np.argmax(over) + 1

        used[t:t + p] += project.demand[j]
        start[j] = t

    makespan = (start + project.durations).max()
    return start, int(makespan)


def check_schedule(project, start):
    errors = []
    finish = start + project.durations

    for j in range(project.n):
        for i in project.preds[j]:
            if start[j] < finish[i]:
                errors.append(f"{project.ids[j]} почалась раніше, "
                              f"ніж завершилась {project.ids[i]}")

    for t in range(finish.max()):
        running = (start <= t) & (t < finish)
        used = project.demand[running].sum(axis=0)
        if np.any(used > project.capacity):
            errors.append(f"у момент {t} перевищено ресурси")
    return errors
