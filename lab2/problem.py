import numpy as np

LOW = -6.0
HIGH = 6.0

# minima of the function without the constraint, f = 0 in all of them
MINIMA = np.array([
    [3.0, 2.0],
    [-2.805118, 3.131312],
    [-3.779310, -3.283186],
    [3.584428, -1.848126],
])


def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def violation(x, y):
    # constraint g(x, y) = x + y >= 0
    return np.maximum(0.0, -(x + y))


def penalty_coef(kind, lam, beta, t, t_max):
    if kind == "quad":
        return lam
    return lam * (t / t_max) ** beta


def nearest_minimum(point):
    dist = np.linalg.norm(MINIMA - point, axis=1)
    k = int(dist.argmin())
    return k, dist[k]
