# Lab 2. Particle Swarm Optimization, Himmelblau with a constraint

Minimize f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2 for x, y in [-6, 6]
with the constraint g(x, y) = x + y >= 0 using gbest-PSO.
The constraint is handled by a penalty F = f + P, quadratic (constant lambda)
or adaptive (lambda grows as lambda0 * (t / T)^beta).

## Run

```sh
python -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -r requirements.txt

python main.py                                   # adaptive penalty, seed 1
python main.py --penalty quad --lam 100
python main.py --init infeasible --live          # window with the swarm at work
python experiments.py                            # 20 runs for each penalty setting
```

In the `--live` window, after the run, the slider or the arrow keys
(left/right: 1 iteration, up/down: 10, Home: start) show any iteration again.

## Files

- `problem.py` - Himmelblau function, constraint violation, penalty coefficient, known minima
- `pso.py` - PSO parameters and the algorithm
- `dashboard.py` - live view: heat map with particles, convergence of f and F, share of particles outside
- `main.py` - command line run
- `experiments.py` - quadratic vs adaptive penalty, 20 seeds each, median convergence plot
