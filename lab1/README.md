# Lab 1. RCPSP + Max-Min Ant System

Resource-constrained project scheduling solved with Max-Min Ant System.
An ant builds an order of activities (only those whose predecessors are already taken),
the serial schedule generation scheme turns the order into a schedule, the makespan is minimized.
Pheromone `tau[i][j]` means "activity i starts before j".

## Run

```sh
python -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -r requirements.txt

python main.py instances/release.json
python main.py instances/j309_1.json --live          # window with the algorithm at work
python main.py instances/j309_1.json --alpha 1 --beta 1 --rho 0.1 --ants 20 --iters 100
python experiments.py                                 # parameter experiments on j309_1
```

## Files

- `rcpsp.py` - reading a project, checks (cycles, resources), lower bound, serial SGS
- `mmas.py` - heuristics and the MMAS algorithm
- `dashboard.py` - live view: Gantt chart, convergence, resource load, pheromone matrix
- `main.py` - command line run
- `experiments.py` - runs with different alpha, beta, rho, number of ants and heuristics
- `instances/` - test projects in JSON (`j309_1` is from PSPLIB, optimum 83)
