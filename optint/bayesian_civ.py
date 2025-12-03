from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import graphical_models as gm

from optint.acquisition import civ_acq
from optint.data import synthetic_instance
from optint.model import linearSCM
from optint.test import test_active, test_bayesian_active
from optint.utils import create_misspecified_dag, learn_dag


# Experiment parameters
p = 10
n_trials = 3
n_obs = 100

class Opts:
    pass

opts = Opts()
opts.T = 50
opts.W = 3
opts.n = 10
opts.R = 1
opts.known_noise = True
opts.measure = 'unif'
opts.acq = 'civ'
opts.active = True


def get_distances(A, a_star):
    return [np.linalg.norm(a - a_star) for a in A]


all_dist_oracle = []
all_dist_misspec = []
all_dist_bayesian = []
all_shd_bayesian = []


for trial in range(n_trials):
    print(f"Trial {trial+1}/{n_trials}")

    seed = trial * 100

    np.random.seed(seed)

    problem = synthetic_instance(
        nnodes=p,
        DAG_type="random",
        sigma_square=0.4 * np.ones(p),
        a_size=p,
        a_target_nodes=None,
        std=False,
        seed=seed,
    )
    
    # Rescale target
    a_star_true = np.linalg.solve(np.eye(p) - problem.B, problem.mu_target)
    scale_factor = np.linalg.norm(a_star_true)
    problem.mu_target = problem.mu_target / scale_factor

    # Generate initial observational data
    X_obs = problem.sample(a = np.zeros((p, 1)), n=100)

    # Create a misspecified DAG
    G_wrong = None
    while G_wrong is None:
        try:
            G_wrong = create_misspecified_dag(problem.DAG, n_changes=8)
        except:
            continue

    # Learn initial DAG
    G_learned = learn_dag(X_obs, problem.nnodes)

    # Run oracle
    A_oracle, _ = test_active(problem, opts)

    # Run misspecified
    true_dag = problem.DAG
    problem.DAG = G_wrong
    A_misspec, _ = test_active(problem, opts)
    problem.DAG = true_dag

    # Run Bayesian
    A_bayesian, _, SHD_bayesian = test_bayesian_active(
        problem, 
        G_learned, 
        opts,
        use_ibge=True,
        K=1,
        am=0.1
    )

    # Compute distances
    all_dist_oracle.append(get_distances(A_oracle, problem.a_target))
    all_dist_misspec.append(get_distances(A_misspec, problem.a_target))
    all_dist_bayesian.append(get_distances(A_bayesian, problem.a_target))
    all_shd_bayesian.append(SHD_bayesian)

all_dist_oracle = np.array(all_dist_oracle)
all_dist_misspec = np.array(all_dist_misspec)
all_dist_bayesian = np.array(all_dist_bayesian)
all_shd_bayesian = np.array(all_shd_bayesian)

results_dir = Path('results')
results_dir.mkdir(parents=True, exist_ok=True)

np.save(results_dir / 'bayesian_civ_distances.npy', all_dist_oracle)
np.save(results_dir / 'bayesian_civ_misspec_distances.npy', all_dist_misspec)
np.save(results_dir / 'bayesian_civ_bayesian_distances.npy', all_dist_bayesian)
np.save(results_dir / 'bayesian_civ_shd.npy', all_shd_bayesian)