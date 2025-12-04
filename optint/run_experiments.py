import argparse
from pathlib import Path

import graphical_models as gm
import numpy as np

from optint.data import synthetic_instance
from optint.utils import create_misspecified_dag, learn_dag
from optint.test import test_active, test_passive, test_bayesian_active


def get_distances(A, a_star):
    return [np.linalg.norm(a - a_star) for a in A]


def run_experiment_1(p, trial_id):
    
    # Define experiment options
    class Opts:
        pass

    opts = Opts()
    opts.T = 50
    opts.W = 3
    opts.n = 10
    opts.known_noise = True
    opts.measure = 'unif'
    opts.acq = 'civ'
    opts.active = True

    seed = trial_id * 100
    np.random.seed(seed)

    # Generate problem instance
    problem = synthetic_instance(
        nnodes=p,
        DAG_type='random',
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
            expected_edges = 0.1 * p * (p - 1)
            target_shd = int(0.6 * expected_edges)
            G_wrong = create_misspecified_dag(problem.DAG, n_changes=target_shd)
        except:
            continue

    # Learn initial DAG
    G_learned = learn_dag(X_obs, problem.nnodes)

    print(f"Trial {trial_id}, p={p}, Running oracle...")
    A_oracle, _ = test_active(problem, opts)

    print(f"Trial {trial_id}, p={p}: Running Misspecified...")
    true_dag = problem.DAG
    problem.DAG = G_wrong
    A_misspec, _ = test_active(problem, opts)
    problem.DAG = true_dag

    print(f"Trial {trial_id}, p={p}: Running Bayesian CIV (K=1)...")
    A_bayesian, _, SHD_bayesian = test_bayesian_active(
        problem, 
        G_learned, 
        opts,
        use_ibge=True,
        K=1,
        am=0.1
    )

    print(f"Trial {trial_id}, p={p}: Running Random...")
    A_random, _ = test_passive(problem, opts)

    # Compute distances
    dist_oracle = get_distances(A_oracle, problem.a_target)
    dist_misspec = get_distances(A_misspec, problem.a_target)
    dist_bayesian = get_distances(A_bayesian, problem.a_target)
    dist_random = get_distances(A_random, problem.a_target)

    # Save results
    results_dir = Path('results') / 'exp1' / f'p{p}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(results_dir / f'trial{trial_id}_oracle.npy', dist_oracle)
    np.save(results_dir / f'trial{trial_id}_misspec.npy', dist_misspec)
    np.save(results_dir / f'trial{trial_id}_bayesian.npy', dist_bayesian)
    np.save(results_dir / f'trial{trial_id}_random.npy', dist_random)
    np.save(results_dir / f'trial{trial_id}_shd.npy', SHD_bayesian)
    
    print(f"Trial {trial_id}, p={p}: Complete!")




def run_experiment_2(p, K, trial_id):
    
    # Define experiment options
    class Opts:
        pass
    
    opts = Opts()
    opts.T = 50
    opts.W = 3
    opts.n = 10
    opts.known_noise = True
    opts.measure = 'unif'
    opts.acq = 'civ'
    opts.active = True
    
    seed = trial_id * 100
    np.random.seed(seed)
    
    # Generate problem instance
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
    X_obs = problem.sample(a=np.zeros((p, 1)), n=100)
    G_learned = learn_dag(X_obs, problem.nnodes)
    
    print(f"Trial {trial_id}, p={p}, K={K}: Running CIV-UG...")
    import time
    start = time.time()
    
    A_civug, _, SHD_civug = test_bayesian_active(
        problem, 
        G_learned, 
        opts,
        use_ibge=True,
        K=K,
        am=0.1
    )
    
    runtime = time.time() - start
    
    # Compute distances
    dist_civug = get_distances(A_civug, problem.a_target)
    
    # Save results
    results_dir = Path('results') / 'exp2' / f'p{p}' / f'K{K}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(results_dir / f'trial{trial_id}_dist.npy', dist_civug)
    np.save(results_dir / f'trial{trial_id}_shd.npy', SHD_civug)
    np.save(results_dir / f'trial{trial_id}_runtime.npy', runtime)
    
    print(f"Trial {trial_id}, p={p}, K={K}: Complete! Runtime: {runtime:.2f}s")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=int, required=True, choices=[1, 2])
    parser.add_argument('--p', type=int, required=True)
    parser.add_argument('--trial', type=int, required=True)
    parser.add_argument('--K', type=int, default=1)
    args = parser.parse_args()

    if args.experiment == 1:
        run_experiment_1(args.p, args.trial)
    else:
        run_experiment_2(args.p, args.K, args.trial)
