import numpy as np
import graphical_models as gm
from .data.dag import *


def create_misspecified_dag(true_dag, n_changes=2):
    """Create a misspecified DAG by randomly adding/removing edges."""
    p = true_dag.nnodes
    wrong_dag = gm.DAG(set(range(p)))

    true_arcs = list(true_dag.arcs)

    # Remove some edges
    if len(true_arcs) >= n_changes:
        keep_arcs = true_arcs[:-n_changes]
        wrong_dag.add_arcs_from(keep_arcs)
    else:
        wrong_dag.add_arcs_from(true_arcs)
    
    # Add some wrong edges
    added = 0
    max_attempts = 50
    attempts = 0
    while added < n_changes and attempts < max_attempts:
        i, j = np.random.choice(p, 2, replace=False)
        if i < j and (i, j) not in true_dag.arcs:
            wrong_dag.add_arc(i, j)
            added += 1
        attempts += 1
    
    return 
    

def learn_dag(data, p):
    Record = ges(data.T, score_func='local_score_BIC')
    G = gm.DAG(set(range(p)))

    learned_graph = Record['G']
    for i in range(p):
        for j in range(p):
            if learned_graph.graph[i, j] == -1:
                G.add_arc(i, j)
    return G


def select_intervention(model, problem, n_per_step, measure='uniform'):