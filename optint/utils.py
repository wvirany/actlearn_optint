import numpy as np
from causallearn.search.ScoreBased.GES import ges
import graphical_models as gm


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
    
    return wrong_dag
    

def learn_dag(data, p):
    Record = ges(data.T, score_func='local_score_BIC')
    G = gm.DAG(set(range(p)))

    learned_graph = Record['G']
    for i in range(p):
        for j in range(p):
            if learned_graph.graph[i, j] == -1 and learned_graph.graph[j, i] == 1:
                # Directed edge i -> j
                G.add_arc(i, j)
            elif learned_graph.graph[i, j] == -1 and learned_graph.graph[j, i] == -1:
                # Undirected edge - pick direction by node index
                if i < j:
                    G.add_arc(i, j)
    return G


def compute_shd(G1, G2):
    edges1 = set(G1.arcs)
    edges2 = set(G2.arcs)
    return len(edges1.symmetric_difference(edges2))


def create_intervention_matrix(batches):
    """
    Convert list of batches to format expected by BiDAG iBGe.
    
    Args:
        batches: list of (p, n_i) arrays, one per experimental condition
        
    Returns:
        data: (total_samples, p) array - all data concatenated
        Imat: (total_samples, n_conditions) binary array - intervention indicators
    """
    import numpy as np
    
    n_conditions = len(batches)
    total_samples = sum(batch.shape[1] for batch in batches)
    p = batches[0].shape[0]
    
    # Pre-allocate
    data = np.zeros((total_samples, p))
    Imat = np.zeros((total_samples, n_conditions))
    
    # Fill in data and intervention indicators
    sample_idx = 0
    for cond_idx, batch in enumerate(batches):
        n_samples = batch.shape[1]
        
        # Transpose batch from (p, n) to (n, p)
        data[sample_idx:sample_idx + n_samples, :] = batch.T
        
        # Mark which condition these samples came from
        Imat[sample_idx:sample_idx + n_samples, cond_idx] = 1
        
        sample_idx += n_samples
    
    return data, Imat


def compute_ibge_score(data, Imat, dag_adjacency, am=0.1):
    """
    Compute iBGe score for a given DAG using BiDAG via rpy2.
    
    Args:
        data: (n_samples, p) numpy array
        Imat: (n_samples, n_conditions) binary numpy array  
        dag_adjacency: (p, p) binary adjacency matrix (numpy array)
        am: iBGe hyperparameter (default 0.1)
    
    Returns:
        score: log iBGe score (higher is better)
    """
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    
    # Define R function (only once per session)
    if not hasattr(compute_ibge_score, '_r_function_defined'):
        ro.r('''
            compute_score_ibge <- function(data, Imat, dag, am) {
                library(BiDAG)
                
                # Create scoring object with iBGe
                scoreObj <- scoreparameters(
                    scoretype = "usr",
                    data = data,
                    usrpar = list(pctesttype = "bge", Imat = Imat, am = am)
                )
                
                # Compute score for this DAG
                score <- DAGscore(scoreObj, dag)
                
                return(score)
            }
        ''')
        compute_ibge_score._r_function_defined = True
    
    # Convert numpy arrays to R objects within conversion context
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_data = ro.conversion.py2rpy(data)
        r_Imat = ro.conversion.py2rpy(Imat)
        r_dag = ro.conversion.py2rpy(dag_adjacency.astype(int))
    
    # Call the R function
    compute_fn = ro.r['compute_score_ibge']
    score = compute_fn(r_data, r_Imat, r_dag, am)[0]
    
    return score