from curses import COLOR_RED
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def draw(dag, edge_weights=None, title=None, savefile=None):
    """ 
    plot a partially directed graph
    """
    plt.clf()

    p = dag.nnodes

    plt.gcf().set_size_inches(4, 4)

    # directed edges
    d = nx.DiGraph()
    d.add_nodes_from(list(range(p)))
    for (i, j) in dag.arcs:
        d.add_edge(i, j)
    
    pos = nx.circular_layout(d)

    nx.draw_networkx_nodes(
        d,
        pos,
        node_color='lightblue',
        node_size=600,
        linewidths=3,
    )

    if edge_weights is not None:
        edge_colors = [abs(edge_weights[i, j]) for i, j in d.edges()]
        nx.draw_networkx_edges(
            d,
            pos,
            edge_color=edge_colors,
            edge_cmap=plt.cm.Blues,
            width=1.5,
            arrowsize=10,
            edge_vmin=0.9,
            edge_vmax=1,
            min_source_margin=20,
            min_target_margin=20,
        )
    else:
        nx.draw_networkx_edges(
            d,
            pos,
            edge_color='midnightblue',
            width=1.5,
            arrowsize=10,
            min_source_margin=20,
            min_target_margin=20,
        )

    nx.draw_networkx_labels(d, pos,
        labels={node: node for node in range(p)},
        font_size=14,
    )

    if title:
        plt.title(title, fontsize=16, fontweight='bold')

    plt.axis('off')
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
    plt.close()


def draw_spectrum(A, B, savefile=None):
    plt.clf()
    plt.figure(figsize=(3,3))
    e_A = np.linalg.eigvalsh(np.matmul(A.T, A))[::-1]
    e_B = np.linalg.eigvalsh(np.matmul(B.T, B))[::-1]
    plt.plot(np.maximum(e_A,0)**0.5, label=r'$(I-B)^{-1}$')
    plt.plot(np.maximum(e_B,0)**0.5, label=r'$B$')
    plt.legend()
    plt.ylabel('eigenvalues')
    plt.xlabel('index')
    plt.title('Spectrum of SCM')
    plt.tight_layout()
    
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
    plt.close()