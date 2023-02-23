import matplotlib.pyplot as plt
import numpy as np


#  --------------------------------  --------------------------------  --------------------------------
# graph distance
#  --------------------------------  --------------------------------  --------------------------------

def _estimate_alpha_xlog2x():
    p = 4
    x1 = np.arange(1e-10, 1.01, 0.01)
    y = x1 * np.log2(x1)
    alpha = np.polyfit(x1, y, p)
    return alpha[::-1]


def _estimate_xlog2x(x):
    if x == 0:
        return 0
    alpha = _estimate_alpha_xlog2x()
    f = np.poly1d(alpha[::-1])
    return f(x)


def von_neumann_entropy(G):
    """
    ref: De Domenico, Manlio et al, Structural reducibility of multilayer networks
    :param G: graph adjacency matrix
    :return:
    """
    n_nodes = G.shape[0]
    d = np.sum(G, axis=0, keepdims=True)
    D = np.eye(n_nodes) * d
    L_hat = (D - G) / np.sum(d)

    alpha = _estimate_alpha_xlog2x()

    L_n = np.eye(n_nodes)
    val = 0
    for n in range(1, 5):
        L_n = L_n @ L_hat
        val += alpha[n] * np.trace(L_n)
    vn_entropy = - alpha[0] * n_nodes - val
    return vn_entropy


def jensen_shannon_dist(G1, G2):
    js_dist = von_neumann_entropy((G1+G2)/2) - 1 / 2 * (von_neumann_entropy(G1) + von_neumann_entropy(G2))
    return js_dist


#  --------------------------------  --------------------------------  --------------------------------
# graph matrices
#  --------------------------------  --------------------------------  --------------------------------


def compute_laplacian(G):
    from scipy.linalg import sqrtm
    d = np.sum(G, axis=1)
    D = np.diag(d)
    L = np.linalg.multi_dot([sqrtm(np.linalg.inv(D)), D - G, sqrtm(np.linalg.inv(D))])
    return L


#  --------------------------------  --------------------------------  --------------------------------
# centrality
#  --------------------------------  --------------------------------  --------------------------------
def pagerank_t(G, alpha=0.85, personalization=None,
             max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
             dangling=None):
    """
    (c) from geeksforgeeks website

    Return the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key for every graph node and nonzero personalization value for each node.
      By default, a uniform distribution is used.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
      Starting value of PageRank iteration for each node.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified). This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The PageRank algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs by converting each edge in the
    directed graph to two edges.


    """
    import networkx as nx
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:

        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError('Personalization dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:

        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:

            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise NetworkXError('pagerank: power iteration failed to converge '
                        'in %d iterations.' % max_iter)


def authority_hub(mat1):
    from scipy.linalg import eig
    l1, eigvec1 = eig(mat1.T @ mat1)
    authority = np.real(eigvec1)[:, 0]

    _, eigvec2 = eig(mat1 @ mat1.T)
    hub = np.real(eigvec2)[:, 0]

    return np.abs(hub), np.abs(authority)
