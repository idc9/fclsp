import numpy as np
from scipy.sparse import diags
from sklearn.metrics import pairwise_distances

from fclsp.reshaping_utils import vec_hollow_sym


def get_lap_coef(V, w, var_type, shape):
    """
    Computes the Laplacian coefficent vector

    TODO: finish documenting

    Parameters
    ----------
    V: array-like

    w: array-like

    var_type: str
        Type of the variable. Must be one of ['hollow_sym', 'rect', 'multi'].

    shape: tuple of ints
        Shape of the variable.

    Output
    ------
    lap_coef:

    """
    assert var_type in ['hollow_sym', 'rect', 'multi']

    if var_type == 'hollow_sym':
        return get_lap_coef_hollow_sym(V=V, w=w)

    elif var_type == 'rect':
        return get_lap_coef_rect(V=V, w=w, shape=shape)

    elif var_type == 'multi':
        return get_lap_coef_multi(V=V, w=w, shape=shape)


def get_lap_coef_hollow_sym(V, w):
    """
    Returns the Laplacian coefficent for an adjaceny matrix.

    Let A(x) in R^{d x d} be an adjacency matrix parametatized by its edges x in R^{d choose 2}. Also let V in R^{d times K} and w in R^K for K <= d.

    The laplacian coefficient M(V, w) in R^{d choose 2} is the vector such that

     M(V, w)^T x = Tr(V^T Laplacian(A(x)) V diag(w))

    Parameters
    ----------
    V: array-like, (n_nodes, K)
        The input matrix.

    w: array-like, (K, )
        The input vector.

    Output
    -------
    M(V, w): array-like, (n_nodes choose 2, )
        The Laplacian coefficient vector.
    """
    assert V.shape[1] == len(w)

    coef = pairwise_distances(V @ diags(np.sqrt(w)), metric='euclidean',
                              n_jobs=None)  # TODO: give option

    coef = vec_hollow_sym(coef) ** 2

    return coef


def get_lap_coef_rect(V, w, shape):
    """
    Returns the Laplacian coefficent for a rectuangular matrix.


    Parameters
    ----------
    V: array-like, (n_nodes, K)
        The input matrix.

    w: array-like, (K, )
        The input vector.

    shape: tuple of two ints
        Size of the rectangular matrix matrix.

    Output
    -------
    M(V, w): array-like, (sum(shape), )
        The Laplacian coefficient vector.
    """
    raise NotImplementedError


def get_lap_coef_multi(V, w, shape):
    """
    Returns the Laplacian coefficent for a multi-array.

    Parameters
    ----------
    V: array-like, (n_nodes, K)
        The input matrix.

    w: array-like, (K, )
        The input vector.

    shape: tuple of two ints
        Size of the rectangular matrix matrix.

    Output
    -------
    M(V, w): array-like
        The Laplacian coefficient vector.
    """
    raise NotImplementedError
