import numpy as np


def get_adj_mat(values, var_type, shape):
    """

    Parameters
    ----------
    value: array-like, (n_variables, )
        The current value of the variable.

    var_type: str
        Type of the variable. Must be one of ['hollow_sym', 'rect', 'multi'].

    shape: tuple of ints
        Shape of the variable.

    Output
    ------
    A: array-like
    """
    if var_type == 'hollow_sym':
        return fill_hollow_sym(values=values)

    elif var_type == 'rect':
        return fill_bipt_matrix(values=values, shape=shape)

    elif var_type == 'multi':
        return fill_multi_array_adj(values=values, shape=shape)


def fill_hollow_sym(values):
    """
    Fills a d x d hollow symmetric matrix.

    Parameters
    ----------
    values: array-like, (d choose 2, 1)
        The upper triangluar values of the array.
        These should have been extracted from the matrix using
        vec_hollow_sym(A)

    Output
    ------
    A: array-like, (d , d)
        The matrix.
    """
    values = np.array(values).reshape(-1)
    d = get_d(len(values))

    A = np.zeros((d, d))
    A[np.triu_indices(d, k=1)] = values
    A = A + A.T
    # A[np.tril_indices(d, k=-1)] = values
    return A


def vec_hollow_sym(A):
    """
    Vectorizes the upper triangular elements of a hollow symmetric matrix.

    Parameters
    ----------
    A: array-like, (d , d)
        The matrix.

    Output
    ------
    utri: array-like, (d choose 2, 1)
    """
    return A[np.triu_indices(A.shape[0], k=1)]


def fill_bipt_matrix(values, shape):
    """
    Fills the bipartite adjacency matrix whose edge weights are given by a rectuangular matrix.

    Parameters
    ----------
    values: array-like, (n_rows * n_cols, )

    shape: tuple of ints
        The shape of the rectuangular matrix, (n_rows, n_cols)

    Output
    ------
    A: array-like, (n_rows + n_cols, n_rows + n_cols)
        The bipartite matrix.
    """
    assert len(shape) == 2
    values = np.array(values).reshape(-1)
    values = values.reshape(shape)

    R, C = shape
    return np.block([[np.zeros((R, R)), values],
                     [values.T, np.zeros((C, C))]])


def fill_multi_array_adj(values, shape):
    raise NotImplementedError


def get_d(D):
    """
    Solves for d given D where
    D = d choose 2 = 0.5 * d * (d - 1)

    Parameters
    ----------
    D: int

    Output
    ------
    d: int
    """
    d = (1 + np.sqrt(1 + 8 * D)) / 2
    if d != int(d):
        raise ValueError("Cannot solve for {} = d choose 2".format(D))
    return int(d)
