import numpy as np
from scipy.linalg import eigh
from sklearn.utils.extmath import svd_flip


def eigh_wrapper(A, rank=None, eval_descending=True):
    """
    Symmetrics eigenvector problem.

    Parameters
    ----------
    A: array-like, shape (n x n)

    rank: None, int
        Maximum number of components to compute.

    eval_descending: bool
        Whether or not to compute largest or smallest eigenvalues.
        If True, will compute largest rank eigenvalues and
        eigenvalues are returned in descending order. Otherwise,
        computes smallest eigenvalues and returns them in ascending order.

    Output
    ------
    evals, evecs
    """

    if rank is not None:
        n_max_evals = A.shape[0]

        if eval_descending:
            eigvals_idxs = (n_max_evals - rank, n_max_evals - 1)
        else:
            eigvals_idxs = (0, rank - 1)
    else:
        eigvals_idxs = None

    evals, evecs = eigh(a=A, eigvals=eigvals_idxs)

    if eval_descending:
        ev_reordering = np.argsort(-evals)
        evals = evals[ev_reordering]
        evecs = evecs[:, ev_reordering]

    evecs = svd_flip(evecs, evecs.T)[0]

    return evals, evecs
