from fclsp.reshaping_utils import get_adj_mat
from fclsp.lla import eigh_lap_abs


def get_max_pen_val_fclsp(lasso_max_val, init,
                          pen_func, pen_func_kws={},
                          var_type='hollow_sym',
                          shape=None):
    """
    Returns the largest reasonable tuning parameter value for fitting a
    FCLS penalized GLM with the LLA algorithm. Larger penalty values
    will result in the LLA algorithm converging to zero.

    Parameters
    ----------
    lasso_max_val:

    init:

    pen_func:

    pen_func_kws:

    var_type:

    shape:
    """

    A = get_adj_mat(values=init,
                    var_type=var_type, shape=shape)
    evals, evecs = eigh_lap_abs(A, rank=None)
    max_eval = max(evals)

    if pen_func == 'scad':
        # # TODO: allow this to depend on the penalty function
        # # pushing the largest init elemnt under the pen param
        # # forces all LLA weights to be equal to the pen param
        # init_max = abs(init_data['coef']).max()

        return max(max_eval, lasso_max_val)

    else:
        # TODO: add this
        raise NotImplementedError
