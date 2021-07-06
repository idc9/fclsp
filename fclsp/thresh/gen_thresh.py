import numpy as np

from ya_glm.opt.concave_penalty import scad_prox, mcp_prox


def gen_thresh(values, pen_val, thresh='hard', thresh_kws={}):
    """
    Applies a generalized thresholding operator to an array.

    Parameters
    ----------
    values: array-like
        The values to threshold.

    pen_val: float
        Amount of shrinkage.

    kind: str
        Which kind of thresholding operator to use. Must be one of ['soft', 'hard']

    kws: dict
        Optional key word arguments to pass into thresholding operator.

    Output
    ------
    values_thresholded: array-like
    """

    if thresh is None:
        return values

    if thresh == 'hard':
        out = hard_thresh(values=values, pen_val=pen_val)

    elif thresh == 'soft':
        out = soft_thresh(values=values, pen_val=pen_val)

    elif thresh == 'adpt_lasso':
        out = adpt_lasso_thresh(values=values, pen_val=pen_val,
                                **thresh_kws)
    elif thresh == 'scad':
        out = scad_prox(values=values, pen_val=pen_val,
                        **thresh_kws)

    elif thresh == 'mcp':
        out = mcp_prox(values=values, pen_val=pen_val,
                       **thresh_kws)

    elif callable(thresh):
        return thresh(values=values, pen_val=pen_val, **thresh_kws)

    else:
        raise ValueError("Bad input to 'thresh' {}".format(thresh))

    return out


def hard_thresh(values, pen_val):
    """
    Hard thresholding opeator.

    Parameters
    ----------
    values: array-like
        The values to threshold.

    pen_val: float
        The cutoff value.

    Output
    ------
    pen_values: array-like
        The values after hard threshodling.
    """
    values = np.array(values)
    out = np.zeros_like(values)
    non_zero_mask = abs(values) > pen_val
    out[non_zero_mask] = values[non_zero_mask]
    return out


def soft_thresh(values, pen_val):
    """
    Soft thresholding opeator.

    Parameters
    ----------
    values: array-like
        The values to threshold.

    pen_val: float
        The cutoff value.

    Output
    ------
    pen_values: array-like
        The values after soft threshodling.
    """
    values = np.array(values)
    return np.sign(values) * np.clip(abs(values) - pen_val,
                                     a_min=0, a_max=np.inf)


def adpt_lasso_thresh(values, pen_val, eta=1):
    """
    Adaptive Lasso based generalized thresholding operator e.g. see (4) from (Rothman et al, 2009).

    Parameters
    ----------
    values: array-like
        The values to threshold.

    pen_val: float
        The cutoff value.

    eta: float
        TODO

    Output
    ------
    pen_values: array-like
        The values after threshodling.
    """
    values = np.array(values)

    pen_vals_trans = (1.0 / abs(values)) ** eta
    pen_vals_trans *= pen_val ** (eta + 1)

    return np.sign(values) * np.clip(abs(values) - pen_vals_trans,
                                     a_min=0, a_max=np.inf)
