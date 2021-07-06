from sklearn.base import BaseEstimator

from ya_glm.autoassign import autoassign
from fclsp.thresh.gen_thresh import gen_thresh


class Thresh(BaseEstimator):

    @autoassign
    def __init__(self, pen_val=1, thresh='hard', thresh_kws={}):
        pass

    def fit(self, X, y=None):
        values, pre_pro_out = self._get_object_from_data(X)
        fit_out = self._apply_shrinkage(values)
        self._set_fit(fit_out, pre_pro_out)
        return self

    def _apply_shrinkage(self, values):
        shrunk_values = gen_thresh(values=values,
                                   pen_val=self.pen_val,
                                   thresh=self.thresh,
                                   thresh_kws=self.thresh_kws)

        return {'values': shrunk_values}

    def _get_object_from_data(self, X):
        raise NotImplementedError

    def _set_fit(self, fit_out, pre_pro_out):
        raise NotImplementedError
