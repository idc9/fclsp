import numpy as np
from sklearn.base import BaseEstimator

from fclsp.thresh.base import Thresh as _Thresh
from fclsp.thresh.cv import ThreshCV as _ThreshCV


class MeansMixin:

    def _get_object_from_data(self, X):

        empir_means = X.mean(axis=0)

        return empir_means, None

    def _set_fit(self, fit_out, pre_pro_out):
        self.means_ = fit_out['values']

        if 'opt_data' in fit_out:
            self.opt_data_ = fit_out['opt_data']

    def score(self, X, y=None):

        test_means = X.mean(axis=0)
        resid = test_means - self.means_

        rmse = np.sqrt((resid ** 2).sum()) / np.sqrt(resid.size)
        return -rmse


class Empirical(MeansMixin, BaseEstimator):
    def fit(self, X, y=None):
        values, pre_pro_out = self._get_object_from_data(X)
        fit_out = {'values': values}  # this is just the empirical means
        self._set_fit(fit_out, pre_pro_out)
        return self


class Thresh(MeansMixin, _Thresh):
    pass


class ThreshCV(MeansMixin, _ThreshCV):
    def _get_base_class(self):
        return Thresh
