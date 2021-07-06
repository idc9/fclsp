from sklearn.covariance import empirical_covariance
import numpy as np

from fclsp.reshaping_utils import vec_hollow_sym, fill_hollow_sym
from fclsp.thresh.base import Thresh as _Thresh
from fclsp.thresh.cv import ThreshCV as _ThreshCV


class CovarianceMixin:

    def _get_object_from_data(self, X):

        means = X.mean(axis=0)

        # TODO: perhaps validate data
        empir_cov = empirical_covariance(X, assume_centered=False)

        variances = np.diag(empir_cov)
        utri = vec_hollow_sym(empir_cov)

        return utri, {'means': means, 'variances': variances}

    def _set_fit(self, fit_out, pre_pro_out):

        utri = fit_out['values']
        cov = fill_hollow_sym(utri)
        np.fill_diagonal(cov, pre_pro_out['variances'])

        self.covariance_ = cov
        self.location_ = pre_pro_out['means']

        if 'opt_data' in fit_out:
            self.opt_data_ = fit_out['opt_data']

    def score(self, X, y=None):

        test_cov = empirical_covariance(X - self.location_,
                                        assume_centered=True)

        resid = test_cov - self.covariance_

        rmse = np.sqrt((resid ** 2).sum()) / np.sqrt(resid.size)

        return -rmse


class Thresh(CovarianceMixin, _Thresh):
    pass


class ThreshCV(CovarianceMixin, _ThreshCV):
    def _get_base_class(self):
        return Thresh
