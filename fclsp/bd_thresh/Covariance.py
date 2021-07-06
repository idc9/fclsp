from copy import deepcopy

from fclsp.thresh.Covariance import CovarianceMixin, ThreshCV
from fclsp.FrobFclsp import FrobFclspFitLLA, FrobFclspFitLLACV
from fclsp.reshaping_utils import vec_hollow_sym


class FclspLLA(FrobFclspFitLLA, CovarianceMixin):

    def _get_defualt_init(self):
        return ThreshCV()

    def _get_init_data_from_fit_est(self, est):
        if hasattr(est, 'covariance_'):
            cov = est.covariance_

        elif hasattr(est, 'best_estimator_') and \
                hasattr(est.best_estimator_, 'covariance_'):
            cov = est.best_estimator_.covariance_

        utri = deepcopy(vec_hollow_sym(cov))
        return {'values': utri}


class FclspLLACV(FrobFclspFitLLACV):

    def _get_base_class(self):
        return FclspLLA
