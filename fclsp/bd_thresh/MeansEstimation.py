from copy import deepcopy

from fclsp.thresh.MeansEstimation import MeansMixin, ThreshCV
from fclsp.FrobFclsp import FrobFclspFitLLA, FrobFclspFitLLACV


class FclspLLA(FrobFclspFitLLA, MeansMixin):

    def _get_defualt_init(self):
        return ThreshCV()

    def _get_init_data_from_fit_est(self, est):
        if hasattr(est, 'means_'):
            m = est.means_

        elif hasattr(est, 'best_estimator_') and \
                hasattr(est.best_estimator_, 'means_'):
            m = est.best_estimator_.means_

        return {'values': deepcopy(m)}


class FclspLLACV(FrobFclspFitLLACV):

    def _get_base_class(self):
        return FclspLLA
