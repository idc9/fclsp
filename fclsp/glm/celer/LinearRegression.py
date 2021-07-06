from ya_glm.models.linear_regression import LinRegMixin

from fclsp.glm.GlmFclsp import GlmFcslpFitLLA, GlmFclspFitLLACV
from fclsp.lla import solve_lla

from ya_glm.backends.celer.fcp_lla_solver import WL1SolverGlm
from ya_glm.backends.celer.LinearRegression import LassoCV


class FclspLLA(LinRegMixin, GlmFcslpFitLLA):
    solve_lla = staticmethod(solve_lla)
    base_wl1_solver = WL1SolverGlm

    def _get_defualt_init(self):
        return LassoCV(fit_intercept=self.fit_intercept,
                       opt_kws=self.opt_kws)


class FclspLLACV(GlmFclspFitLLACV):

    def _get_base_class(self):
        return FclspLLA
