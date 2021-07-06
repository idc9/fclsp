from ya_glm.fcp.GlmFcp import GlmFcpFitLLA
from ya_glm.fcp.GlmFcpCV import GlmFcpFitLLACV

from ya_glm.autoassign import autoassign
from ya_glm.glm_pen_max_lasso import lasso_max

from fclsp.max_pen_val_fclsp import get_max_pen_val_fclsp


class GlmFcslpFitLLA(GlmFcpFitLLA):
    """
    solve_lla
    """

    # @autoassign
    def __init__(self,
                 init='default',
                 pen_val=1,
                 pen_func='scad',
                 pen_func_kws={},
                 **kws):
        super().__init__(**kws)

        # TODO: HACK because of signature issue
        # fix this once I figure out a better solution
        self.init = init
        self.pen_val = pen_val
        self.pen_func = pen_func
        self.pen_func_kws = pen_func_kws

    def get_pen_val_max(self, X, y, init_data):

        lasso_max_val = lasso_max(X=X, y=y,
                                  fit_intercept=self.fit_intercept,
                                  model_type=self._model_type)

        # TODO: add these
        var_type = 'hollow_sym'
        var_shape = None

        return get_max_pen_val_fclsp(lasso_max_val=lasso_max_val,
                                     init=init_data['coef'],
                                     pen_func=self.pen_func,
                                     pen_func_kws=self.pen_func_kws,
                                     var_type=var_type,
                                     shape=var_shape)


class GlmFclspFitLLACV(GlmFcpFitLLACV):

    @autoassign
    def __init__(self,
                 pen_func='scad',
                 pen_func_kws={},
                 **kws):
        super().__init__(**kws)
