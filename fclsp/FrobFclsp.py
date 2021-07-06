import numpy as np
from time import time
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, FLOAT_DTYPES

from ya_glm.lla.WeightedLassoSolver import WeightedLassoSolver
from ya_glm.fcp.GlmFcp import InitMixin
from ya_glm.utils import get_sequence_decr_max
from ya_glm.autoassign import autoassign
from ya_glm.opt.concave_penalty import get_penalty_func

from ya_glm.cv.CVGridSearch import CVGridSearchMixin
from ya_glm.cv.cv_select import CVSlectMixin

from fclsp.lla import solve_lla
from fclsp.max_pen_val_fclsp import get_max_pen_val_fclsp


class FrobFclspFitLLA(InitMixin, BaseEstimator):

    solve_lla = staticmethod(solve_lla)
    base_wl1_solver = None

    @autoassign
    def __init__(self,
                 pen_val=1,
                 pen_func='scad',
                 pen_func_kws={},
                 init='default',
                 lla_n_steps=2,
                 lla_kws={}):
        pass

    def fit(self, X, y=None):

        # validate the data!
        X = self._validate_data(X)

        # get data for initialization
        init_data = self.get_init_data(X)
        if 'est' in init_data:
            self.init_est_ = init_data['est']

        values, pre_pro_out = self._get_object_from_data(X)
        fit_out = self.run_fclsp_lla(values, init_data)
        self._set_fit(fit_out, pre_pro_out)

        return self

    def _validate_data(self, X, accept_sparse=False):
        """
        Validates the X/y data. This should not change the raw input data, but may reformat the data (e.g. convert pandas to numpy).


        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        """

        # TODO: waht to do about sparse???
        X = check_array(X, accept_sparse=accept_sparse,
                        dtype=FLOAT_DTYPES)

        return X

    def run_fclsp_lla(self, values, init_data):

        values_init = init_data['values']

        # TODO: give option for other variable types
        var_type = 'hollow_sym'

        penalty_func = get_penalty_func(pen_func=self.pen_func,
                                        pen_val=self.pen_val,
                                        pen_func_kws=self.pen_func_kws)

        values_shrunk, _, opt_data = \
            solve_lla(wl1_solver=FrobWL1Solver(x0=values),
                      penalty_fcn=penalty_func,
                      init=values_init,
                      init_upv=None,
                      var_type=var_type,
                      n_steps=self.lla_n_steps,
                      **self.lla_kws)

        # # compute coefficient using LLA algorithm
        # values_shrunk, _, opt_data = \
        #     self.solve_lla(wl1_solver=wl1_solver,
        #                    init=values_init)

        return {'values': values_shrunk, 'opt_data': opt_data}

    def _get_defualt_init(self):
        raise NotImplementedError

    def _get_init_data_from_fit_est(self, est):
        raise NotImplementedError

    def get_pen_val_max(self, X, init_data):
        values, pre_pro_data = self._get_object_from_data(X)

        lasso_max_val = abs(values).max()

        var_type = 'hollow_sym'  # TODO: give option to customize

        return get_max_pen_val_fclsp(lasso_max_val=lasso_max_val,
                                     init=init_data['values'],
                                     pen_func=self.pen_func,
                                     pen_func_kws=self.pen_func_kws,
                                     var_type=var_type,
                                     shape=None)


class FrobWL1Solver(WeightedLassoSolver):
    """
    min_x 0.5 * ||x0 - x||_2^2 + ||x||_{w, 1}
    """
    def __init__(self, x0):
        self.x0 = np.array(x0)

    def solve(self, L1_weights, opt_init=None, opt_init_upv=None):

        L1_weights = np.array(L1_weights).reshape(self.x0.shape)

        # solution obtained from soft thresholding
        soln = np.sign(self.x0) * np.clip(abs(self.x0) - L1_weights,
                                          a_min=0, a_max=np.inf)

        return soln, None, None

    def loss(self, value, upv=None):
        resid = (self.x0 - value).reshape(-1)
        return 0.5 * sum(r ** 2 for r in resid)


class FrobFclspFitLLACV(CVSlectMixin, CVGridSearchMixin,
                        InitMixin, BaseEstimator):

    @autoassign
    def __init__(self,
                 pen_val=1,
                 pen_func='scad',
                 pen_func_kws={},
                 init='default',
                 lla_n_steps=2,
                 lla_kws={},

                 n_pen_vals=100,
                 pen_vals=None,
                 pen_min_mult=1e-3,
                 pen_spacing='log',

                 cv=None,
                 cv_select_rule='best',
                 cv_select_metric=None,
                 cv_scorer=None,
                 cv_verbose=0, cv_n_jobs=None,
                 cv_pre_dispatch='2*n_jobs'):
        pass

    def fit(self, X, y=None):

        # validate the data!
        est = self._get_base_estimator()
        X = est._validate_data(X)

        # get initialization from raw data
        init_data = est.get_init_data(X)
        if 'est' in init_data:
            self.init_est_ = init_data['est']

        # set the initializer
        est.set_params(init=deepcopy(init_data))

        # set up the tuning parameter values sequence
        self._set_tuning_values(X=X, init_data=init_data)

        # run cross-validation on the raw data
        self.cv_data_ = {}
        start_time = time()
        self.cv_results_ = self._run_cv(X, cv=self.cv, estimator=est)
        self.cv_data_['cv_runtime'] = time() - start_time

        # select best tuning parameter values
        self.best_tune_idx_, self.best_tune_params_ = \
            self._select_tune_param(self.cv_results_)
        est.set_params(**self.best_tune_params_)

        # refit on the raw data
        start_time = time()
        est.fit(X)
        self.cv_data_['refit_runtime'] = time() - start_time
        self.best_estimator_ = est

        return self

    def _get_base_estimator(self):
        c = self._get_base_class()
        p = self._get_base_est_params()
        return c(**p)

    def _get_base_class(self):
        raise NotImplementedError

    def _get_base_est_params(self):
        return {'pen_func': self.pen_func,
                'pen_func_kws': self.pen_func_kws,
                'init': self.init,

                'lla_n_steps': self.lla_n_steps,
                'lla_kws': self.lla_kws
                }

    def _set_tuning_values(self, X, init_data):

        if self.pen_vals is None:

            est = self._get_base_estimator()
            pen_val_max = est.get_pen_val_max(X, init_data)

            pen_val_seq = get_sequence_decr_max(max_val=pen_val_max,
                                                min_val_mult=self.pen_min_mult,
                                                num=self.n_pen_vals,
                                                spacing=self.pen_spacing)
        else:
            pen_val_seq = np.array(self.pen_vals)

        self.pen_val_seq_ = np.sort(pen_val_seq)[::-1]  # ensure decreasing

    def get_tuning_param_grid(self):
        return {'pen_val': self.pen_val_seq_}
