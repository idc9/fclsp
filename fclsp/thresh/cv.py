from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._split import check_cv
from functools import partial

import numpy as np
from time import time

from ya_glm.autoassign import autoassign
from ya_glm.utils import get_sequence_decr_max
from ya_glm.cv.cv_select import CVSlectMixin
from ya_glm.cv.CVPath import CVPathMixin, run_cv_path, add_params_to_cv_results


class ThreshCV(CVSlectMixin, CVPathMixin, BaseEstimator):
    """
    solve_path:

    """
    @autoassign
    def __init__(self,
                 thresh='hard', thresh_kws={},
                 n_pen_vals=100,
                 pen_vals=None,
                 pen_min_mult=1e-3,
                 pen_spacing='log',
                 opt_kws={},
                 cv=None,
                 cv_select_rule='best',
                 cv_select_metric=None,
                 cv_scorer=None,
                 cv_verbose=0, cv_n_jobs=None,
                 cv_pre_dispatch='2*n_jobs'):
        pass

    def fit(self, X, y=None):

        # set up the tuning parameter values sequence
        self._set_tuning_values(X=X)

        # run cross-validation on the raw data
        est = self._get_base_estimator()
        self.cv_data_ = {}
        start_time = time()
        self.cv_results_ = self._run_cv(X, cv=self.cv)
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

    def get_tuning_sequence(self):
        """
        Returns a list of tuning parameter values.
        Make sure the method that computes the cross-validation results
        orders the parameters in the same order as get_tuning_sequence()

        Output
        ------
        values: iterable
        """
        param_grid = self.get_tuning_param_grid()
        return list(ParameterGrid(param_grid))

    # def _get_cv_select_metric(self):

    #     if self.cv_select_metric is not None:
    #         # return cv_select_metric if provided by the user
    #         return self.cv_select_metric

    #     elif self.cv_scorer is not None:
    #         # if a cv_scorer is provided, try to infer the defualt score
    #         if hasattr(self.cv_scorer, 'default') \
    #                 and self.cv_scorer.default is not None:
    #             return self.cv_scorer.default

    #     # otherwise fall base on score
    #     return 'score'

    # def _select_tune_param(self, cv_results):
    #     """

    #     Parameters
    #     ----------
    #     cv_results:

    #     Output
    #     ------
    #     best_tune_idx, best_tune_params

    #     best_tune_idx:
    #     """
    #     select_score = self._get_cv_select_metric()

    #     return select_best_cv_tune_param(cv_results,
    #                                      rule=self.cv_select_rule,
    #                                      prefer_larger_param=True,
    #                                      score_name=select_score)

    def _get_base_estimator(self):
        c = self._get_base_class()
        p = self._get_base_params()
        return c(**p)

    def get_tuning_param_grid(self):
        return {'pen_val': self.pen_val_seq_}

    def _run_cv(self, X, cv=None):

        fit_and_score_path = partial(fit_and_score_path_from_est,
                                     est=self._get_base_estimator(),
                                     scorer=self.cv_scorer,
                                     pen_val_seq=self.pen_val_seq_)

        cv = check_cv(cv)

        cv_results, _ = \
            run_cv_path(X=X, y=None,
                        fold_iter=cv.split(X),
                        fit_and_score_path=fit_and_score_path,
                        kws={},
                        include_spilt_vals=False,  # maybe make this True?
                        add_params=False,
                        n_jobs=self.cv_n_jobs,
                        verbose=self.cv_verbose,
                        pre_dispatch=self.cv_pre_dispatch)

        # add parameter sequence to CV results
        # this allows us to pass fit_path a one parameter sequence
        # while cv_results_ uses different names
        param_seq = self.get_tuning_sequence()
        cv_results = add_params_to_cv_results(param_seq=param_seq,
                                              cv_results=cv_results)

        return cv_results

    def _get_base_params(self):
        return {'thresh': self.thresh, 'thresh_kws': self.thresh_kws}

    def _set_tuning_values(self, X, y=None):

        if self.pen_vals is None:

            pen_val_max = self.get_pen_val_max(X)

            pen_val_seq = get_sequence_decr_max(max_val=pen_val_max,
                                                min_val_mult=self.pen_min_mult,
                                                num=self.n_pen_vals,
                                                spacing=self.pen_spacing)
        else:
            pen_val_seq = np.array(self.pen_vals)

        self.pen_val_seq_ = np.sort(pen_val_seq)[::-1]  # ensure decreasing

    def get_pen_val_max(self, X):
        est = self._get_base_estimator()
        values, pre_pro_out = est._get_object_from_data(X)
        # return self._get_pen_val_max_from_values(values, pre_pro_out)
        return abs(values).max()


def fit_and_score_path_from_est(X, y, train, test,
                                est, pen_val_seq, scorer=None, kws={}):

    X_train = X[train, :]
    X_test = X[test, :]

    # TODO: maybe add score time
    path_results = {'train': [],
                    'test': [],
                    'param_seq': [],
                    'fit_runtime': []
                    # 'score_time': []
                    }

    # compute the solution path
    values, pre_pro_out = est._get_object_from_data(X_train)

    def shink(pen_val):
        est.set_params(pen_val=pen_val)
        return est._apply_shrinkage(values)
    solution_path = ((shink(pen_val), pen_val) for pen_val in pen_val_seq)

    for fit_out, pen_val in solution_path:

        est._set_fit(fit_out, pre_pro_out)

        if scorer is None:
            # score with estimator's defualt
            tst = est.score(X=X_test)
            tr = est.score(X=X_train)

        else:
            # custom scorer
            tst = scorer(est, X=X_test)
            tr = scorer(est, X=X_train)

        path_results['test'].append(tst)
        path_results['train'].append(tr)

        path_results['param_seq'].append({'pen_val': pen_val})

    # make sure these are formatted as list of dicts
    if not isinstance(path_results['test'][0], dict):
        path_results['test'] = [{'score': val}
                                for val in path_results['test']]

        path_results['train'] = [{'score': val}
                                 for val in path_results['train']]

    if len(path_results['fit_runtime']) == 0:
        del path_results['fit_runtime']

    return path_results
