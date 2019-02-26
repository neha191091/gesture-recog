from sklearn.linear_model import LogisticRegression


class LogisticRegressionSklearn(LogisticRegression):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='warn', max_iter=100,
                 multi_class='warn', verbose=0, warm_start=False, n_jobs=None):
        super().__init__(penalty, dual, tol, C,
                         fit_intercept, intercept_scaling, class_weight,
                         random_state, solver, max_iter,
                         multi_class, verbose, warm_start, n_jobs)
