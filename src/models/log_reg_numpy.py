import numpy as np

_TINY = 1e-8


class LogisticRegressionNumpy:
    def __init__(self, max_iter, num_classes, num_features, lr=0.001, reg=0.1):
        self.max_iter = max_iter
        self.num_classes = num_classes
        self.num_features = num_features
        self.W = np.random.rand(num_features, num_classes)
        self.lr = lr
        self.reg = reg

    def fit(self, X, t, X_val=None, t_val=None, solver='gd', lbfgs_m=20):

        # Get one hot target
        t = self._get_one_hot_encoding(t)
        if t_val is not None:
            t_val = self._get_one_hot_encoding(t_val)

        if solver == 'gd':
            return self._grad_descent(X, t, X_val, t_val)
        elif solver == 'lbfgs':
            return self._limited_bfgs(X, t, lbfgs_m, X_val, t_val)
        else:
            assert 0, 'Bad optimizer name, try \'gd\' or \'lbfgs\''

    def score(self, X, t):

        # Get one hot target
        t = self._get_one_hot_encoding(t)

        # Get predictions
        y = self._get_prediction(X)

        score = self._get_score(t, y)

        return score

    def predict(self, X):
        return np.argmax(self._get_prediction(X), axis=1)

    def _grad_descent(self, X, t, X_val=None, t_val=None):

        scores = []
        log_losses = []
        scores_val = []
        log_losses_val = []

        for i in range(self.max_iter):

            # Get predictions
            y = self._get_prediction(X)

            # Get scores
            scores.append(self._get_score(t, y))
            log_losses.append(self._get_log_loss(t, y))

            if X_val is not None and t_val is not None:

                # Get val preds
                y_val = self._get_prediction(X_val)

                # Get val scores
                scores_val.append(self._get_score(t_val, y_val))
                log_losses_val.append(self._get_log_loss(t_val, y_val))

            # Get gradient
            g = np.matmul(X.T, (y-t))/y.shape[0] + self.reg*self.W

            # Gradient Descent
            self.W = self.W - self.lr * g

        if X_val is not None and t_val is not None:
            return scores, log_losses, scores_val, log_losses_val
        else:
            return scores, log_losses

    def _limited_bfgs(self, X, t, m, X_val=None, t_val=None):

        """
        Limited bfgs

        :param X:
        :param t:
        :param m:
        :return:
        """

        # Two loop recursion

        ws = []

        gs = []

        scores = []
        log_losses = []
        scores_val = []
        log_losses_val = []

        for i in range(self.max_iter):

            # Get predictions
            y = self._get_prediction(X)

            # Get scores
            scores.append(self._get_score(t, y))
            log_losses.append(self._get_log_loss(t, y))

            if X_val is not None and t_val is not None:
                # Get val preds
                y_val = self._get_prediction(X_val)

                # Get val scores
                scores_val.append(self._get_score(t_val, y_val))
                log_losses_val.append(self._get_log_loss(t_val, y_val))

            ws.append(self.W.flatten())

            if len(ws) > m+1:
                ws.pop(0)

            g = np.matmul(X.T, (y-t))/y.shape[0] + self.reg*self.W
            gs.append(g.flatten())

            if len(gs) > m+1:
                gs.pop(0)

            alphas = []
            rhos = []
            q = gs[-1]
            for j in range(len(ws)-2, -1, -1):
                del_w = ws[j+1] - ws[j]
                del_g = gs[j+1] - gs[j]
                rho = 1/np.dot(del_g, del_w)
                rhos.append(rho)
                alpha = rho * np.dot(del_w, q)
                alphas.append(alpha)
                q = q - alpha*del_g

            rhos.reverse()
            alphas.reverse()
            if len(ws) < 2:
                del_w = ws[-1]
                del_g = gs[-1]
            else:
                del_w = ws[-1] - ws[-2]
                del_g = gs[-1] - gs[-2]
            gamma_num = np.dot(del_w, del_g)
            gamma_den = np.dot(del_g, del_g)
            gamma = gamma_num/gamma_den
            z = -gamma*q
            for j in range(len(ws)-1):
                del_w = ws[j + 1] - ws[j]
                del_g = gs[j + 1] - gs[j]
                beta = rhos[j]*np.dot(del_g, z)
                z = z + (alphas[j] - beta)*del_w

            self.W = (ws[-1] - self.lr*z).reshape([self.num_features, self.num_classes])

        if X_val is not None and t_val is not None:
            return scores, log_losses, scores_val, log_losses_val
        else:
            return scores, log_losses

    def _get_one_hot_encoding(self, target):
        onehot = np.zeros([len(target), self.num_classes])
        onehot[np.arange(len(target)), target] = 1
        return onehot

    def _get_prediction(self, data):

        # Calculate activation
        y = np.matmul(data, self.W)

        # Calculate exponent
        y = np.exp(y - np.max(y, axis=1)[:, None])

        # Normalize
        y = y/(np.sum(y, axis=1) + _TINY)[:, None]

        return y

    def _get_score(self, t, y):
        y_classes = np.argmax(y, axis=1)
        t_classes = np.argmax(t, axis=1)
        return np.sum(t_classes == y_classes)/y_classes.shape[0]

    def _get_log_loss(self, t, y):
        return -np.sum(t*np.log(y))

