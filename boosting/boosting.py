from __future__ import annotations

from collections import defaultdict

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        model = self.base_model_class(**self.base_model_params)
        residual = -self.loss_derivative(y, predictions)
        bs_idx = np.random.choice(x.shape[0], int(self.subsample * x.shape[0]))
        X = x[bs_idx]
        residual = residual[bs_idx]
        new_model = model.fit(X, residual)
        gamma = self.find_optimal_gamma(y, predictions, new_model.predict(x))

        self.gammas.append(gamma)
        self.models.append(new_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        num_bad_iters = 0
        last_score = 1

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)
            self.history['train_loss'].append(self.loss_fn(y_train, train_predictions))
            self.history['validation_loss'].append(self.loss_fn(y_valid, valid_predictions))

            if self.early_stopping_rounds is not None:
                if num_bad_iters == self.early_stopping_rounds:
                    break
                current_score = self.score(x_valid, y_valid)
                if last_score >= current_score:
                    num_bad_iters += 1
                else:
                    last_score = current_score
                    num_bad_iters = 0

        if self.plot:
            sns.lineplot(data=self.history)

    def predict_proba(self, x):
        prediction = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            prediction += model.predict(x) * self.learning_rate * gamma
        
        probabilities = self.sigmoid(prediction)
        return np.vstack([1 - probabilities, probabilities]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        ans = np.zeros(self.models[0].feature_importances_.shape[0])
        for model in self.models:
            ans += model.feature_importances_
        return ans / len(self.models)
            
