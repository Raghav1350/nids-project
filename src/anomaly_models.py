"""Custom anomaly estimators used by the NIDS project.

Kept in their own module (not inside a train script) so model bundles
pickled with these classes can be loaded from any process -- pickle
resolves classes by module path, and a class defined in a ``__main__``
train script cannot be found on load.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator


class AutoencoderAnomaly(BaseEstimator):
    """MLP autoencoder wrapped in sklearn's outlier-detector convention.

    decision_function returns -MSE so that *higher = more normal*, matching
    IsolationForest's convention used throughout this project.
    """

    def __init__(self, hidden_layer_sizes=(64, 32, 64), activation="relu",
                 solver="adam", max_iter=150, early_stopping=True,
                 validation_fraction=0.1, n_iter_no_change=10,
                 random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.neural_network import MLPRegressor

        X = np.asarray(X, dtype=np.float64)
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
        )
        self.mlp_.fit(X, X)  # autoencoder: reconstruct the input
        return self

    def _mse(self, X):
        X = np.asarray(X, dtype=np.float64)
        recon = self.mlp_.predict(X)
        return ((X - recon) ** 2).mean(axis=1)

    def decision_function(self, X):
        return -self._mse(X)

    def score_samples(self, X):
        return -self._mse(X)
