"""
V-Learner: Doubly Robust (DR) Learner with Cross-Fitting, Bootstrap Uncertainty,
and Leakage-Free Conformal Prediction Intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
)
from sklearn.utils.validation import check_is_fitted


def _is_binary(y: np.ndarray) -> bool:
    y = np.asarray(y).ravel()
    u = np.unique(y)
    return u.size <= 2 and np.all(np.isin(u, [0, 1]))


def _predict_mean_outcome(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Predict E[Y|X]. If classifier -> predict_proba[:,1], else -> predict."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)
    return model.predict(X)


@dataclass
class _ConformalState:
    alpha: float
    quantile: float


class VLearner(BaseEstimator):
    """
    V-Learner (DR + Cross-Fitting):

    1) Cross-fit nuisance models:
       - propensity e(x) = P(T=1|X)
       - outcome regression μ0(x)=E[Y|X,T=0], μ1(x)=E[Y|X,T=1]

    2) Build DR pseudo-outcome:
       τ_dr = (μ1 - μ0) + T*(Y-μ1)/e - (1-T)*(Y-μ0)/(1-e)

    3) Fit a CATE model τ(x) ≈ τ_dr with variance-stabilizing weights w(x) = e(1-e)

    4) Bootstrap ensemble for epistemic uncertainty and leakage-free conformal intervals
       calibrated on a held-out split.
    """

    def __init__(
        self,
        propensity_model: Optional[BaseEstimator] = None,
        outcome_model: Optional[BaseEstimator] = None,
        cate_model: Optional[BaseEstimator] = None,
        n_splits: int = 5,
        n_bootstrap: int = 50,
        conformal_alpha: float = 0.1,
        propensity_clip: Tuple[float, float] = (0.02, 0.98),
        mu_clip: Tuple[float, float] = (1e-6, 1 - 1e-6),
        pseudo_outcome_clip_quantiles: Optional[Tuple[float, float]] = (0.01, 0.99),
        use_variance_stabilizing_weights: bool = True,
        calibration_fraction: float = 0.2,
        random_state: Optional[int] = None,
    ):
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.cate_model = cate_model
        self.n_splits = n_splits
        self.n_bootstrap = n_bootstrap
        self.conformal_alpha = conformal_alpha
        self.propensity_clip = propensity_clip
        self.mu_clip = mu_clip
        self.pseudo_outcome_clip_quantiles = pseudo_outcome_clip_quantiles
        self.use_variance_stabilizing_weights = use_variance_stabilizing_weights
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def _default_propensity_model(self) -> BaseEstimator:
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=10,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _default_outcome_model_binary(self) -> BaseEstimator:
        return GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=self.random_state,
        )

    def _default_cate_model(self) -> BaseEstimator:
        return ExtraTreesRegressor(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=10,
            min_samples_split=20,
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _compute_dr_pseudo_outcome(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        e: np.ndarray,
        mu0: np.ndarray,
        mu1: np.ndarray,
    ) -> np.ndarray:
        lo, hi = self.propensity_clip
        e = np.clip(e, lo, hi)
        return (mu1 - mu0) + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)

    def _winsorize(self, v: np.ndarray) -> np.ndarray:
        q = self.pseudo_outcome_clip_quantiles
        if q is None:
            return v
        q_lo, q_hi = q
        lo, hi = np.quantile(v, [q_lo, q_hi])
        return np.clip(v, lo, hi)

    def _variance_weights(self, e: np.ndarray) -> np.ndarray:
        lo, hi = self.propensity_clip
        e = np.clip(e, lo, hi)
        w = e * (1.0 - e)
        w = w / (np.mean(w) + 1e-12)
        return w

    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray):
        X = np.asarray(X)
        T = np.asarray(T).ravel().astype(int)
        Y = np.asarray(Y).ravel()

        n = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self._is_binary_outcome_ = _is_binary(Y)

        prop_model = (
            self._default_propensity_model()
            if self.propensity_model is None
            else self.propensity_model
        )

        if self.outcome_model is not None:
            out_model = self.outcome_model
        else:
            if self._is_binary_outcome_:
                out_model = self._default_outcome_model_binary()
            else:
                raise ValueError(
                    "For non-binary outcomes, please pass an outcome_model (regressor)."
                )

        cate_model = (
            self._default_cate_model() if self.cate_model is None else self.cate_model
        )

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        e_hat = np.zeros(n, dtype=float)
        mu0_hat = np.zeros(n, dtype=float)
        mu1_hat = np.zeros(n, dtype=float)

        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            T_tr = T[train_idx]
            Y_tr = Y[train_idx]

            pm = clone(prop_model)
            pm.fit(X_tr, T_tr)
            e_hat[test_idx] = pm.predict_proba(X_te)[:, 1]

            m0 = clone(out_model)
            m1 = clone(out_model)

            if np.any(T_tr == 0):
                m0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
            else:
                m0.fit(X_tr, Y_tr)

            if np.any(T_tr == 1):
                m1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
            else:
                m1.fit(X_tr, Y_tr)

            mu0_hat[test_idx] = _predict_mean_outcome(m0, X_te)
            mu1_hat[test_idx] = _predict_mean_outcome(m1, X_te)

        if self._is_binary_outcome_:
            lo, hi = self.mu_clip
            mu0_hat = np.clip(mu0_hat, lo, hi)
            mu1_hat = np.clip(mu1_hat, lo, hi)

        tau_dr = self._compute_dr_pseudo_outcome(Y, T, e_hat, mu0_hat, mu1_hat)
        tau_dr = self._winsorize(tau_dr)

        if self.use_variance_stabilizing_weights:
            w = self._variance_weights(e_hat)
        else:
            w = np.ones_like(tau_dr)

        X_train, X_cal, y_train, y_cal, w_train, _ = train_test_split(
            X,
            tau_dr,
            w,
            test_size=self.calibration_fraction,
            random_state=self.random_state,
        )

        self.final_model_ = clone(cate_model)
        try:
            self.final_model_.fit(X_train, y_train, sample_weight=w_train)
        except TypeError:
            self.final_model_.fit(X_train, y_train)

        self.ensemble_models_ = []
        rng = np.random.RandomState(self.random_state)
        n_train = X_train.shape[0]

        for b in range(int(self.n_bootstrap)):
            idx = rng.choice(n_train, size=n_train, replace=True)
            xb = X_train[idx]
            yb = y_train[idx]
            wb = w_train[idx]

            mb = clone(cate_model)
            try:
                mb.fit(xb, yb, sample_weight=wb)
            except TypeError:
                mb.fit(xb, yb)
            self.ensemble_models_.append(mb)

        cal_pred = self.effect(X_cal)
        cal_resid = np.abs(y_cal - cal_pred)
        q = float(np.quantile(cal_resid, 1.0 - self.conformal_alpha))
        self._conformal_ = _ConformalState(alpha=self.conformal_alpha, quantile=q)

        self.propensity_ = np.clip(e_hat, *self.propensity_clip)
        self.mu0_ = mu0_hat
        self.mu1_ = mu1_hat
        self.tau_dr_ = tau_dr
        self.sample_weight_ = w

        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["final_model_"])
        X = np.asarray(X)
        return self.final_model_.predict(X)

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["ensemble_models_"])
        X = np.asarray(X)
        preds = np.array([m.predict(X) for m in self.ensemble_models_])
        return preds.mean(axis=0)

    def effect_uncertainty(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["ensemble_models_"])
        X = np.asarray(X)
        preds = np.array([m.predict(X) for m in self.ensemble_models_])
        return preds.std(axis=0)

    def effect_interval(self, X: np.ndarray, alpha: float = 0.1):
        check_is_fitted(self, ["_conformal_"])
        X = np.asarray(X)
        tau = self.effect(X)

        if abs(alpha - self._conformal_.alpha) < 1e-12:
            margin = self._conformal_.quantile
        else:
            margin = (
                self._conformal_.quantile
                * (self._conformal_.alpha / max(alpha, 1e-6)) ** 0.5
            )

        return tau - margin, tau + margin
