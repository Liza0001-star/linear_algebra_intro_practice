"""Practice 5: regularization as a cure for an ill-posed estimation problem.

Ordinary least squares on correlated, noisy features is ill-posed in the sense of
Hadamard: the solution is not stable under small perturbations of the data. Ridge
and Lasso restore stability by adding a penalty term. Your job is to build the six
estimators below, then write up the comparison (see README, "What to hand in").
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes

ALPHA_GRID = np.linspace(1e-3, 1e2, 300)
C_GRID = np.logspace(-3, 2, 60)


def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """Split into train/test, then standardise using training statistics only.

    The split comes first on purpose. Fitting the scaler on the full matrix would
    leak test-set means and variances into training - a small, silent error that
    inflates every score you report afterwards.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): target vector.

    Returns:
        list[np.ndarray]: [X_train, X_test, y_train, y_test], features standardised.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)

    return [scaler.transform(X_train), scaler.transform(X_test), y_train, y_test]


def get_regression_data() -> list[np.ndarray]:
    """Load and preprocess the diabetes dataset for regression tasks.

    Returns:
        list[np.ndarray]: [X_train, X_test, y_train, y_test].
    """
    data = load_diabetes()
    return preprocess(data.data, data.target)


def get_classification_data() -> list[np.ndarray]:
    """Load and preprocess the breast cancer dataset for classification tasks.

    Returns:
        list[np.ndarray]: [X_train, X_test, y_train, y_test].
    """
    data = load_breast_cancer()
    return preprocess(data.data, data.target)


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit an unregularised least-squares model. This is your baseline.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): target vector.

    Returns:
        BaseEstimator: fitted LinearRegression.
    """
    raise NotImplementedError


def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit a ridge (L2) model, choosing `alpha` from ALPHA_GRID with GridSearchCV.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): target vector.

    Returns:
        BaseEstimator: the refitted best estimator (`.best_estimator_`), so that
            the returned object exposes `.alpha` and `.coef_`.
    """
    raise NotImplementedError


def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit a lasso (L1) model, choosing `alpha` from ALPHA_GRID with GridSearchCV.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): target vector.

    Returns:
        BaseEstimator: the refitted best estimator, exposing `.alpha` and `.coef_`.
    """
    raise NotImplementedError


def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit logistic regression with no regularisation at all.

    Note: `penalty=` was deprecated in scikit-learn 1.8 and is removed in 1.10.
    Switch the penalty off with `C=np.inf` instead.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): binary target vector.

    Returns:
        BaseEstimator: fitted LogisticRegression.
    """
    raise NotImplementedError


def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit L2-penalised logistic regression, tuning `C` over C_GRID with GridSearchCV.

    Note: L2 is `l1_ratio=0` in scikit-learn >= 1.8 (not `penalty="l2"`).

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): binary target vector.

    Returns:
        BaseEstimator: the refitted best estimator, exposing `.C` and `.coef_`.
    """
    raise NotImplementedError


def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit L1-penalised logistic regression, tuning `C` over C_GRID with GridSearchCV.

    Note: L1 is `l1_ratio=1` in scikit-learn >= 1.8, and it needs a solver that
    supports it - `liblinear` or `saga`. The default `lbfgs` raises ValueError.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): binary target vector.

    Returns:
        BaseEstimator: the refitted best estimator, exposing `.C` and `.coef_`.
    """
    raise NotImplementedError
