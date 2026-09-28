from typing import Sequence

import numpy as np
from scipy import sparse


def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        np.ndarray: column vector.
    """
    return np.random.rand(dim, 1)


def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        sparse.coo_matrix: sparse column vector.
    """
    return sparse.random(
        dim,
        1,
        density=0.1,
        format="coo",
        data_rvs=np.random.rand,
    )


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition.

    Args:
        x (np.ndarray): 1st vector.
        y (np.ndarray): 2nd vector.

    Returns:
        np.ndarray: vector sum.
    """
    return x + y


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar.

    Args:
        x (np.ndarray): vector.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied vector.
    """
    return a * x


def linear_combination(
    vectors: Sequence[np.ndarray],
    coeffs: Sequence[float],
) -> np.ndarray:
    """Linear combination of vectors.

    Args:
        vectors (Sequence[np.ndarray]): list of vectors of len N.
        coeffs (Sequence[float]): list of coefficients of len N.

    Returns:
        np.ndarray: linear combination of vectors.
    """
    if len(vectors) != len(coeffs):
        raise ValueError("vectors and coeffs must have the same length")

    result = np.zeros_like(vectors[0], dtype=float)

    for vector, coeff in zip(vectors, coeffs):
        result += coeff * vector

    return result


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product.

    Args:
        x (np.ndarray): 1st vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: dot product.
    """
    return float(np.dot(x.ravel(), y.ravel()))


def norm(x: np.ndarray, order: int | float) -> float:
    """Vector norm: Manhattan, Euclidean or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 1, 2 or inf.

    Returns:
        float: vector norm
    """
    return float(np.linalg.norm(x.ravel(), ord=order))


def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors.

    Args:
        x (np.ndarray): 1st vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: distance.
    """
    return float(np.linalg.norm(x.ravel() - y.ravel(), ord=2))


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Angle between two vectors, in degrees.

    Despite the name, this returns the angle itself and not its cosine:
    0 for parallel vectors, 90 for orthogonal ones, 180 for opposite ones.

    Args:
        x (np.ndarray): 1st vector, shape (n, 1).
        y (np.ndarray): 2nd vector, shape (n, 1).

    Returns:
        float: angle in degrees, in [0, 180].
    """
    x_flat = x.ravel()
    y_flat = y.ravel()

    x_norm = np.linalg.norm(x_flat)
    y_norm = np.linalg.norm(y_flat)

    if x_norm == 0 or y_norm == 0:
        raise ValueError("Angle is undefined for a zero vector")

    cosine = np.dot(x_flat, y_flat) / (x_norm * y_norm)

    # Protect arccos from tiny floating-point errors
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check is vectors orthogonal.

    Args:
        x (np.ndarray): 1st vector.
        y (np.ndarray): 2nd vector.


    Returns:
        bool: are vectors orthogonal.
    """
    return bool(np.isclose(dot_product(x, y), 0.0))


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations.

    Args:
        a (np.ndarray): coefficient matrix.
        b (np.ndarray): ordinate values.

    Returns:
        np.ndarray: sytems solution
    """
    return np.linalg.solve(a, b)
