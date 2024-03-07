"""
Package contains simplified njit-compilable variants of numpy and scipy 
functions and based on them implementations of some other helpfull functions.
"""

import numpy as np
import numba as nb


@nb.njit
def linalg_norm(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Returns norm of matrix via specified axis. It is njit-compilable.

    :param np.ndarray x: array of shape (N, M)
    :param int axis: axis along which to compute the vector norms. 
        Defaults to 0
    :raises ValueError: if axis does not equal to 0 or 1.
    :return np.ndarray: norm of vectors.
    """
    N, M = x.shape
    if axis == 0:
        norm = np.zeros(M)
        for j in range(M):
            norm[j] = np.linalg.norm(x[:, j])
        return norm
    
    elif axis == 1:
        norm = np.zeros(N)
        for i in range(N):
            norm[i] = np.linalg.norm(x[i])
        return norm

    else:
        raise ValueError(f"axis {axis} is out of bounds for array of \
            dimension {x.ndim}")


@nb.njit
def mean(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Returns mean of matrix via specified axis. It is njit-compilable.

    :param np.ndarray x: array of shape (N, M).
    :param int axis: axis along which to compute the mean, defaults to 0.
    :raises ValueError: if axis does not equal to 0 or 1.
    :return np.ndarray: mean of vectors.
    """
    N, M = x.shape
    if axis == 0:
        m = np.zeros(M)
        for j in range(M):
            m[j] = np.mean(x[:, j])
        return m
    elif axis == 1:
        m = np.zeros(N)
        for i in range(N):
            m[i] = np.mean(x[i])
        return m
    else:
        raise ValueError(f"axis {axis} is out of bounds for array of \
            dimension {x.ndim}") 


@nb.njit
def clip_float(value: float, v_min: float, v_max: float) -> float:
    """
    Clips (limits) the float value in specified range.
    
    Given an interval, value outside the interval are clipped to the 
    interval edges.

    :param float value: value to clip.
    :param float v_min: minimum acceptable value.
    :param float v_max: maximum acceptable value.
    :return float: value clipped to the interval.
    """
    return np.array([value]).clip(v_min, v_max).item()


@nb.njit
def clip_vector(a: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    """
    Clips (limits) all values of vector in specified range.

    :param np.ndarray a: array of shape (N,).
    :param float a_min: minimum acceptable value.
    :param float a_max: maximum acceptable value. 
    :raises ValueError: if a has more dimensions than one.
    :return np.ndarray: array of shape (N,) with clipped values.
    """
    if a.ndim != 1:
        raise ValueError(f"a must be one-dimensional array but has \
            {a.ndim} dimensions")
    
    N = a.shape[0]
    clipped_a = np.zeros(N)
    for i in range(N):
        clipped_a[i] = clip_float(a[i], a_min, a_max)
    return clipped_a


@nb.njit
def normalized_vector(vector: np.ndarray) -> np.ndarray:
    """
    Returns normalized vector. If input vector has norm is close to zero then 
    will be returned himself.

    :param np.ndarray vector: vector to normalize.
    :return np.ndarray: normalized input vector. 
    """
    vector_norm = np.linalg.norm(vector)
    if np.isclose(vector_norm, 0.0):
        return vector
    return vector / vector_norm


@nb.njit
def angle_between(v1: np.ndarray, 
                  v2: np.ndarray, 
                  degrees: bool = True) -> float:
    """
    Returns angle between two vectors. If degrees is True angle will be 
    returned in degrees, if degrees is False angle will be returned in radians. 

    :param np.ndarray v1: first vector.
    :param np.ndarray v2: second vector.
    :param bool: if True the angle will be in degrees, if False the angle will
        be in radians.
    :return float angle: angle between v1 and v2.
    """
    v1_n = normalized_vector(v1)
    v2_n = normalized_vector(v2)
    
    angle_cos = clip_float(np.dot(v1_n, v2_n), -1.0, 1.0)
    angle = np.arccos(angle_cos)
    if degrees:
        return np.degrees(angle)
    return angle


@nb.njit
def angles_vector(v: np.ndarray, 
                  x: np.ndarray, 
                  degrees: bool = True) -> np.ndarray:
    """
    Returns angles between vector and all vectors in matrix.

    :param np.ndarray v: array of shape (K,) containing vector in K dimensions.
    :param np.ndarray x: array of shape (M, K) containing M vectors in K 
        dimensions.
    :param bool degrees: whether angles will be in degrees or radians, if True
        angles will be measured in degrees, defaults to True.
    :return np.ndarray angles: array of shape (M,) containing angles between
        v and all vectors from matrix x.
    """
    M, K = x.shape
    angles = np.zeros(M)
    for i in nb.prange(M):
        angles[i] = angle_between(v, x[i], degrees=degrees)
    return angles
