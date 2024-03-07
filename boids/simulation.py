"""
The module contains functions for boids simulation.    
"""

from tools import *

import numpy as np
import numba as nb

def init_boids(n_boids: int,
               asp: float,
               v_range: tuple[float, float] = (0.0, 1.0)
               ) -> np.ndarray:
    """
    Initializes boid's coordinates and speeds from uniform distribution.

    :param int n_boids: number of boids.
    :param float asp: maximum value of boid's x-coordinate.
    :param tuple[float, float] v_range: range of boid's speed's norm, 
        defaults to (0.0, 1.0).
    """

    boids = np.zeros(shape=(n_boids, 6))

    boids[:, 0] = np.random.uniform(low=0.0, high=asp, size=n_boids)
    boids[:, 1] = np.random.uniform(low=0.0, high=1.0, size=n_boids)

    v_min, v_max = v_range
    v_norm = np.random.uniform(low=v_min, high=v_max, size=n_boids)
    v_angle = np.random.uniform(low=0.0, high=2 * np.pi, size=n_boids)
    boids[:, 2] = v_norm * np.cos(v_angle)
    boids[:, 3] = v_norm * np.sin(v_angle)

    return boids


@nb.njit
def get_directions(boids: np.ndarray, dt: float) -> np.ndarray:
    """
    Returns the start and end coordinates of the last boids move.

    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param float dt: magnitude of time change.
    :return np.ndarray arr: array of shape (n_boids, 4), where
        arr[i] = [x0_i, y0_i, x1_i, y1_i].
    """
    return np.hstack((
        boids[:, 0:2] - dt * boids[:, 2:4],
        boids[:, 0:2]
    ))


@nb.njit
def clip_v(boids: np.ndarray, v_range: tuple[float, float]) -> None:
    """
    Fixes boid's acceleration's norms in specified range.

    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param tuple[float, float] v_range: range of acceptable values of 
        velocity vector norm.
    """
    _, v_max = v_range
    v_norm = linalg_norm(boids[:, 2:4], axis=1)
    mask = (v_norm > v_max)

    boids[mask, 2:4] /= v_norm[mask].reshape(-1, 1)
    boids[mask, 2:4] *= v_max


@nb.njit
def clip_w(boids: np.ndarray, w_range: tuple[float, float]) -> None:
    """
    Fixes boid's acceleration's norms in specified range. 

    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param tuple[float, float] w_range: range of acceptable values of 
        acceleration vector norm.
    """
    _, w_max = w_range
    w_norm = linalg_norm(boids[:, 4:6], axis=1)
    mask = (w_norm > w_max)

    boids[mask, 4:6] /= w_norm[mask].reshape(-1, 1)
    boids[mask, 4:6] *= w_max


@nb.njit
def propagate(boids: np.ndarray,
              dt: float,
              v_range: tuple[float, float],
              w_range: tuple[float, float]) -> None:
    """
    Updates boid's speeds and coordinates, previously clamping accelerations
    and velocities norms at specified intervals (ranges). 
    
    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param float dt: magnitude of time change.
    :param tuple[float, float] v_range: range of acceptable values of velocity
        vector norm.
    :param tuple[float, float] w_range: range of acceptable values of 
        acceleration vector norm. 
    """
    clip_w(boids, w_range)
    boids[:, 2:4] += dt * boids[:, 4:6]
    clip_v(boids, v_range)
    boids[:, 0:2] += dt * boids[:, 2:4]


@nb.njit
def distance_mask(
        boids: np.ndarray,
        idx: int,
        perception: float
) -> np.ndarray:
    """
    Returns mask of being close to specified boid.

    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param int idx: index of boid of interest.
    :param float perception: maximum distance from specified boid.
    :return np.ndarray: bool array of shape (n_boids,).
    """
    n_boids, _ = boids.shape
    indeces = np.arange(n_boids)
    
    delta_r = boids[:, 0:2] - boids[idx, 0:2]
    delta_x = delta_r[:, 0]
    delta_y = delta_r[:, 1]
    
    # Discard boids are far away from boid of interest. 
    x_mask = (-perception <= delta_x) & (delta_x <= perception)
    y_mask = (-perception <= delta_y) & (delta_y <= perception)
    first_mask = x_mask & y_mask
    
    dist = linalg_norm(delta_r[first_mask], axis=1)
    second_mask = dist <= perception
    
    mask_indeces = indeces[first_mask][second_mask]
    mask = np.full(shape=n_boids, fill_value=False)
    mask[mask_indeces] = True
    mask[idx] = False
    
    return mask


@nb.njit
def visibility_mask(
        boids: np.ndarray,
        idx: int,
        alpha: float,
        perception: float
) -> np.ndarray:
    """
    Returns visibility mask for specified boid.

    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param int idx: index of boid of interest.
    :param float alpha: angle of view in degrees.
    :param float perception: maximum distance from specified boid.
    :return np.ndarray: bool array of shape (n_boids, 6).
    """
    n_boids, _ = boids.shape
    indeces = np.arange(n_boids)
    
    dist_mask = distance_mask(boids, idx, perception)
    if not dist_mask.any():
        return np.full(shape=n_boids, fill_value=False)
    
    v = boids[idx, 2:4]
    r = boids[dist_mask, 0:2] - boids[idx, 0:2]
    angles = angles_vector(v, r)
    angle_mask = np.logical_and(-alpha/2 <= angles, angles <= alpha/2)
    
    mask_indeces = indeces[dist_mask][angle_mask]
    mask = np.full(shape=n_boids, fill_value=False)
    mask[mask_indeces] = True
    mask[idx] = False
    return mask


@nb.njit
def cohesion(
        boids: np.ndarray,
        idx: int,
        neighbor_mask: np.ndarray,
        perception: float
) -> np.ndarray:
    """
    Returns cohesion component to update acceleration of specified boid. 

    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param int idx: index of boid of interest.
    :param np.ndarray neighbor_mask: mask of specified boid's neighborhood.
    :param float perception: maximum distance from specified boid.
    :return np.ndarray: cohesion component - array of shape (2,).
    """
    center = mean(boids[neighbor_mask, 0:2], axis=0)
    a = (center - boids[idx, 0:2]) / perception
    return a


@nb.njit
def separation(
        boids: np.ndarray,
        idx: int,
        neighbor_mask: np.ndarray,
        perception: float
) -> np.ndarray:
    """
    Returns separation component to update acceleration of specified boid.

    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param int idx: index of boid of interest.
    :param np.ndarray neighbor_mask: mask of specified boid's neighborhood.
    :param float perception: maximum distance from specified boid.
    :return np.ndarray: separation component - array of shape (2,).
    """
    d = mean(boids[neighbor_mask, 0:2] - boids[idx, 0:2], axis=0)
    return -d / (np.linalg.norm(d) ** 2 + 1)


@nb.njit
def alignment(
        boids: np.ndarray,
        idx: int,
        neighbor_mask: np.ndarray,
        v_range: tuple[float, float]
) -> np.ndarray:
    """
    Returns alignment component to update acceleration of specified boid.

    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param int idx: index of boid of interest.
    :param np.ndarray neighbor_mask: mask of specified boid's neighborhood.
    :param tuple[float, float] v_range: range of acceptable values of 
        velocity vector norm.
    :return np.ndarray: alignment component - array of shape (2,).
    """
    v_min, v_max = v_range
    v_mean = mean(boids[neighbor_mask, 2:4], axis=0)
    a = (v_mean - boids[idx, 2:4]) / (2 * v_max)
    return a


@nb.njit
def walls(
        boids: np.ndarray,
        asp: float
) -> np.ndarray:
    """
    Returns proximity components to wall to update acceleration 
    of all boids.

    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param float asp: maximum value of boid's x-coordinate.
    :return np.ndarray: wall components - array of shape (n_boids, 2).
    """
    c, order = 1, 2
    x = boids[:, 0]
    y = boids[:, 1]

    w_left = 1 / (np.abs(x) + c) ** order
    w_right = - 1 / (np.abs(x - asp) + c) ** order

    w_bottom = 1 / (np.abs(y) + c) ** order
    w_top = -1 / (np.abs(y - 1.0) + c) ** order

    return np.column_stack((w_left + w_right, w_bottom + w_top))


@nb.njit
def walls_collision(
    boids: np.ndarray,
    asp: float
) -> None:
    """
    Handles boids collision with wall. In event of a collision, it changes
    the sign of corresponding velocity component.

    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param float asp: maximum value of boid's x-coordinate.
    """
    eps = 1e-3
    
    x_left = (boids[:, 0] < 0.0)
    x_right = (boids[:, 0] > asp)
    y_bottom = (boids[:, 1] < 0)
    y_top = (boids[:, 1] > 1.0)
    
    boids[x_left, 0] = 0.0 + eps
    boids[x_left, 2] *= -1
    
    boids[x_right, 0] = asp - eps
    boids[x_right, 2] *= -1
    
    boids[y_bottom, 1] = 0.0 + eps
    boids[y_bottom, 3] *= -1
    
    boids[y_top, 1] = 1.0 - eps
    boids[y_top, 3] *= -1       
    

@nb.njit
def noise(n_boids: int,
          eps_min: float = -1.0,
          eps_max: float = 1.0) -> np.ndarray:
    """
    Generates noise component for boids model simulation. Returns
    random noise in interval (eps_min, eps_max) for each boid.

    :param int n_boids: number of boids.
    :param float eps_min: noise minimum value, defaults to -1.0.
    :param float eps_max: noise maximum value, defaults to 1.0.
    :return np.ndarray: array of shape (n_boids, 2) with noise. 
    """
    return np.random.uniform(low=eps_min, high=eps_max, size=(n_boids, 2))


@nb.njit(parallel=True)
def flocking(
        boids: np.ndarray,
        alpha: float,
        perception: float,
        coeffs: np.ndarray,
        asp: float,
        v_range: tuple[float, float]
) -> None:
    """
    Updates acceleration of boids according to the model and 
    specified interaction coefficients.

    :param np.ndarray boids: boids array of shape (n_boids, 6).
    :param float alpha: angle of view in degrees.
    :param float perception: maximum distance between boids to be in 
        visibility area.
    :param np.ndarray coeffs: interaction coefficients - array of shape (5,). 
    :param float asp: maximum value of boid's x-coordinate.
    :param tuple[float, float] v_range: range of acceptable values of 
        velocity vector norm
    :raises ValueError: In the case when the size of the coeffs array 
        is not equal to (5,). 
    """
    if coeffs.shape != (5,):
        raise ValueError(f"coeffs parameter must be array of shape (5,) but \
                        has shape {coeffs.shape}")

    n_boids, _ = boids.shape

    wall = walls(boids, asp)
    noises = noise(n_boids)

    a_coef = coeffs[0]
    c_coef = coeffs[1]
    s_coef = coeffs[2]
    w_coef = coeffs[3]
    n_coef = coeffs[4]

    for idx in nb.prange(n_boids):
        neighbor_mask = visibility_mask(boids, idx, alpha, perception)

        wal = wall[idx]
        eps = noises[idx]

        if neighbor_mask.any():
            alg = alignment(boids, idx, neighbor_mask, v_range)
            coh = cohesion(boids, idx, neighbor_mask, perception)
            sep = separation(boids, idx, neighbor_mask, perception)
        else:
            alg = np.zeros(shape=2)
            coh = np.zeros(shape=2)
            sep = np.zeros(shape=2)

        boids[idx, 4:6] = (
                a_coef * alg +
                c_coef * coh +
                s_coef * sep +
                w_coef * wal +
                n_coef * eps
        )
