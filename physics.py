"""
physics.py
----------
Core physics routines for particle-in-wave simulations.
All functions are stateless and dependency-free (only numpy).
"""

import numpy as np


def boris_pusher(
    x: np.ndarray,
    v: np.ndarray,
    E: np.ndarray,
    B: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boris algorithm: advance a charged particle one time step.

    Parameters
    ----------
    x  : position  (3,)
    v  : velocity  (3,)
    E  : electric field at x  (3,)
    B  : magnetic field at x  (3,)
    dt : time step

    Returns
    -------
    x_new, v_new : updated position and velocity
    """
    v_minus  = v + (E * dt / 2)
    t_vec    = B * dt / 2
    s_vec    = 2 * t_vec / (1.0 + np.dot(t_vec, t_vec))
    v_prime  = v_minus + np.cross(v_minus, t_vec)
    v_plus   = v_minus + np.cross(v_prime, s_vec)
    v_new    = v_plus + (E * dt / 2)
    x_new    = x + v_new * dt
    return x_new, v_new


def alfven_wave_fields(
    pos: np.ndarray,
    t: float,
    k_vec: np.ndarray,
    dB: float,
    phi: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slab Alfvén wave: k in x-z plane, δB in y-direction.

    Physics
    -------
    - ω  = k_∥ = k_z          (shear Alfvén dispersion, v_A = 1)
    - δB = dB · cos(k·x − ωt + φ) ŷ
    - E  = −k̂ × δB            (Faraday's law in ideal MHD)

    Parameters
    ----------
    pos   : particle position  (3,)
    t     : current time
    k_vec : wave vector        (3,)
    dB    : wave amplitude
    phi   : phase offset (rad); modes sharing a phi value form coherent structures

    Returns
    -------
    E     : electric field perturbation  (3,)
    dB_vec: magnetic field perturbation  (3,)
    """
    k_mag   = np.linalg.norm(k_vec)
    k_hat   = k_vec / k_mag
    omega   = k_vec[2]                          # k_∥ for shear Alfvén
    phase   = np.dot(k_vec, pos) - omega * t + phi
    dBy     = dB * np.cos(phase)
    dB_vec  = np.array([0.0, dBy, 0.0])
    E       = -np.cross(k_hat, dB_vec)
    return E, dB_vec