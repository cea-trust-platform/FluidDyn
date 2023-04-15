"""
In this module common local interpolation functions are defined and jit-compiled.
"""


import numpy as np
from numba import njit


@njit
def lagrange_amont(
    T0: float,
    T1: float,
    T2: float,
    x0: float,
    x1: float,
    x2: float,
    x_int: float,
) -> (float, float, float):
    """
    Dans cette méthode on veux que T0 et T1 soient amont de x_int
    C'est une interpolation d'ordre 3

    Args:
        T0:
        T1:
        T2:
        x0:
        x1:
        x2:
        x_int:

    Returns:

    """
    d0 = x0 - x_int
    d1 = x1 - x_int
    d2 = x2 - x_int

    mat = np.array(
        [
            [1.0, d0, d0**2 / 2.0],
            [1.0, d1, d1**2 / 2.0],
            [1.0, d2, d2**2 / 2.0],
        ],
        dtype=np.float_,
    )
    Tint, dTdx_int, d2Tdx2_int = np.dot(np.linalg.inv(mat), np.array([T0, T1, T2]))
    return Tint, dTdx_int, d2Tdx2_int


@njit
def lagrange_amont_grad(
    T0: float, T1: float, gradT0: float, x0: float, x1: float, x_int: float
) -> (float, float, float):

    """
    Schéma d'ordre 3 mais avec gradTi d'ordre 2, on peut dire qu'il est décentré car on utilise à la fois la valeur
    et le gradient amont.
    Args:
        T0:
        T1:
        gradT0:
        x0:
        x1:
        x_int:

    Returns:
        Les dérivées successives
    """
    d0 = x0 - x_int
    d1 = x1 - x_int

    mat = np.array(
        [[1.0, d0, d0**2 / 2.0], [1.0, d1, d1**2 / 2.0], [0.0, 1.0, d0]],
        dtype=np.float_,
    )
    Tint, dTdx_int, d2Tdx2_int = np.dot(np.linalg.inv(mat), np.array([T0, T1, gradT0]))
    return Tint, dTdx_int, d2Tdx2_int


@njit
def lagrange_centre_grad(
    Tgg: float,
    Tg: float,
    Ti: float,
    gradTi: float,
    xgg: float,
    xg: float,
    xi: float,
    x_face: float,
) -> (float, float, float):
    """
    On utilise la continuité de lad_grad_T et on écrit un DL à l'ordre 3 avec les points Tgg, Tg et Ti.
    On peut appliquer la méthode aussi bien à droite qu'à gauche.
    On résout l'inconnue en trop avec un DL 2 sur dTdx
    On retourne les gradients suivants ::

                              dgg    dg     I
                |---------------|-----------|
        +---------------+---------------+----
        |               |               |   |
        |       +      -|>      +      -|>  |
        |               |               |   |
        +---------------+---------------+----

    Args:

    Returns:

    """
    d0 = xgg - x_face
    d1 = xg - x_face
    d2 = xi - x_face

    mat = np.array(
        [
            [1.0, d0, d0**2 / 2.0, d0**3 / 6.0],
            [1.0, d1, d1**2 / 2.0, d1**3 / 6.0],
            [1.0, d2, d2**2 / 2.0, d2**3 / 6.0],
            [0.0, 1.0, d2, d2**2 / 2.0],
        ],
        dtype=np.float_,
    )
    Tint, dTdx_int, d2Tdx2_int, _ = np.dot(
        np.linalg.inv(mat), np.array([Tgg, Tg, Ti, gradTi])
    )
    return Tint, dTdx_int, d2Tdx2_int


@njit
def upwind(T0: float, T1: float, x0: float, x1: float) -> (float, float, float):
    """
    Uses upwind temperature for interpolation.

    Args:
        T0:
        T1:
        x0:
        x1:

    Returns:
        T, dTdx, d2Tdx2
    """
    Tint = T0
    dTdx_int = (T1 - T0) / (x1 - x0)
    d2Tdx2_int = 0.0
    return Tint, dTdx_int, d2Tdx2_int


@njit
def lagrange_amont_centre(
    T0: float,
    T1: float,
    T2: float,
    x0: float,
    x1: float,
    x2: float,
    x_int: float,
) -> (float, float, float):
    """
    Dans cette méthode on veux que T0 et T1 soient amont de x_int
    C'est une interpolation d'ordre 1, qui mélange une interpolation amont 1 décentrée d'une interpolation
    linéaire 1

    Args:
        T0:
        T1:
        T2:
        x0:
        x1:
        x2:
        x_int:

    Returns:

    """
    T_int_amont = T1 + (T1 - T0) / (x1 - x0) * (x_int - x1)
    T_int_centre = T1 + (T2 - T1) / (x2 - x1) * (x_int - x1)
    Tint = (T_int_centre + T_int_amont) / 2.0

    return Tint, np.nan, np.nan


@njit
def amont_decentre(
    T0: float, gradT0: float, x0: float, xint: float
) -> (float, float, float):
    """
    Dans cette méthode on veux que T0 soit amont de xint, et gradT0 soit le gradient en T0.
    C'est une interpolation d'ordre 1 décentrée amont.
    """

    Tint = T0 + gradT0 * (xint - x0)
    d2Tdx2_int = 0.0
    return Tint, gradT0, d2Tdx2_int
