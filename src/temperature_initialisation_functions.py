import numpy as np
from src.main import Bulles


def get_T(x, markers=None, phy_prop=None):
    if phy_prop is None:
        raise Exception
    else:
        lda_1 = phy_prop.lda1
        lda_2 = phy_prop.lda2
    dx = x[1] - x[0]
    Delta = x[-1] + dx / 2.0
    if markers is None:
        marker = np.array([[0.25 * Delta, 0.75 * Delta]])
    elif isinstance(markers, Bulles):
        marker = markers.markers
    else:
        marker = markers.copy()

    if len(marker) > 1:
        raise Exception("Le cas pour plus d une bulle n est pas enore implémenté")
    marker = marker.squeeze()
    if marker[0] < marker[1]:
        m = np.mean(marker)
    else:
        m = np.mean([marker[0], marker[1] + Delta])
        if m > Delta:
            m -= Delta
    T1 = lda_2 * np.cos(2 * np.pi * (x - m) / Delta)
    w = opt.fsolve(
        lambda y: y * np.sin(2 * np.pi * y * (marker[0] - m) / Delta)
        - np.sin(2 * np.pi * (marker[0] - m) / Delta),
        np.array(1.0),
    )
    b = lda_2 * np.cos(2 * np.pi / Delta * (marker[0] - m)) - lda_1 * np.cos(
        2 * np.pi * w / Delta * (marker[0] - m)
    )
    T2 = lda_1 * np.cos(w * 2 * np.pi * ((x - m) / Delta)) + b
    T = T1.copy()
    if marker[0] < marker[1]:
        bulle = (x > marker[0]) & (x < marker[1])
    else:
        bulle = (x < marker[1]) | (x > marker[0])
    T[bulle] = T2[bulle]
    T -= np.min(T)
    T /= np.max(T)
    return T


def get_T_creneau(x, markers=None, phy_prop=None):
    if phy_prop is None:
        raise Exception(
            "Attetion, il faut des propriétés thermiques pour déterminer auto le nbre de bulles"
        )
    if markers is None:
        markers = Bulles(markers=markers, phy_prop=phy_prop)
    indic_liqu = markers.indicatrice_liquide(x)
    # T = 1. dans la vapeur, 0. dans le liquide, et Tm = int rhoCpT / int rhoCp dans les mailles diph.
    T = (
        phy_prop.rho_cp2
        * (1.0 - indic_liqu)
        / (phy_prop.rho_cp1 * indic_liqu + phy_prop.rho_cp2 * (1.0 - indic_liqu))
    )
    return T
