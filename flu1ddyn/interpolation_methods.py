#############################################################################
# Copyright (c) 2021 - 2022, CEA
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
##############################################################################

import numpy as np

EPS = 10**-6


def integrale_vol_div(flux, dx):
    return 1.0 / dx * (flux[1:] - flux[:-1]).view(np.ndarray)


def interpolate(center_value, I=None, cl=1, schema="weno", cv_0=0.0, cv_n=0.0):
    """
    Calcule le delta de convection aux bords des cellules

    Args:
        I: s'il faut prendre en compte si le stencil traverse une interface
        schema:
        center_value: les valeurs présentes aux centres des cellules
        cl: si cl = 1, on utilise la préiodicité, si cl = 0 on utilise cv_0 et cv_n
        cv_0 : la valeur au centre au bord en -1
        cv_n : la valeur au centre au bord en n+1

    Returns:
        l'interpolation aux faces
    """
    if len(center_value.shape) != 1:
        raise NotImplementedError
    if schema == "upwind":
        interpolated_value = interpolate_from_center_to_face_upwind(
            center_value, cl=cl, cv_0=cv_0
        )
    elif schema == "center":
        interpolated_value = interpolate_from_center_to_face_center(
            center_value, cl=cl, cv_0=cv_0, cv_n=cv_n
        )
    elif schema == "center_h":
        interpolated_value = interpolate_from_center_to_face_center_h(
            center_value, cl=cl
        )
    elif schema == "quick":
        interpolated_value = interpolate_from_center_to_face_quick(
            center_value, cl=cl, cv_0=cv_0, cv_n=cv_n
        )
    elif schema == "quick upwind":
        if I is None:
            raise NotImplementedError
        interpolated_value = interpolate_from_center_to_face_quick_upwind_interface(
            center_value, I, cl=cl, cv_0=cv_0, cv_n=cv_n
        )
    elif schema == "weno":
        interpolated_value = interpolate_from_center_to_face_weno(
            center_value, cl=cl, cv_0=cv_0, cv_n=cv_n
        )
    elif schema == "weno upwind":
        if I is None:
            raise NotImplementedError
        interpolated_value = interpolate_center_value_weno_to_face_upwind_interface(
            center_value, I, cl=cl, cv_0=cv_0, cv_n=cv_n
        )
    else:
        raise NotImplementedError
    if cl == 1:
        if np.abs(interpolated_value[0] - interpolated_value[-1]) > 10**-10:
            raise Exception("Les flux entrants et sortants sont censés être identiques")
    res = Flux(interpolated_value)
    if cl == 1:
        res.perio()
    return res


def interpolate_from_center_to_face_center(center_value, cl=1, cv_0=0.0, cv_n=0.0):
    interpolated_value = np.zeros((center_value.shape[0] + 1,))
    interpolated_value[1:-1] = (center_value[:-1] + center_value[1:]) / 2.0
    if cl == 1:
        interpolated_value[0] = (center_value[0] + center_value[-1]) / 2.0
        interpolated_value[-1] = (center_value[0] + center_value[-1]) / 2.0
    elif cl == 2:
        interpolated_value[0] = interpolated_value[1]
        interpolated_value[-1] = interpolated_value[-2]
    elif cl == 0:
        interpolated_value[0] = (center_value[0] + cv_0) / 2.0
        interpolated_value[-1] = (center_value[-1] + cv_n) / 2.0
    else:
        raise NotImplementedError
    return interpolated_value


def interpolate_from_center_to_face_center_h(center_value, cl=1):
    ext_center = np.zeros((center_value.shape[0] + 2,))
    ext_center[1:-1] = center_value

    if cl == 1:
        ext_center[0] = center_value[-1]
        ext_center[-1] = center_value[0]
    else:
        raise NotImplementedError
    cent0 = ext_center[:-1]
    cent1 = ext_center[1:]
    zero = np.abs(cent1 + cent0) < EPS
    interpolated_value = np.where(zero, 0.0, cent0 * cent1 / (cent1 + cent0) * 2.0)
    return interpolated_value


def interpolate_from_center_to_face_upwind(center_value, cl=1, cv_0=0.0):
    interpolated_value = np.zeros((center_value.shape[0] + 1,))
    interpolated_value[1:] = center_value
    if cl == 2:
        interpolated_value[0] = interpolated_value[1]
        interpolated_value[-1] = interpolated_value[-2]
    elif cl == 1:
        interpolated_value[0] = center_value[-1]
    elif cl == 0:
        interpolated_value[0] = cv_0
    else:
        raise NotImplementedError
    return interpolated_value


def interpolate_from_center_to_face_weno(a, cl=1, cv_0=0.0, cv_n=0.0):
    """
    Weno scheme

    Args:
        cv_n: center value at n+1
        cv_0: center value at -1
        a: the scalar value at the center of the cell
        cl: conditions aux limites, cl = 1: périodicité, cl=0 valeurs imposées aux bords à cv_0 et cv_n avec gradients
            nuls

    Returns:
        les valeurs interpolées aux faces de la face -1/2 à la face n+1/2
    """
    center_values = np.empty(a.size + 5)
    i0 = 3
    i_n = -2
    if cl == 1:
        center_values[:i0] = a[-i0:]
        center_values[i0:i_n] = a
        center_values[i_n:] = a[:-i_n]
    # elif cl == 0:
    #     center_values[:i0] = cv_0
    #     center_values[i0:i_n] = a
    #     center_values[i_n:] = cv_n
    else:
        raise NotImplementedError
    i_c0 = i0 - 1  # en amont de la premiere face car v > 0.
    i_cn = i_n  # la derniere cellule est en amont de la derniere face.

    # le premiere valeur de uj est en i0 - 1, c'est le centre de la cellule avant
    # le début du domaine (la première face). Par périodicité ça doit être le dernier
    # centre de a.
    ujm2 = center_values[i_c0 - 2 : i_cn - 2]
    ujm1 = center_values[i_c0 - 1 : i_cn - 1]
    uj = center_values[i_c0:i_cn]
    ujp1 = center_values[i_c0 + 1 : i_cn + 1]
    ujp2 = center_values[i_c0 + 2 :]

    # si on fait -1/6 * -3/2 + 5/6 * -1/2 + 1/3 * 1/2 on a 0.
    # on interpole donc bien pour la premiere face du domaine.
    f1 = 1.0 / 3 * ujm2 - 7.0 / 6 * ujm1 + 11.0 / 6 * uj
    f2 = -1.0 / 6 * ujm1 + 5.0 / 6 * uj + 1.0 / 3 * ujp1
    f3 = 1.0 / 3 * uj + 5.0 / 6 * ujp1 - 1.0 / 6 * ujp2
    eps = np.array(10.0**-6)
    b1 = (
        13.0 / 12 * (ujm2 - 2 * ujm1 + uj) ** 2
        + 1.0 / 4 * (ujm2 - 4 * ujm1 + 3 * uj) ** 2
    )
    b2 = 13.0 / 12 * (ujm1 - 2 * uj + ujp1) ** 2 + 1.0 / 4 * (ujm1 - ujp1) ** 2
    b3 = (
        13.0 / 12 * (uj - 2 * ujp1 + ujp2) ** 2
        + 1.0 / 4 * (3 * uj - 4 * ujp1 + ujp2) ** 2
    )
    w1 = 1.0 / 10 / (eps + b1) ** 2
    w2 = 3.0 / 5 / (eps + b2) ** 2
    w3 = 3.0 / 10 / (eps + b3) ** 2
    sum_w = w1 + w2 + w3
    w1 /= sum_w
    w2 /= sum_w
    w3 /= sum_w
    interpolated_value = f1 * w1 + f2 * w2 + f3 * w3
    return interpolated_value


def interpolate_from_center_to_face_quick(a, cl=1, cv_0=0.0, cv_n=0.0):
    """
    Quick scheme, in this case upwind is always on the left side.

    Args:
        cv_n: center value at n+1
        cv_0: center value at -1
        a: the scalar value at the center of the cell
        cl: conditions aux limites, cl = 1: périodicité, cl=0 valeurs imposées aux bords à cv_0 et cv_n avec gradients
            nuls

    Returns:
        les valeurs interpolées aux faces de la face -1/2 à la face n+1/2
    """
    center_values = np.empty(a.size + 4)
    if cl == 1:
        center_values[:2] = a[-2:]
        center_values[2:-2] = a
        center_values[-2:] = a[:2]
    elif cl == 0:
        center_values[:2] = cv_0
        center_values[2:-2] = a
        center_values[-2:] = cv_n
    else:
        raise NotImplementedError
    t0 = center_values[:-2]
    t1 = center_values[1:-1]
    t2 = center_values[2:]
    curv = (t2 - t1) - (t1 - t0)  # curv est trop long de 1 devant et derrière
    curv = curv[:-1]  # selection du curv amont, curv commence donc à -1 et finit à n

    smax = np.where(t2 > t0, t2, t0)
    smin = np.where(t2 < t0, t2, t0)
    dmax = smax - smin
    DMINFLOAT = 10.0**-15
    ds = np.where(np.abs(dmax) > DMINFLOAT, dmax, 1.0)
    fram = np.where(np.abs(dmax) > DMINFLOAT, ((t1 - smin) / ds * 2.0 - 1.0) ** 3, 0.0)
    fram = np.where(
        fram < 1.0, fram, 1.0
    )  # idem fram est trop long de 1 devant et derrière
    fram = np.where(
        fram[1:] > fram[:-1], fram[1:], fram[:-1]
    )  # selection du max entre fram(i-1) et fram(i), fram est
    # de taille n+1
    tamont = t1[:-1]
    taval = t1[1:]
    interp_1 = (tamont + taval) / 2.0 - 1.0 / 8.0 * curv  # + 1/4. * tamont
    interpolated_value = (1.0 - fram) * interp_1 + fram * tamont
    return interpolated_value


def interpolate_from_center_to_face_lin3(a, cl=1, cv_0=0.0, cv_n=0.0):
    center_values = np.empty(a.size + 4)
    if cl == 1:
        center_values[:2] = a[-2:]
        center_values[2:-2] = a
        center_values[-2:] = a[:2]
    else:
        raise NotImplementedError
    t0 = center_values[:-2]
    t1 = center_values[1:-1]
    t2 = center_values[2:]
    interpolated_value = 0.75 * t1 - 0.125 * t0 + 0.375 * t2
    return interpolated_value


def interpolate_from_center_to_face_quick_upwind_interface(
    a, I, cl=1, cv_0=0.0, cv_n=0.0
):
    """
    Quick scheme and upwind at the interface, in this case upwind is always on the left side.
    On doit ajouter pour la périodicité 2 cellules amont (-2 et -1) et une cellule après (n+1)

    Args:
        cv_n: center value at n+1
        cv_0: center value at -1
        a: the scalar value at the center of the cell
        I: the indicator function
        cl: conditions aux limites, cl = 1: périodicité, cl=0 valeurs imposées aux bords à cv_0 et cv_n avec gradients
            nuls

    Returns:
        les valeurs interpolées aux faces de la face -1/2 à la face n+1/2
    """
    res = interpolate_from_center_to_face_quick(a, cl)
    center_values = np.empty(a.size + 3)
    phase_indicator = np.empty(I.size + 3)
    if cl == 1:
        center_values[:2] = a[-2:]
        center_values[2:-1] = a
        center_values[-1:] = a[:1]
        phase_indicator[:2] = I[-2:]
        phase_indicator[2:-1] = I
        phase_indicator[-1:] = I[:1]
        center_diph = phase_indicator * (1.0 - phase_indicator) != 0.0
    elif cl == 0:
        center_values[:2] = cv_0
        center_values[2:-1] = a
        center_values[-1:] = cv_n
        phase_indicator[:2] = 0.0
        phase_indicator[2:-1] = I
        phase_indicator[-1:] = 0.0
        center_diph = phase_indicator * (1.0 - phase_indicator) != 0.0
    else:
        raise NotImplementedError
    diph_jm1 = center_diph[:-2]
    diph_j = center_diph[1:-1]
    diph_jp1 = center_diph[2:]
    # f diphasic correspond aux faces dont le stencil utilisé par le QUICK traverse l'interface
    f_diph = diph_jm1 | diph_j | diph_jp1

    # interpolation upwind
    # res[f_diph] = center_values[2:-1][f_diph]
    res[f_diph] = center_values[1:-1][f_diph]
    return res


def interpolate_center_value_weno_to_face_upwind_interface(
    a, I, cl=1, cv_0=0.0, cv_n=0.0
):
    """
    interpolate the center value a[i] at the face res[i+1] (corresponding to the upwind scheme) on diphasic cells

    Args:
        cv_n:
        cv_0:
        cl: the limit condition, 1 is periodic
        a: the center values
        I: the phase indicator

    Returns:
        res
    """
    res = interpolate_from_center_to_face_weno(a, cl)
    center_values = np.empty(a.size + 5)
    phase_indicator = np.empty(I.size + 5)
    if cl == 1:
        center_values[:3] = a[-3:]
        center_values[3:-2] = a
        center_values[-2:] = a[:2]
        phase_indicator[:3] = I[-3:]
        phase_indicator[3:-2] = I
        phase_indicator[-2:] = I[:2]
        center_diph = phase_indicator * (1.0 - phase_indicator) != 0.0
    elif cl == 0:
        center_values[:3] = cv_0
        center_values[3:-2] = a
        center_values[-2:] = cv_n
        # en cas d'utilisation du schéma avec des conditions aux limites on n'est pas diphasique aux bords
        phase_indicator[:3] = 0.0
        phase_indicator[3:-2] = I
        phase_indicator[-2:] = 0.0
        center_diph = phase_indicator * (1.0 - phase_indicator) != 0.0
    else:
        raise NotImplementedError
    diph_jm2 = center_diph[:-4]
    diph_jm1 = center_diph[1:-3]
    diph_j = center_diph[2:-2]
    diph_jp1 = center_diph[3:-1]
    diph_jp2 = center_diph[4:]
    # f diphasic correspond aux faces dont le stencil utilisé par le WENO traverse l'interface
    f_diph = diph_jm2 | diph_jm1 | diph_j | diph_jp1 | diph_jp2

    # interpolation upwind
    res[f_diph] = center_values[2:-2][f_diph]
    return res


def grad(center_value, dx=1.0, cl=1):
    """
    Calcule le gradient aux faces

    Args:
        center_value: globalement lambda
        cl: si cl = 1 les gradients aux bords sont périodiques
        dx: le delta x

    Returns:
        le gradient aux faces
    """
    if len(center_value.shape) != 1:
        raise NotImplementedError
    gradient = np.zeros(center_value.shape[0] + 1)
    gradient[1:-1] = (center_value[1:] - center_value[:-1]) / dx
    if cl == 1:
        gradient[0] = (center_value[0] - center_value[-1]) / dx
        gradient[-1] = (center_value[0] - center_value[-1]) / dx
    if cl == 2:
        gradient[0] = 0
        gradient[-1] = 0
    return gradient


def grad_center4(center_value, dx=1.0, cl=1):
    """
    Calcule le gradient aux faces

    Args:
        center_value: globalement lambda
        cl: si cl = 1 les gradients aux bords sont périodiques
        dx: le delta x

    Returns:
        le gradient aux faces
    """
    if len(center_value.shape) != 1:
        raise NotImplementedError
    gradient = np.zeros(center_value.shape[0] + 1)
    gradient[2:-2] = (
        9.0 / 8 * (center_value[2:-1] - center_value[1:-2])
        - 1.0 / 24 * (center_value[3:] - center_value[:-4])
    ) / dx
    if cl == 1:
        gradient[0] = (
            9.0 / 8 * (center_value[0] - center_value[-1])
            - 1 / 24 * (center_value[1] - center_value[-2])
        ) / dx
        gradient[-1] = gradient[0]  # periodicity
        gradient[1] = (
            9.0 / 8 * (center_value[1] - center_value[0])
            - 1 / 24 * (center_value[2] - center_value[-1])
        ) / dx
        gradient[-2] = (
            9.0 / 8 * (center_value[-1] - center_value[-2])
            - 1 / 24 * (center_value[0] - center_value[-3])
        ) / dx
    else:
        raise NotImplementedError
    return gradient


class Flux(np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        obj = np.asarray(input_array, *args, **kwargs).view(cls)
        return obj

    def perio(self):
        self[-1] = self[0]
