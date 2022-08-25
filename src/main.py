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
from scipy import optimize as opt
from copy import deepcopy
import pickle
from glob import glob
import os


def integrale_vol_div(flux, dx):
    return 1.0 / dx * (flux[1:] - flux[:-1])


def interpolate(center_value, I=None, cl=1, schema="weno", cv_0=0.0, cv_n=0.0):
    """
    Calcule le delta de convection aux bords des cellules

    Args:
        I: s'il faut prendre en compte si le stencil traverse une interface
        schema:
        center_value: les valeurs présentes aux centres des cellules
        cl: si cl = 1, on prend des gradients nuls aux bords du domaine, si cl = 0 on utilise cv_0 et cv_n
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
            center_value, cl=cl, cv_0=cv_0, cv_n=cv_n
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
    return interpolated_value


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


def interpolate_from_center_to_face_center_h(center_value, cl=1, cv_0=0.0, cv_n=0.0):
    ext_center = np.zeros((center_value.shape[0] + 2,))
    ext_center[1:-1] = center_value

    if cl == 1:
        ext_center[0] = center_value[-1]
        ext_center[-1] = center_value[0]
    else:
        raise NotImplementedError
    cent0 = ext_center[:-1]
    cent1 = ext_center[1:]
    zero = np.abs(cent1 + cent0) < 10**-10
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
    if cl == 1:
        center_values[:3] = a[-3:]
        center_values[3:-2] = a
        center_values[-2:] = a[:2]
    elif cl == 0:
        center_values[:3] = cv_0
        center_values[3:-2] = a
        center_values[-2:] = cv_n
    else:
        raise NotImplementedError
    ujm2 = center_values[:-4]
    ujm1 = center_values[1:-3]
    uj = center_values[2:-2]
    ujp1 = center_values[3:-1]
    ujp2 = center_values[4:]
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


class Bulles:
    def __init__(
        self, markers=None, phy_prop=None, n_bulle=None, Delta=1.0, alpha=0.06, a_i=None
    ):
        self.diam = 0.0
        if phy_prop is not None:
            self.Delta = phy_prop.Delta
            self.alpha = phy_prop.alpha
            self.a_i = phy_prop.a_i
        else:
            self.Delta = Delta
            self.alpha = alpha
            self.a_i = a_i

        if markers is None:
            self.markers = []
            if n_bulle is None:
                if self.a_i is None:
                    raise Exception(
                        "On ne peut pas déterminer auto la géométrie des bulles sans le rapport surfacique"
                    )
                else:
                    # On détermine le nombre de bulle pour avoir une aire interfaciale donnée.
                    # On considère ici une géométrie 1D comme l'équivalent d'une situation 3D
                    n_bulle = int(self.a_i / 2.0 * self.Delta) + 1
            # Avec le taux de vide, on en déduit le diamètre d'une bulle. On va considérer que le taux de vide
            # s'exprime en 1D, cad : phy_prop.alpha = n*d*dS/(Dx*dS)
            self.diam = self.alpha * self.Delta / n_bulle
            centers = np.linspace(self.diam, self.Delta + self.diam, n_bulle + 1)[:-1]
            for center in centers:
                self.markers.append(
                    (center - self.diam / 2.0, center + self.diam / 2.0)
                )
            self.markers = np.array(self.markers)
            self.shift(self.Delta * 1./4)
        else:
            self.markers = np.array(markers).copy()
            mark1 = self.markers[0][1]
            mark0 = self.markers[0][0]
            while mark1 < mark0:
                mark1 += self.Delta
            self.diam = mark1 - mark0

        depasse = (self.markers > self.Delta) | (self.markers < 0.0)
        if np.any(depasse):
            print("Delta : ", self.Delta)
            print("markers : ", self.markers)
            print("depasse : ", depasse)
            raise Exception("Les marqueurs dépassent du domaine")

        self.init_markers = self.markers.copy()

    def __call__(self, *args, **kwargs):
        return self.markers

    def copy(self):
        cls = self.__class__
        copie = cls(markers=self.markers.copy(), Delta=self.Delta)
        copie.diam = self.diam
        return copie

    def indicatrice_liquide(self, x):
        """
        Calcule l'indicatrice qui correspond au liquide avec les marqueurs selon la grille x

        Args:
            x: les positions des centres des mailles

        Returns:
            l'indicatrice
        """
        i = np.ones_like(x)
        dx = x[1] - x[0]
        for markers in self.markers:
            if markers[0] < markers[1]:
                i[(x > markers[0]) & (x < markers[1])] = 0.0
            else:
                i[(x > markers[0]) | (x < markers[1])] = 0.0
            diph0 = np.abs(x - markers[0]) < dx / 2.0
            i[diph0] = (markers[0] - x[diph0]) / dx + 0.5
            diph1 = np.abs(x - markers[1]) < dx / 2.0
            i[diph1] = -(markers[1] - x[diph1]) / dx + 0.5
        return i

    def shift(self, dx):
        """
        On déplace les marqueurs vers la droite

        Args:
            dx: la distance du déplacement

        """
        self.markers += dx
        depasse = self.markers > self.Delta
        self.markers[depasse] -= self.Delta


class PhysicalProperties:
    def __init__(
        self,
        Delta=1.0,
        lda1=1.0,
        lda2=0.0,
        rho_cp1=1.0,
        rho_cp2=1.0,
        v=1.0,
        diff=1.0,
        a_i=358.0,
        alpha=0.06,
        dS=1.0,
    ):
        self._Delta = Delta
        self._lda1 = lda1
        self._lda2 = lda2
        self._rho_cp1 = rho_cp1
        self._rho_cp2 = rho_cp2
        self._v = v
        self._diff = diff
        self._a_i = a_i
        self._alpha = alpha
        self._dS = dS
        if self._v == 0.0:
            self._cas = "diffusion"
        elif self._diff == 0.0:
            self._cas = "convection"
        else:
            self._cas = "mixte"

    @property
    def Delta(self):
        return self._Delta

    @property
    def cas(self):
        return self._cas

    @property
    def lda1(self):
        return self._lda1

    @property
    def lda2(self):
        return self._lda2

    @property
    def rho_cp1(self):
        return self._rho_cp1

    @property
    def rho_cp2(self):
        return self._rho_cp2

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, val):
        self._v = val

    @property
    def diff(self):
        return self._diff

    @property
    def alpha(self):
        return self._alpha

    @property
    def a_i(self):
        return self._a_i

    @property
    def dS(self):
        return self._dS

    def isequal(self, prop):
        dic = {key: self.__dict__[key] == prop.__dict__[key] for key in self.__dict__}
        equal = np.all(list(dic.values()))
        if not equal:
            print("Attention, les propriétés ne sont pas égales :")
            print(dic)
        return equal


class NumericalProperties:
    def __init__(
        self,
        dx=0.1,
        dt=1.0,
        cfl=1.0,
        fo=1.0,
        schema="weno",
        time_scheme="euler",
        phy_prop=None,
        Delta=None,
    ):
        if phy_prop is None and Delta is None:
            raise Exception("Impossible sans phy_prop ou Delta")
        if phy_prop is not None:
            Delta = phy_prop.Delta
        self._cfl_lim = cfl
        self._fo_lim = fo
        self._schema = schema
        self._time_scheme = time_scheme
        self._dx_lim = dx
        nx = int(Delta / dx)
        dx = Delta / nx
        self._dx = dx
        self._x = np.linspace(dx / 2.0, Delta - dx / 2.0, nx)
        self._x_f = np.linspace(0.0, Delta, nx + 1)
        self._dt_min = dt

    @property
    def cfl_lim(self):
        return self._cfl_lim

    @property
    def fo_lim(self):
        return self._fo_lim

    @property
    def schema(self):
        return self._schema

    @property
    def time_scheme(self):
        return self._time_scheme

    @property
    def x(self):
        return self._x

    @property
    def x_f(self):
        return self._x_f

    @property
    def dx(self):
        return self._dx

    @property
    def dt_min(self):
        return self._dt_min

    @property
    def dx_lim(self):
        return self._dx_lim

    def isequal(self, prop):
        dic = {key: self.__dict__[key] == prop.__dict__[key] for key in self.__dict__}
        for key in dic.keys():
            if isinstance(dic[key], np.ndarray):
                dic[key] = np.all(dic[key])
        equal = np.all(list(dic.values()))
        if not equal:
            print("Attention, les propriétés numériques ne sont pas égales :")
            print(dic)
        return equal


class Problem:
    bulles: Bulles
    num_prop: NumericalProperties
    phy_prop: PhysicalProperties

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, name=None):
        self._imposed_name = name
        print()
        print(self.name)
        print("=" * len(self.name))
        if phy_prop is None:
            print("Attention, les propriétés physiques par défaut sont utilisées")
            phy_prop = PhysicalProperties()
        if num_prop is None:
            print("Attention, les propriétés numériques par défaut sont utilisées")
            num_prop = NumericalProperties()
        self.phy_prop = deepcopy(phy_prop)  # type: PhysicalProperties
        self.num_prop = deepcopy(num_prop)  # type: NumericalProperties
        self.bulles = self._init_bulles(markers)
        self.T = T0(self.num_prop.x, markers=self.bulles, phy_prop=self.phy_prop)
        self.dt = self.get_time()
        self.time = 0.0
        self.I = self.update_I()
        self.If = self.update_If()
        self.iter = 0
        self.flux_conv = Flux(np.zeros_like(self.num_prop.x_f))
        self.flux_diff = Flux(np.zeros_like(self.num_prop.x_f))
        self.E = None
        self.t = None
        print("Db / dx = %.2i" % (self.bulles.diam / self.num_prop.dx))

    def copy(self, pb):
        equal_prop = self.phy_prop.isequal(pb.phy_prop)
        if not equal_prop:
            raise Exception(
                "Impossible de copier le Problème, il n'a pas les mm propriétés physiques"
            )
        equal_prop_num = self.num_prop.isequal(pb.num_prop)
        if not equal_prop_num:
            raise Exception(
                "Impossible de copier le Problème, il n'a pas les mm propriétés numériques"
            )
        try:
            equal_init_markers = np.all(self.bulles.init_markers == pb.bulles.init_markers)
        except:
            equal_init_markers = True
            print("Attention, les markers initiaux ne sont pas enregistrés dans la référence")
        if not equal_init_markers:
            raise Exception(
                "Impossible de copier le Problème, il n'a pas les mm markers de départ"
            )
        # self.num_prop = deepcopy(self.num_prop)
        self.bulles = deepcopy(pb.bulles)
        self.T = pb.T.copy()
        self.dt = pb.dt
        self.time = pb.time
        self.I = self.update_I()
        self.If = self.update_If()
        self.iter = pb.iter
        self.flux_conv = pb.flux_conv.copy()
        self.flux_diff = pb.flux_diff.copy()
        self.t = pb.t.copy()
        self.E = pb.E.copy()

    def _init_bulles(self, markers=None):
        if markers is None:
            return Bulles(markers=markers, phy_prop=self.phy_prop)
        elif isinstance(markers, Bulles):
            return markers.copy()
        else:
            print(markers)
            raise NotImplementedError

    @property
    def full_name(self):
        return "%s, %s" % (self.name, self.char)

    @property
    def name(self):
        if self._imposed_name is None:
            return self.name_cas
        else:
            return self._imposed_name

    @property
    def name_cas(self):
        return "TOF"

    @property
    def char(self):
        if self.phy_prop.v == 0.0:
            return "%s, %s, dx = %g, dt = %.2g" % (
                self.num_prop.time_scheme,
                self.num_prop.schema,
                self.num_prop.dx,
                self.dt,
            )
        elif self.phy_prop.diff == 0.0:
            return "%s, %s, dx = %g, cfl = %g" % (
                self.num_prop.time_scheme,
                self.num_prop.schema,
                self.num_prop.dx,
                self.cfl,
            )
        else:
            return "%s, %s, dx = %g, dt = %.2g, cfl = %g" % (
                self.num_prop.time_scheme,
                self.num_prop.schema,
                self.num_prop.dx,
                self.dt,
                self.cfl,
            )

    @property
    def cfl(self):
        return self.phy_prop.v * self.dt / self.num_prop.dx

    @property
    def Lda_h(self):
        return 1.0 / (self.I / self.phy_prop.lda1 + (1.0 - self.I) / self.phy_prop.lda2)

    @property
    def rho_cp_a(self):
        return self.I * self.phy_prop.rho_cp1 + (1.0 - self.I) * self.phy_prop.rho_cp2

    @property
    def rho_cp_f(self):
        return self.If * self.phy_prop.rho_cp1 + (1.0 - self.If) * self.phy_prop.rho_cp2

    @property
    def rho_cp_h(self):
        return 1.0 / (
            self.I / self.phy_prop.rho_cp1 + (1.0 - self.I) / self.phy_prop.rho_cp2
        )

    def update_I(self):
        i = self.bulles.indicatrice_liquide(self.num_prop.x)
        return i

    def update_If(self):
        i_f = self.bulles.indicatrice_liquide(self.num_prop.x_f)
        return i_f

    def get_time(self):
        # nombre CFL = 1. par défaut
        if self.phy_prop.v > 10 ** (-12):
            dt_cfl = self.num_prop.dx / self.phy_prop.v * self.num_prop.cfl_lim
        else:
            dt_cfl = 10**15
        # nombre de fourier = 1. par défaut
        dt_fo = (
            self.num_prop.dx**2
            / max(self.phy_prop.lda1, self.phy_prop.lda2)
            * min(self.phy_prop.rho_cp2, self.phy_prop.rho_cp1)
            * self.num_prop.fo_lim
        )
        # dt_fo = dx**2/max(lda1/rho_cp1, lda2/rho_cp2)*fo
        # minimum des 3
        list_dt = [self.num_prop.dt_min, dt_cfl, dt_fo]
        i_dt = np.argmin(list_dt)
        temps = ["dt min", "dt cfl", "dt fourier"][i_dt]
        dt = list_dt[i_dt]
        print(temps)
        print(dt)
        return dt

    @property
    def energy(self):
        return np.sum(self.rho_cp_a * self.T * self.phy_prop.dS * self.num_prop.dx)

    @property
    def energy_m(self):
        return np.sum(self.rho_cp_a * self.T * self.num_prop.dx) / self.phy_prop.Delta

    def update_markers(self):
        self.bulles.shift(self.phy_prop.v * self.dt)
        self.I = self.update_I()
        self.If = self.update_If()

    def timestep(
        self,
        n=None,
        t_fin=None,
        plot_for_each=1,
        number_of_plots=None,
        plotter=None,
        debug=None,
        **kwargs
    ):
        if plotter is None:
            raise (Exception("plotter is a mandatory argument"))
        if (n is None) and (t_fin is None):
            raise NotImplementedError
        elif (n is not None) and (t_fin is not None):
            n = min(n, int(t_fin / self.dt))
        elif t_fin is not None:
            n = int(t_fin / self.dt)
        if number_of_plots is not None:
            plot_for_each = int((n - 1) / number_of_plots)
        if plot_for_each <= 0:
            plot_for_each = 1
        # if isinstance(plotter, list):
        #     for plott in plotter:
        #         plott.plot(self)
        # else:
        #     plotter.plot(self)
        if self.E is None:
            offset = 0
            self.E = np.zeros((n + 1,))
            self.t = np.linspace(0, n * self.dt, n + 1).copy()
            self.E[0] = self.energy
        else:
            offset = self.E.size - 1
            self.E = np.r_[self.E, np.zeros((n,))]
            self.t = np.r_[
                self.t, np.linspace(self.time + self.dt, self.time + n * self.dt, n)
            ]
        for i in range(n):
            if self.num_prop.time_scheme == "euler":
                self._euler_timestep(debug=debug, bool_debug=(i % plot_for_each == 0))
            elif self.num_prop.time_scheme == "rk4":
                self._rk4_timestep(debug=debug, bool_debug=(i % plot_for_each == 0))
            elif self.num_prop.time_scheme == "rk3":
                self._rk3_timestep(debug=debug, bool_debug=(i % plot_for_each == 0))
            self.update_markers()
            self.time += self.dt
            self.iter += 1
            self.E[offset + i + 1] = self.energy
            # intermediary plots
            if (i % plot_for_each == 0) and (i != 0) and (i != n - 1):
                if isinstance(plotter, list):
                    for plott in plotter:
                        plott.plot(self, **kwargs)
                else:
                    plotter.plot(self, **kwargs)

        # final plot
        if isinstance(plotter, list):
            for plott in plotter:
                plott.plot(self, **kwargs)
        else:
            plotter.plot(self, **kwargs)
        return self.t, self.E

    def _echange_flux(self):
        """
        Cette méthode permet de forcer que le flux sortant soit bien égal au flux entrant

        Returns:

        """
        self.flux_conv.perio()
        self.flux_diff.perio()

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Cette méthode sert à corriger dans les versions discontinues de Problem directement les flux et ainsi de
        garantir la conservation.

        Args:
            flux_conv:
            flux_diff:
            coeff_diff:

        Returns:
            Corrige les flux et coeff sur place
        """
        pass

    def _compute_convection_flux(self, T, bulles, bool_debug=False, debug=None):
        indic = bulles.indicatrice_liquide(self.num_prop.x)
        T_u = interpolate(T, I=indic, schema=self.num_prop.schema) * self.phy_prop.v
        return Flux(T_u)

    def _compute_diffusion_flux(self, T, bulles, bool_debug=False, debug=None):
        indic = bulles.indicatrice_liquide(self.num_prop.x)

        Lda_h = 1.0 / (indic / self.phy_prop.lda1 + (1.0 - indic) / self.phy_prop.lda2)
        lda_grad_T = interpolate(Lda_h, I=indic, schema="center_h") * grad(
            T, self.num_prop.dx
        )

        if (debug is not None) and bool_debug:
            # debug.set_title('sous-pas de temps %f' % (len(K) - 2))
            debug.plot(
                self.num_prop.x_f,
                lda_grad_T,
                label="lda_h grad T, time = %f" % self.time,
            )
            debug.plot(
                self.num_prop.x_f, lda_grad_T, label="lda_grad_T, time = %f" % self.time
            )
            debug.set_xticks(self.num_prop.x_f)
            debug.grid(b=True, which="major")
            debug.legend()
        return Flux(lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        self.flux_conv = self._compute_convection_flux(
            self.T, self.bulles, bool_debug, debug
        )
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )
        rho_cp_inv_h = 1.0 / self.rho_cp_h
        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff
        )
        self._echange_flux()
        dTdt = -integrale_vol_div(
            self.flux_conv, dx
        ) + self.phy_prop.diff * rho_cp_inv_h * integrale_vol_div(self.flux_diff, dx)
        self.T += self.dt * dTdt

    def _rk3_timestep(self, debug=None, bool_debug=False):
        T_int = self.T.copy()
        markers_int = self.bulles.copy()
        K = 0.0
        coeff_h = np.array([1.0 / 3, 5.0 / 12, 1.0 / 4])
        coeff_dTdtm1 = np.array([0.0, -5.0 / 9, -153.0 / 128])
        coeff_dTdt = np.array([1.0, 4.0 / 9, 15.0 / 32])
        for step, h in enumerate(coeff_h):
            # convection, conduction, dTdt = self.compute_dT_dt(T_int, markers_int, bool_debug, debug)
            convection = self._compute_convection_flux(
                T_int, markers_int, bool_debug, debug
            )
            conduction = self._compute_diffusion_flux(
                T_int, markers_int, bool_debug, debug
            )
            # TODO: vérifier qu'il ne faudrait pas plutôt utiliser rho_cp^{n,k}
            rho_cp_inv_h = 1.0 / self.rho_cp_h
            self._corrige_flux_coeff_interface(
                T_int, markers_int, convection, conduction
            )
            convection[-1] = convection[0]
            conduction[-1] = conduction[0]
            dTdt = -integrale_vol_div(
                convection, self.num_prop.dx
            ) + self.phy_prop.diff * rho_cp_inv_h * integrale_vol_div(
                conduction, self.num_prop.dx
            )
            K = K * coeff_dTdtm1[step] + dTdt
            if bool_debug and (debug is not None):
                print("step : ", step)
                print("dTdt : ", dTdt)
                print("K    : ", K)
            T_int += h * self.dt * K / coeff_dTdt[step]  # coeff_dTdt est calculé de
            # sorte à ce que le coefficient total devant les dérviées vale 1.
            # convection_l.append(convection)
            # conduction_l.append(conduction)
            markers_int.shift(self.phy_prop.v * h * self.dt)
        # coeff = np.array([1./6, 3./10, 8/15.])
        self.T = T_int

    def _rk4_timestep(self, debug=None, bool_debug=False):
        K = [0.0]
        T_u_l = []
        lda_gradT_l = []
        pas_de_temps = np.array([0.0, 0.5, 0.5, 1.0])
        dx = self.num_prop.dx
        for h in pas_de_temps:
            markers_int = self.bulles.copy()
            markers_int.shift(self.phy_prop.v * h * self.dt)
            T = self.T + h * self.dt * K[-1]
            convection = self._compute_convection_flux(
                T, markers_int, bool_debug, debug
            )
            conduction = self._compute_diffusion_flux(T, markers_int, bool_debug, debug)
            # TODO: vérifier qu'il ne faudrait pas plutôt utiliser rho_cp^{n,k}
            rho_cp_inv_h = 1.0 / self.rho_cp_h
            self._corrige_flux_coeff_interface(T, markers_int, convection, conduction)
            convection.perio()
            conduction.perio()
            T_u_l.append(convection)
            lda_gradT_l.append(conduction)
            K.append(
                -integrale_vol_div(convection, dx)
                + self.phy_prop.diff * rho_cp_inv_h * integrale_vol_div(conduction, dx)
            )
        coeff = np.array([1.0 / 6, 1 / 3.0, 1 / 3.0, 1.0 / 6])
        self.flux_conv = np.sum(coeff * Flux(T_u_l).T, axis=-1)
        self.flux_diff = np.sum(coeff * Flux(lda_gradT_l).T, axis=-1)
        self.T += np.sum(self.dt * coeff * np.array(K[1:]).T, axis=-1)

    def load_or_compute(
        self,
        pb_name=None,
        t_fin=0.0,
        n=None,
        number_of_plots=1,
        plotter=None,
        debug=None,
        **kwargs
    ):
        if pb_name is None:
            pb_name = self.full_name

        simu_name = SimuName(pb_name)
        closer_simu = simu_name.get_closer_simu(self.time + t_fin)

        if closer_simu is not None:
            with open(closer_simu, "rb") as f:
                saved = pickle.load(f)
            self.copy(saved)
            launch_time = t_fin - self.time
            print(
                "Loading ======> %s\nremaining time to compute : %f"
                % (closer_simu, launch_time)
            )
        else:
            launch_time = t_fin - self.time

        t, E = self.timestep(
            t_fin=launch_time,
            n=n,
            number_of_plots=number_of_plots,
            plotter=plotter,
            debug=debug,
            **kwargs
        )

        save_name = simu_name.get_save_path(self.time)
        with open(save_name, "wb") as f:
            pickle.dump(self, f)

        return t, E


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


class SimuName:
    def __init__(self, name: str, directory=None):
        self._name = name
        if directory is None:
            self.directory = "../References"
        else:
            self.directory = directory

    @property
    def name(self):
        return self._name

    def get_closer_simu(self, t: float):
        simu_list = glob(self.directory + "/" + self.name + "_t_" + "*" + ".pkl")
        print("Liste des simus similaires : ")
        print(simu_list)
        closer_time = 0.0
        closer_simu = None
        for simu in simu_list:
            time = self._get_time(simu)
            if (time <= t) & (time > closer_time):
                closer_time = time
                closer_simu = simu
        remaining_running_time = t - closer_time
        assert remaining_running_time >= 0.0
        return closer_simu  # , remaining_running_time

    @staticmethod
    def _get_time(path_to_save_file: str) -> float:
        time = path_to_save_file.split("_t_")[-1].split(".pkl")[0]
        return round(float(time), 6)

    def get_save_path(self, t) -> str:
        return self.directory + "/" + self.name + "_t_%f" % round(t, 6) + ".pkl"


class Flux(np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        obj = np.asarray(input_array, *args, **kwargs).view(cls)
        return obj

    def perio(self):
        self[-1] = self[0]
