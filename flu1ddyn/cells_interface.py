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
"""
This module defines the different InterfaceInterpolation classes.
"""

import numpy as np
from abc import ABC, abstractmethod

from numba import float64  # import the types

# @jitclass([('Td', float64[:]), ('Tg', float64[:]), ('_rhocp_f', float64[:]), ('_lda_f', float64[:]), ('_Ti', float64),
#            ('_lda_gradTi', float64), ('ldag', float64), ('ldad', float64), ('rhocpg', float64), ('rhocpd', float64),
#            ('ag', float64), ('ad', float64), ('dx', float64), ('vdt', float64), ('Tgc', float64), ('Tdc', float64),
#            ('_dTdxg', float64), ('_dTdxd', float64), ('_d2Tdx2g', float64), ('_d2Tdx2d', float64),
#            ('_d3Tdx3g', float64), ('_d3Tdx3d', float64), ('_T_f', float64[:]), ('_gradT_f', float64[:]),
#            ('_T', float64[:]), ('time_integral', float64)])
class InterfaceCellsBase:
    """
    Classe abstraite de base pour définir les quantités présentes dans les cellules à proximité d'une interface.
    Cette classe est ensuite surclassée selon les méthodes d'interpolation qui sont réalisées dans les différents cas.
    """

    def __init__(
        self,
        dx=1.0,
    ):
        self.ldad = 1.0
        self.ldag = 1.0
        self.ag = 0.0
        self.ad = 1.0
        self.dx = dx
        self._T = np.zeros(7)  # dans la suite, _T sera une référence
        self.Tg = np.zeros(4)
        self.Tg[:] = np.nan
        self.Td = np.zeros(4)
        self.Td[:] = np.nan
        self._Ti = -1.0
        self._lda_gradTi = 0.0
        self._d2Tdx2g = 0.0
        self._d2Tdx2d = 0.0
        self._d3Tdx3g = 0.0
        self._d3Tdx3d = 0.0
        self.Tgc = -1.0
        self.Tdc = -1.0

    @staticmethod
    def pid_interp(T: float64[:], d: float64[:]) -> float:
        TH = 10**-15
        inf_lim = d < TH
        Tm = np.sum(T / d) / np.sum(1.0 / d)
        if np.any(inf_lim):
            Tm = T[inf_lim][0]
        return Tm

    def _T_dlg(self, x: float) -> float:
        xi = self.ag * self.dx
        return (
            self.Ti
            + (x - xi) * self.dTdxg
            + (x - xi) ** 2 / 2.0 * self._d2Tdx2g
            + (x - xi) ** 3 / 6.0 * self._d3Tdx3g
        )

    def _T_dld(self, x: float) -> float:
        xi = self.ag * self.dx
        return (
            self.Ti
            + (x - xi) * self.dTdxd
            + (x - xi) ** 2 / 2.0 * self._d2Tdx2d
            + (x - xi) ** 3 / 6.0 * self._d3Tdx3d
        )

    @property
    def gradTg(self) -> np.ndarray((3,), dtype=float):
        return (self.Tg[1:] - self.Tg[:-1]) / self.dx

    @property
    def gradTd(self) -> np.ndarray((3,), dtype=float):
        return (self.Td[1:] - self.Td[:-1]) / self.dx

    @property
    def Ti(self) -> float:
        """
        Returns:
            La température interfaciale calculée à l'initialisation
        """
        return self._Ti

    @property
    def lda_gradTi(self) -> float:
        return self._lda_gradTi

    @property
    def dTdxg(self) -> float:
        return self._lda_gradTi / self.ldag

    @property
    def dTdxd(self) -> float:
        return self._lda_gradTi / self.ldad


class InterfaceInterpolationBase(InterfaceCellsBase, ABC):
    def __init__(self, *args, volume_integration=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.volume_integration = volume_integration
        self._interpolation_name = "InterfaceInterpolationBase"

    @property
    def name(self):
        name = self._interpolation_name
        if self.volume_integration:
            name += "_vol"
        return name

    def interpolate(self, T, ldag, ldad, ag):
        self.ldag = ldag
        self.ldad = ldad
        self.ag = ag
        self.ad = 1.0 - ag
        if len(T) < 7:
            raise (Exception("T n est pas de taille 7"))
        self._T = T  # ceci n'est pas une copie mais une référence !
        self.Tg = T[:4].copy()
        self.Tg[-1] = np.nan
        self.Td = T[3:].copy()
        self.Td[0] = np.nan

        self._compute_Ti()
        self._compute_ghosts()

    def _compute_Ti(self):
        if self.volume_integration:
            self._compute_Ti_with_volume_integration()
        else:
            self._compute_Ti_without_volume_integration()

    def _compute_Ti_without_volume_integration(self):
        raise NotImplemented

    def _compute_Ti_with_volume_integration(self):
        raise NotImplemented

    @abstractmethod
    def _compute_ghosts(self):
        raise NotImplemented

    def _get_T_i_and_lda_grad_T_i(
        self, Tg: float, Td: float, dg: float, dd: float
    ) -> (float, float):
        """
        On utilise la continuité de lad_grad_T pour interpoler linéairement à partir des valeurs en im32 et ip32
        On retourne les gradients suivants ::

                                 dg              dd
                            |-----------|-------------------|
                    +---------------+---------------+---------------+
                    |               |   |           |               |
                   -|>      +      -|>  |   +      -|>      +      -|>
                    |               |   |           |               |
                    +---------------+---------------+---------------+

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        T_i = (self.ldag / dg * Tg + self.ldad / dd * Td) / (
            self.ldag / dg + self.ldad / dd
        )
        lda_grad_T_ig = self.ldag * (T_i - Tg) / dg
        lda_grad_T_id = self.ldad * (Td - T_i) / dd
        return T_i, (lda_grad_T_ig + lda_grad_T_id) / 2.0


class InterfaceInterpolation1_1(InterfaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interpolation_name = "Ti-gradT"

    def _compute_Ti_without_volume_integration(self):
        self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(
            self.Tg[-2],
            self.Td[1],
            (1.0 / 2 + self.ag) * self.dx,
            (1.0 / 2 + self.ad) * self.dx,
        )

    def _compute_Ti_with_volume_integration(self):
        raise NotImplementedError

    def _compute_ghosts(self):
        grad_Tg = self.pid_interp(
            np.array([self.gradTg[1], self._lda_gradTi / self.ldag]),
            np.array([1.0, self.ag]) * self.dx,
        )
        grad_Td = self.pid_interp(
            np.array([self._lda_gradTi / self.ldad, self.gradTd[1]]),
            np.array([self.ad, 1.0]) * self.dx,
        )
        self.Tg[-1] = self.Tg[-2] + grad_Tg * self.dx
        self.Td[0] = self.Td[1] - grad_Td * self.dx


class InterfaceInterpolation1_0(InterfaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interpolation_name = "Ti-pure"

    def _compute_Ti_with_volume_integration(self):
        raise NotImplementedError

    def _compute_Ti_without_volume_integration(self):
        self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(
            self.Tg[-2],
            self.Td[1],
            (0.5 + self.ag) * self.dx,
            (0.5 + self.ad) * self.dx,
        )

    def _compute_ghosts(self):
        self.Tg[-1] = self._Ti + self.dTdxg * self.dx * (0.5 - self.ag)
        self.Td[0] = self._Ti + self.dTdxd * self.dx * (self.ad - 0.5)


class InterfaceInterpolation2(InterfaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interpolation_name = "Ti2"

    def _compute_Ti_without_volume_integration(self):
        (
            self._Ti,
            self._lda_gradTi,
            self._d2Tdx2g,
            self._d2Tdx2d,
        ) = self._get_T_i_and_lda_grad_T_i2(
            self.Tg[-2],
            self.Td[1],
            self.Tg[-3],
            self.Td[2],
            (1.0 / 2 + self.ag) * self.dx,
            (1.0 / 2 + self.ad) * self.dx,
        )

    def _compute_Ti_with_volume_integration(self):
        (
            self._Ti,
            self._lda_gradTi,
            self._d2Tdx2g,
            self._d2Tdx2d,
        ) = self._get_T_i_and_lda_grad_T_i2_vol(
            self.Tg[-2],
            self.Td[1],
            self.Tg[-3],
            self.Td[2],
            (1.0 / 2 + self.ag) * self.dx,
            (1.0 / 2 + self.ad) * self.dx,
        )

    def _compute_ghosts(self):
        self.Tg[-1] = self._T_dlg(0.5 * self.dx)
        self.Td[0] = self._T_dld(0.5 * self.dx)

    def _get_T_i_and_lda_grad_T_i2(
        self,
        Tg: float,
        Td: float,
        Tgg: float,
        Tdd: float,
        dg: float,
        dd: float,
    ) -> (float, float, float, float):
        """
        On utilise la continuité de lad_grad_T pour interpoler linéairement à partir des valeurs en im32 et ip32
        On retourne les gradients suivants ::

                                          dgg    dg              dd      ddd
                            |---------------|-----------|-------------------|---------------|
                    +---------------+---------------+---------------+---------------+---------------+
                    |               |               |   |           |               |               |
                   -|>      +      -|>      +      -|>  |   +      -|>      +      -|>      +      -|>
                    |               |               |   |           |               |               |
                    +---------------+---------------+---------------+---------------+---------------+

        Returns:
            Calcule Ti, lda_grad_T ainsi que les dérivées d'ordre 2 g et d à l'interface
        """
        dgg = dg + self.dx
        ddd = dd + self.dx
        mat = np.array(
            [
                [1.0, -dg, dg**2 / 2.0, 0.0],
                [1.0, -dgg, dgg**2 / 2.0, 0.0],
                [1.0, dd * self.ldag / self.ldad, 0.0, dd**2 / 2.0],
                [1.0, ddd * self.ldag / self.ldad, 0.0, ddd**2 / 2.0],
            ],
            dtype=np.float_,
        )
        temp = np.array([Tg, Tgg, Td, Tdd], dtype=np.float_).T
        inv_mat = np.linalg.inv(mat)
        T_i, dTdxg, d2Tdx2g, d2Tdx2d = np.dot(inv_mat, temp)
        lda_grad_T = self.ldag * dTdxg
        return T_i, lda_grad_T, d2Tdx2g, d2Tdx2d

    def _get_T_i_and_lda_grad_T_i2_vol(
        self,
        Tg: float,
        Td: float,
        Tgg: float,
        Tdd: float,
        dg: float,
        dd: float,
    ) -> (float, float, float, float):
        """
        On utilise la continuité de lad_grad_T pour interpoler linéairement à partir des valeurs en im32 et ip32
        On retourne les gradients suivants ::

                                          dgg    dg              dd      ddd
                            |---------------|-----------|-------------------|---------------|
                    +---------------+---------------+---------------+---------------+---------------+
                    |               |               |   |           |               |               |
                   -|>      +      -|>      +      -|>  |   +      -|>      +      -|>      +      -|>
                    |               |               |   |           |               |               |
                    +---------------+---------------+---------------+---------------+---------------+

        Returns:
            Calcule Ti, lda_grad_T ainsi que les dérivées d'ordre 2 g et d à l'interface
        """
        dgg = dg + self.dx
        ddd = dd + self.dx
        mat = np.array(
            [
                [1.0, -dg, dg**2 * 2.0 / 3.0, 0.0],
                [1.0, -dgg, dgg**2 * 2.0 / 3.0, 0.0],
                [1.0, dd * self.ldag / self.ldad, 0.0, dd**2 * 2.0 / 3.0],
                [1.0, ddd * self.ldag / self.ldad, 0.0, ddd**2 * 2.0 / 3.0],
            ],
            dtype=np.float_,
        )
        temp = np.array([Tg, Tgg, Td, Tdd], dtype=np.float_).T
        inv_mat = np.linalg.inv(mat)
        T_i, dTdxg, d2Tdx2g, d2Tdx2d = np.dot(inv_mat, temp)
        lda_grad_T = self.ldag * dTdxg
        return T_i, lda_grad_T, d2Tdx2g, d2Tdx2d


class InterfaceInterpolation3(InterfaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interpolation_name = "Ti3"

    def _compute_ghosts(self):
        self.Tg[-1] = self._T_dlg(0.5 * self.dx)
        self.Td[0] = self._T_dld(0.5 * self.dx)

    def _compute_Ti_without_volume_integration(self):
        (
            self._Ti,
            self._lda_gradTi,
            self._d2Tdx2g,
            self._d2Tdx2d,
            self._d3Tdx3g,
            self._d3Tdx3d,
        ) = self._get_T_i_and_lda_grad_T_i3(
            self.Tg[-2],
            self.Td[1],
            self.Tg[-3],
            self.Td[2],
            self.Tg[-4],
            self.Td[3],
            (1.0 / 2 + self.ag) * self.dx,
            (1.0 / 2 + self.ad) * self.dx,
        )

    def _get_T_i_and_lda_grad_T_i3(
        self,
        Tg: float,
        Td: float,
        Tgg: float,
        Tdd: float,
        Tggg: float,
        Tddd: float,
        dg: float,
        dd: float,
    ) -> (float, float, float, float, float, float):
        """
        On utilise la continuité de lad_grad_T et on écrit un DL à l'ordre 3 à droite et à gauche.
        On retourne les gradients suivants ::

                                           dggg   dgg    dg              dd      ddd        dddd
                    |---------------|---------------|-----------|-------------------|---------------|--------------|
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+
            |               |               |               |   |           |               |               |              |
            |>      +      -|>      +      -|>      +      -|>  |   +      -|>      +      -|>      +      -|>     +      -|>
            |               |               |               |   |           |               |               |              |
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+

        Returns:
            Calcule Ti, lda_grad_T ainsi que les dérivées d'ordre 2 g et d à l'interface
        """
        dgg = dg + self.dx
        dggg = dg + 2.0 * self.dx
        ddd = dd + self.dx
        dddd = dd + 2.0 * self.dx
        mat = np.array(
            [
                [1.0, -dg, dg**2 / 2.0, 0.0, -(dg**3) / 6, 0.0],
                [1.0, -dgg, dgg**2 / 2.0, 0.0, -(dgg**3) / 6, 0.0],
                [1.0, -dggg, dggg**2 / 2.0, 0.0, -(dggg**3) / 6, 0.0],
                [
                    1.0,
                    dd * self.ldag / self.ldad,
                    0.0,
                    dd**2 / 2.0,
                    0.0,
                    dd**3 / 6.0,
                ],
                [
                    1.0,
                    ddd * self.ldag / self.ldad,
                    0.0,
                    ddd**2 / 2.0,
                    0.0,
                    ddd**3 / 6.0,
                ],
                [
                    1.0,
                    dddd * self.ldag / self.ldad,
                    0.0,
                    dddd**2 / 2.0,
                    0.0,
                    dddd**3 / 6.0,
                ],
            ],
            dtype=np.float_,
        )
        temp = np.array([Tg, Tgg, Tggg, Td, Tdd, Tddd], dtype=np.float_).T
        inv_mat = np.linalg.inv(mat)
        T_i, dTdxg, d2Tdx2g, d2Tdx2d, d3Tdx3g, d3Tdx3d = np.dot(inv_mat, temp)
        lda_grad_T = self.ldag * dTdxg
        return T_i, lda_grad_T, d2Tdx2g, d2Tdx2d, d3Tdx3g, d3Tdx3d

    def _compute_Ti_with_volume_integration(self):
        (
            self._Ti,
            self._lda_gradTi,
            self._d2Tdx2g,
            self._d2Tdx2d,
            self._d3Tdx3g,
            self._d3Tdx3d,
        ) = self._get_T_i_and_lda_grad_T_i3_vol(
            self.Tg[-2],
            self.Td[1],
            self.Tg[-3],
            self.Td[2],
            self.Tg[-4],
            self.Td[3],
            (1.0 / 2 + self.ag) * self.dx,
            (1.0 / 2 + self.ad) * self.dx,
        )

    def _get_T_i_and_lda_grad_T_i3_vol(
        self,
        Tg: float,
        Td: float,
        Tgg: float,
        Tdd: float,
        Tggg: float,
        Tddd: float,
        dg: float,
        dd: float,
    ) -> (float, float, float, float, float, float):
        """
        On utilise la continuité de lad_grad_T et on écrit un DL à l'ordre 3 à droite et à gauche.
        On retourne les gradients suivants ::

                                           dggg   dgg    dg              dd      ddd        dddd
                    |---------------|---------------|-----------|-------------------|---------------|--------------|
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+
            |               |               |               |   |           |               |               |              |
            |>      +      -|>      +      -|>      +      -|>  |   +      -|>      +      -|>      +      -|>     +      -|>
            |               |               |               |   |           |               |               |              |
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+

        Returns:
            Calcule Ti, lda_grad_T ainsi que les dérivées d'ordre 2 g et d à l'interface
        """
        dgg = dg + self.dx
        dggg = dg + 2.0 * self.dx
        ddd = dd + self.dx
        dddd = dd + 2.0 * self.dx
        mat = np.array(
            [
                [
                    1.0,
                    -dg,
                    dg**2 * 2.0 / 3.0,
                    0.0,
                    -(dg**3) * 1.0 / 3.0,
                    0.0,
                ],
                [
                    1.0,
                    -dgg,
                    dgg**2 * 2.0 / 3.0,
                    0.0,
                    -(dgg**3) * 1.0 / 3.0,
                    0.0,
                ],
                [
                    1.0,
                    -dggg,
                    dggg**2 * 2.0 / 3.0,
                    0.0,
                    -(dggg**3) * 1.0 / 3.0,
                    0.0,
                ],
                [
                    1.0,
                    dd * self.ldag / self.ldad,
                    0.0,
                    dd**2 * 2.0 / 3.0,
                    0.0,
                    dd**3 * 1.0 / 3.0,
                ],
                [
                    1.0,
                    ddd * self.ldag / self.ldad,
                    0.0,
                    ddd**2 * 2.0 / 3.0,
                    0.0,
                    ddd**3 * 1.0 / 3.0,
                ],
                [
                    1.0,
                    dddd * self.ldag / self.ldad,
                    0.0,
                    dddd**2 * 2.0 / 3.0,
                    0.0,
                    dddd**3 * 1.0 / 3.0,
                ],
            ],
            dtype=np.float_,
        )
        temp = np.array([Tg, Tgg, Tggg, Td, Tdd, Tddd], dtype=np.float_).T
        inv_mat = np.linalg.inv(mat)
        T_i, dTdxg, d2Tdx2g, d2Tdx2d, d3Tdx3g, d3Tdx3d = np.dot(inv_mat, temp)
        lda_grad_T = self.ldag * dTdxg
        return T_i, lda_grad_T, d2Tdx2g, d2Tdx2d, d3Tdx3g, d3Tdx3d

    # TODO: quelle différence avec la méthode précédente ?
    def _get_T_i_and_lda_grad_T_i3_1_vol(
        self,
        Tggg: float,
        Tgg: float,
        Tg: float,
        Td: float,
        dg: float,
        dd: float,
    ) -> (float, float, float, float):
        """
        On utilise la continuité de lad_grad_T et on écrit un DL à l'ordre 3 à droite et à gauche.
        On retourne les gradients suivants ::

                                           dggg   dgg    dg              dd      ddd        dddd
                    |---------------|---------------|-----------|-------------------|---------------|--------------|
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+
            |               |               |               |   |           |               |               |              |
            |>      +      -|>      +      -|>      +      -|>  |   +      -|>      +      -|>      +      -|>     +      -|>
            |               |               |               |   |           |               |               |              |
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+

        Returns:
            Calcule Ti, lda_grad_T ainsi que les dérivées d'ordre 2 g et d à l'interface
        """
        dgg = dg + self.dx
        dggg = dg + 2.0 * self.dx
        mat = np.array(
            [
                [1.0, -dg, dg**2 * 2.0 / 3.0, 0.0],
                [1.0, -dgg, dgg**2 * 2.0 / 3.0, 0.0],
                [1.0, -dggg, dggg**2 * 2.0 / 3.0, 0.0],
                [1.0, dd * self.ldag / self.ldad, 0.0, dd**2 * 2.0 / 3.0],
            ],
            dtype=np.float_,
        )
        temp = np.array([Tg, Tgg, Tggg, Td], dtype=np.float_).T
        inv_mat = np.linalg.inv(mat)
        T_i, dTdxg, d2Tdx2g, d2Tdx2d = np.dot(inv_mat, temp)
        lda_grad_T = self.ldag * dTdxg
        return T_i, lda_grad_T, d2Tdx2g, d2Tdx2d


class InterfaceInterpolationIntegral(InterfaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interpolation_name = "Integral"

    def _compute_Ti(self):
        (
            self._Ti,
            self._lda_gradTi,
        ) = self._get_T_i_and_lda_grad_T_i_from_integrated_temperature(
            self._T[2], self._T[3], self._T[4], self.ag, self.dx
        )

    def _compute_ghosts(self):
        grad_Tg = self.pid_interp(
            np.array([self.gradTg[1], self.dTdxg]),
            np.array([1.0, self.ag]) * self.dx,
        )
        grad_Td = self.pid_interp(
            np.array([self.dTdxd, self.gradTd[1]]),
            np.array([self.ad, 1.0]) * self.dx,
        )
        self.Tg[-1] = self.Tg[-2] + grad_Tg * self.dx
        self.Td[0] = self.Td[1] - grad_Td * self.dx

    def _get_T_i_and_lda_grad_T_i_from_integrated_temperature(
        self, Tg: float, Tc: float, Td: float, ag: float, dx: float
    ) -> (float, float):
        if ag > 0.5:
            xi = ag * dx
            xg = ag / 2.0 * dx
            xc = (ag + (1.0 - ag) / 2) * dx
            xd = 3.0 / 2.0 * dx
            Ig = ag
            Id = 1.0 - ag
            a = np.array(
                [
                    [
                        1.0,
                        1.0 / self.ldag * Ig * (xg - xi)
                        + 1.0 / self.ldad * Id * (xc - xi),
                    ],
                    [1.0, 1.0 / self.ldad * (xd - xi)],
                ]
            )
            b = np.array([[Tc], [Td]])
        else:
            xi = ag * dx
            xg = -1.0 / 2.0 * dx
            xc = ag / 2.0 * dx
            xd = (ag + (1.0 - ag) / 2) * dx
            Ig = ag
            Id = 1.0 - ag
            a = np.array(
                [
                    [
                        1.0,
                        1.0 / self.ldag * Ig * (xc - xi)
                        + 1.0 / self.ldad * Id * (xd - xi),
                    ],
                    [1.0, 1.0 / self.ldag * (xg - xi)],
                ]
            )
            b = np.array([[Tc], [Tg]])
        Ti, qi = np.dot(np.linalg.inv(a), b)
        return Ti, qi


class InterfaceInterpolationEnergieTemperature(InterfaceInterpolationBase):
    """
    Cellule type ::

                                         Ti,
                                         lda_gradTi
                                         Ti0g
                                         Ti0d
                                    Tgf Tgc Tdc Tdf
                +----------+---------+---------+---------+---------+
                |          |         | gc|  dc |         |         |
                |    +    -|>   +   -|>* | +* -|>   +   -|>   +    |
                |          |         |   |     |         |         |
                +----------+---------+---------+---------+---------+
                        gradTi-3/2  gradTg    gradTd    gradTi+3/2

    Dans ce modèle on connait initialement les températures moyenne aux centes de toutes les cellules.
    On reconstruit les valeurs de Tgc et Tdc avec le système sur la valeur moyenne de température dans la maille et
    la valeur moyenne de l'énergie.
    Ensuite évidemment on interpole là ou on en aura besoin.
    Il faudra faire 2 équations d'évolution dans la cellule i, une sur h et une sur T.

    Selon la method calcule les flux et les températures aux interfaces.
    Si la méthode est classique, on calcule tout en utilisant Tim1, Tgc et T_I (et de l'autre côté T_I, Tdc et Tip1)

    Args:
        h:
        T_mean:

    Returns:
        Rien mais met à jour self.grad_T et self.T_f
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rhocpg = 0.0
        self.rhocpd = 0.0
        self.h = 0.0
        self._interpolation_name = "hT"

    def interpolate(self, *args, h, rhocpg, rhocpd, **kwargs):
        self.h = h
        self.rhocpg = rhocpg
        self.rhocpd = rhocpd
        super().interpolate(*args, **kwargs)

    def _compute_Tgc_Tdc(self, h: float, T_mean: float):
        """
        Résout le système d'équation entre la température moyenne et l'énergie de la cellule pour trouver les valeurs de
        Tgc et Tdc.
        Le système peut toujours être résolu car rhocpg != rhocpd

        Returns:
            Rien mais mets à jour Tgc et Tdc
        """
        system = np.array(
            [
                [self.ag * self.rhocpg, self.ad * self.rhocpd],
                [self.ag, self.ad],
            ]
        )
        self.Tgc, self.Tdc = np.dot(np.linalg.inv(system), np.array([h, T_mean]))

    def _compute_Ti(self):
        # On commence par calculer T_I et lda_grad_Ti en fonction de Tgc et Tdc :
        EPSILON = 10**-6
        if self.ag < EPSILON:
            self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(
                self.Tg[-2],
                self._T[3],
                (1.0 / 2.0 + self.ag) * self.dx,
                self.ad / 2.0 * self.dx,
            )
        elif self.ad < EPSILON:
            self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(
                self._T[3],
                self.Td[1],
                self.ag / 2.0 * self.dx,
                (1.0 / 2.0 + self.ad) * self.dx,
            )
        else:
            self._compute_Tgc_Tdc(self.h, self._T[3])
            self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(
                self.Tgc,
                self.Tdc,
                self.ag / 2.0 * self.dx,
                self.ad / 2.0 * self.dx,
            )

    def _compute_ghosts(self):
        self.Tg[-1] = self._T_dlg(0.5 * self.dx)
        self.Td[0] = self._T_dld(0.5 * self.dx)


class InterfaceInterpolationContinuousFluxBase(InterfaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lda_gradTid = 0.0
        self.lda_gradTig = 0.0
        self.lda_gradTd = 0.0
        self.lda_gradTg = 0.0
        self._interpolation_name = "ldagradT"

    def _compute_Ti(self):
        """
        On commence par récupérer lda_grad_Ti, gradTg, gradTd par continuité à partir de gradTim32 et gradTip32.
        On en déduit Ti par continuité à gauche et à droite.

        Returns:
            Calcule les gradients g, I, d, et Ti. gradTig et gradTid sont les gradients centrés entre TI et Tim1 et TI
            et Tip1

        Warnings:
            Cette méthode est dépréciée, elle a recours à des extrapolation d'ordre 1
        """
        self._interpolate_ldagradT()
        self._Ti = self._get_Ti_from_lda_grad_Ti(
            self.Tg[-2],
            self.Td[1],
            (0.5 + self.ag) * self.dx,
            (0.5 + self.ad) * self.dx,
            self.lda_gradTig,
            self.lda_gradTid,
        )

    def _interpolate_ldagradT(self):
        raise NotImplemented

    def _compute_ghosts(self):
        raise NotImplemented

    def _get_Ti_from_lda_grad_Ti(
        self,
        Tim1: float,
        Tip1: float,
        dg: float,
        dd: float,
        lda_gradTgi: float,
        lda_gradTdi: float,
    ) -> float:
        """
        Méthode d'extrapolation d'ordre 1, ou l'on récupère la température à l'interface à partir de la température et
        du gradient du côté du gradient le plus faible.

        Warnings:
            Déprécié car d'ordre 1 et extrapolation.

        Args:
            Tim1:
            Tip1:
            dg:
            dd:
            lda_gradTgi:
            lda_gradTdi:

        Returns:

        """
        if self.ldag > self.ldad:
            Ti = Tim1 + lda_gradTgi / self.ldag * dg
        else:
            Ti = Tip1 - lda_gradTdi / self.ldad * dd
        return Ti


class InterfaceInterpolationContinuousFlux1(InterfaceInterpolationContinuousFluxBase):
    def __init__(self, *args, **kwargs):
        super(InterfaceInterpolationContinuousFlux1, self).__init__(*args, **kwargs)
        self._interpolation_name = "ldagradT1"

    def _get_lda_grad_T_i_from_ldagradT_continuity(
        self,
        Tim2: float,
        Tim1: float,
        Tip1: float,
        Tip2: float,
        dg: float,
        dd: float,
    ) -> (float, float, float, float, float):
        """
        On utilise la continuité de lad_grad_T pour interpoler linéairement à partir des valeurs en im32 et ip32
        On retourne les gradients suivants ::

                             dg                      dd
                    |-------------------|---------------------------|
                    +---------------+---------------+---------------+
                    |               |   |           |               |
                   -|>      +    o -|>  |   +     o-|>      +      -|>
                    |               |   |           |               |
                    +---------------+---------------+---------------+
                 gradTi-3/2       gradTg          gradTd            gradTip32
                                gradTgi          gradTdi

        Warnings:
            Cette méthode est non convergente dans certains cas et d'ordre assez faible. Il est relativement faux de
            considérer qu'il y a linéarité de lda grad T étant donné qu'il n'y a pas continuité de la dérivée seconde.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        ldagradTgg = self.ldag * (Tim1 - Tim2) / self.dx
        ldagradTdd = self.ldad * (Tip2 - Tip1) / self.dx
        lda_gradTg = self.pid_interp(
            np.array([ldagradTgg, ldagradTdd]),
            np.array([self.dx, 2.0 * self.dx]),
        )
        lda_gradTgi = self.pid_interp(
            np.array([ldagradTgg, ldagradTdd]),
            np.array(
                [
                    dg - (dg - 0.5 * self.dx) / 2.0,
                    dd + (dg - 0.5 * self.dx) / 2.0,
                ]
            ),
        )
        lda_gradTi = self.pid_interp(
            np.array([ldagradTgg, ldagradTdd]), np.array([dg, dd])
        )
        lda_gradTdi = self.pid_interp(
            np.array([ldagradTgg, ldagradTdd]),
            np.array(
                [
                    dg + (dd - 0.5 * self.dx) / 2.0,
                    dd - (dd - 0.5 * self.dx) / 2.0,
                ]
            ),
        )
        lda_gradTd = self.pid_interp(
            np.array([ldagradTgg, ldagradTdd]),
            np.array([2.0 * self.dx, self.dx]),
        )
        return lda_gradTg, lda_gradTgi, lda_gradTi, lda_gradTdi, lda_gradTd

    def _interpolate_ldagradT(self):
        (
            self.lda_gradTg,
            self.lda_gradTig,
            self._lda_gradTi,
            self.lda_gradTid,
            self.lda_gradTd,
        ) = self._get_lda_grad_T_i_from_ldagradT_continuity(
            self.Tg[1],
            self.Tg[2],
            self.Td[1],
            self.Td[2],
            (1.0 + self.ag) * self.dx,
            (1.0 + self.ad) * self.dx,
        )

    def _compute_ghosts(self):
        self.Tg[-1] = self.Tg[-2] + self.lda_gradTg / self.ldag * self.dx
        self.Td[0] = self.Td[1] - self.lda_gradTd / self.ldad * self.dx


class InterfaceInterpolationContinuousFlux2(InterfaceInterpolationContinuousFluxBase):
    def __init__(self, *args, **kwargs):
        super(InterfaceInterpolationContinuousFlux2, self).__init__(*args, **kwargs)
        self._interpolation_name = "ldagradT2"

    def _interpolate_ldagradT(self):
        """
        On commence par récupérer lda_grad_Ti par continuité à partir de gradTim52 gradTim32 gradTip32
        et gradTip52.
        On en déduit Ti par continuité à gauche et à droite.

        Returns:
            Calcule les gradients g, I, d, et Ti

        Warnings:
            Cette méthode est dépréciée, elle a recours à des extrapolation d'ordre 1
        """
        (
            lda_gradTg,
            lda_gradTig,
            self._lda_gradTi,
            lda_gradTid,
            lda_gradTd,
        ) = self._get_lda_grad_T_i_from_ldagradT_interp(
            self.Tg[0],
            self.Tg[1],
            self.Tg[2],
            self.Td[1],
            self.Td[2],
            self.Td[3],
            self.ag,
            self.ad,
        )

    def _compute_ghosts(self):
        self.Tg[-1] = self.Tg[-2] + self.lda_gradTig / self.ldag * self.dx
        self.Td[0] = self.Td[1] - self.lda_gradTid / self.ldad * self.dx

    def _get_lda_grad_T_i_from_ldagradT_interp(
        self,
        Tim3: float,
        Tim2: float,
        Tim1: float,
        Tip1: float,
        Tip2: float,
        Tip3: float,
        ag: float,
        ad: float,
    ) -> (float, float, float, float, float):
        """
        On utilise la continuité de lad_grad_T pour extrapoler linéairement par morceau à partir des valeurs en im52,
        im32, ip32 et ip52. On fait ensuite la moyenne des deux valeurs trouvées.
        On retourne les gradients suivants ::

                                      ag       dg
                                    |---|-----------|
                        dgi |-----o-----|---------o---------| ddi
                    +---------------+---------------+---------------+
                    |               |   |           |               |
                   -|>      +     o-|>  |   +     o-|>      +      -|>
                    |               |   |           |               |
                    +---------------+---------------+---------------+
                 gradTim32          gradTg          gradTd          gradTip32
                                 gradTgi         gradTdi

        Warnings:
            Cette méthode est non convergente dans certains cas, elle est décentrée ce qui n'est pas souvent une bonne
            idée.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        lda_gradTim52 = self.ldag * (Tim2 - Tim3) / self.dx
        lda_gradTim32 = self.ldag * (Tim1 - Tim2) / self.dx
        lda_gradTip32 = self.ldad * (Tip2 - Tip1) / self.dx
        lda_gradTip52 = self.ldad * (Tip3 - Tip2) / self.dx
        gradgradg = (lda_gradTim32 - lda_gradTim52) / self.dx
        gradgrad = (lda_gradTip52 - lda_gradTip32) / self.dx

        ldagradTig = lda_gradTim32 + gradgradg * (self.dx + ag * self.dx)
        ldagradTid = lda_gradTip32 - gradgrad * (self.dx + ad * self.dx)
        lda_gradTi = (ldagradTig + ldagradTid) / 2.0

        lda_gradTg = self.pid_interp(
            np.array([lda_gradTim32, lda_gradTi]),
            np.array([self.dx, ag * self.dx]),
        )
        lda_gradTd = self.pid_interp(
            np.array([lda_gradTi, lda_gradTip32]),
            np.array([ad * self.dx, self.dx]),
        )
        dgi = (1 / 2.0 + ag) * self.dx
        lda_gradTgi = self.pid_interp(
            np.array([lda_gradTim32, lda_gradTi]),
            np.array([self.dx / 2 + dgi / 2, dgi / 2.0]),
        )
        ddi = (1.0 / 2 + ad) * self.dx
        lda_gradTdi = self.pid_interp(
            np.array([lda_gradTi, lda_gradTip32]),
            np.array([ddi / 2.0, self.dx / 2 + ddi / 2.0]),
        )
        return lda_gradTg, lda_gradTgi, lda_gradTi, lda_gradTdi, lda_gradTd