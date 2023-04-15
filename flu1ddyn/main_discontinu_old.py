import numpy as np

from flu1ddyn.interpolation_methods import (
    Flux,
    interpolate,
    grad,
    integrale_vol_div,
    interpolate_from_center_to_face_quick,
)
from flu1ddyn.main import Problem
from flu1ddyn.main_discontinu import BulleTemperature, cl_perio, get_prop
from flu1ddyn.problem_definition import Bulles


class CellsInterface:
    schema_conv: str
    schema_diff: str
    interp_type: str

    """
    Cellule type ::

            Tg, gradTg                          Tghost
           +---------+----------+---------+---------+
           |         |          |         |   |     |
           |    +   -|>   +    -|>   +   -|>  |+    |
           |    0    |    1     |    2    |   |3    |
           +---------+----------+---------+---------+---------+---------+---------+
                                          |   |     |         |         |         |
                             Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                          |   |0    |    1    |    2    |    3    |
                                          +---------+---------+---------+---------+
                                              Ti,                              |--|
                                              lda_gradTi                         vdt

    Dans cette représentation de donnée de base, on peut stocker les valeurs de la température au centre, aux faces,
    la valeur des gradients aux faces, la valeur des lda et rhocp aux faces. On stocke aussi les valeurs à
    l'interface.

    Args:
        ldag:
        ldad:
        ag:
        dx:
        T:
        rhocpg:
        rhocpd:
    """

    def __init__(
        self,
        ldag=1.0,
        ldad=1.0,
        ag=1.0,
        dx=1.0,
        T=np.zeros((7,)),
        rhocpg=1.0,
        rhocpd=1.0,
        vdt=0.0,
        schema_conv="upwind",
        interp_type="Ti",
        schema_diff="centre",
        time_integral="exact",
    ):
        self.ldag = ldag
        self.ldad = ldad
        self.rhocpg = rhocpg
        self.rhocpd = rhocpd
        self.ag = ag
        self.ad = 1.0 - ag
        self.dx = dx
        self.time_integral = time_integral
        if len(T) < 7:
            raise (Exception("T n est pas de taille 7"))
        self._T = T[
            :
        ].copy()  # use with extra care !!! Meant only to be used with the integrated method
        self.Tg = T[:4].copy()
        self.Tg[-1] = np.nan
        self.Td = T[3:].copy()
        self.Td[0] = np.nan
        self._rhocp_f = np.array([rhocpg, rhocpg, rhocpg, np.nan, rhocpd, rhocpd])
        self._lda_f = np.array([ldag, ldag, ldag, np.nan, ldad, ldad])
        self._Ti = -1.0
        self._lda_gradTi = 0.0
        self._dTdxg = 0.0
        self._dTdxd = 0.0
        self._d2Tdx2g = 0.0
        self._d2Tdx2d = 0.0
        self._d3Tdx3g = 0.0
        self._d3Tdx3d = 0.0
        self.schema_conv = schema_conv
        self.schema_diff = schema_diff
        self.vdt = vdt
        self.interp_type = interp_type
        self.Tgc = -1.0
        self.Tdc = -1.0
        self._T_f = np.empty((6,), dtype=np.float_)
        self._gradT_f = np.empty((6,), dtype=np.float_)
        # On fait tout de suite le calcul qui nous intéresse, il est nécessaire pour la suite
        if self.schema_conv.endswith("ghost"):
            self.compute_from_Ti_ghost()
        elif self.interp_type == "Ti" or self.interp_type == "Ti_vol":
            self.compute_from_Ti()
        elif self.interp_type == "Ti2":
            self.compute_from_Ti2()
        elif self.interp_type == "Ti3":
            self.compute_from_Ti3()
        elif self.interp_type == "Ti2_vol":
            self.compute_from_Ti2_vol()
        elif self.interp_type == "Ti3_vol":
            self.compute_from_Ti3_vol()
        elif self.interp_type == "Ti3_1_vol":
            self.compute_from_Ti3_1_vol()
        elif self.interp_type == "gradTi":
            self.compute_from_ldagradTi()
        elif self.interp_type == "gradTi2":
            self.compute_from_ldagradTi_ordre2()
        elif self.interp_type == "energie_temperature":
            pass
        elif self.interp_type == "integrale":
            self.compute_from_Ti_and_ip1()
        else:
            raise NotImplementedError

        if self.interp_type.endswith("_vol"):
            self.compute_T_f_gradT_f_quick_vol()
        elif self.interp_type == "energie_temperature":
            pass
        else:
            if self.schema_conv == "weno":
                self.compute_T_f_gradT_f_quick()
            if self.schema_conv == "quick":
                self.compute_T_f_gradT_f_quick()
            if self.schema_conv == "quick_ghost":
                self.compute_T_f_gradT_f_quick_ghost()
            if self.schema_conv == "quick_upwind_ghost":
                self.compute_T_f_gradT_f_quick_upwind_ghost()
            if self.schema_conv == "upwind":
                self.compute_T_f_gradT_f_upwind()
            if self.schema_conv == "amont_centre":
                self.compute_T_f_gradT_f_amont_centre()

    @staticmethod
    def pid_interp(T: np.ndarray, d: np.ndarray) -> float:
        TH = 10**-15
        inf_lim = d < TH
        Tm = np.sum(T / d) / np.sum(1.0 / d)
        if np.any(inf_lim):
            Tm = T[inf_lim][0]
        return Tm

    @property
    def T_f(self):
        if self.schema_conv == "upwind":
            return np.concatenate((self.Tg[:-1], self.Td[:-1]))
        if self.schema_conv == "weno":
            return self._T_f
        if self.schema_conv == "quick":
            return self._T_f
        if self.schema_conv == "quick_ghost":
            return self._T_f
        if self.schema_conv == "quick_upwind_ghost":
            return self._T_f
        if self.schema_conv == "amont_centre":
            return self._T_f
        else:
            # schema centre
            raise Exception("Le schema n est pas implémenté")
            # return np.concatenate(((self.Tg[1:] + self.Tg[:-1])/2., (self.Td[1:] + self.Td[:-1])/2.))

    @property
    def gradTg(self) -> np.ndarray((3,), dtype=float):
        return (self.Tg[1:] - self.Tg[:-1]) / self.dx

    @property
    def gradTd(self) -> np.ndarray((3,), dtype=float):
        return (self.Td[1:] - self.Td[:-1]) / self.dx

    @property
    def grad_lda_gradT_n_g(self) -> float:
        # remarque : cette valeur n'est pas calculée exactement à l'interface
        # mais plutôt entre l'interface et la face 32, mais je pense pas que ce soit très grave
        # et j'ai pas le courage de faire autre chose
        d = self.dx * (1.0 + self.ag)
        lda_gradT_32 = self.ldag * (self.Tg[-2] - self.Tg[-3]) / self.dx
        return (self.lda_gradTi - lda_gradT_32) / d

    @property
    def grad_lda_gradT_n_d(self) -> float:
        # remarque idem que au dessus
        d = self.dx * (1.0 + self.ad)
        lda_gradT_32 = self.ldad * (self.Td[2] - self.Td[1]) / self.dx
        return (lda_gradT_32 - self.lda_gradTi) / d

    @property
    def gradT(self) -> np.ndarray((6,), dtype=float):
        if (
            self.schema_diff == "DL"
            or self.schema_conv == "weno"
            or self.schema_conv == "quick"
            or self.schema_conv == "quick_ghost"
            or self.schema_conv == "quick_upwind_ghost"
            or self.schema_conv == "amont_centre"
        ):
            return self._gradT_f
        else:
            return np.concatenate((self.gradTg, self.gradTd))

    @property
    def rhocp_f(self) -> np.ndarray((6,), dtype=float):
        if self.vdt > 0.0:
            if self.time_integral == "exact":
                coeff_d = min(self.vdt, self.ad * self.dx) / self.vdt
                self._rhocp_f[3] = coeff_d * self.rhocpd + (1.0 - coeff_d) * self.rhocpg
            elif self.time_integral == "CN":
                if self.ad * self.dx > self.vdt:
                    self._rhocp_f[3] = self.rhocpd
                else:
                    self._rhocp_f[3] = (self.rhocpd + self.rhocpg) / 2.0
            else:
                raise Exception(
                    "L'attribut time_integral : %s n'est pas reconnu"
                    % self.time_integral
                )
            return self._rhocp_f
        else:
            self._rhocp_f[3] = self.rhocpd
            return self._rhocp_f

    @property
    def coeff_d(self) -> float:
        if self.vdt > 0.0:
            return min(self.vdt, self.ad * self.dx) / self.vdt
        else:
            return 1.0

    @property
    def inv_rhocp_f(self) -> np.ndarray((6,), dtype=float):
        if self.vdt > 0.0:
            coeff_d = min(self.vdt, self.ad * self.dx) / self.vdt
            inv_rho_cp_f = 1.0 / self.rhocp_f
            inv_rho_cp_f[3] = (
                coeff_d * 1.0 / self.rhocpd + (1.0 - coeff_d) * 1.0 / self.rhocpg
            )
            return inv_rho_cp_f
        else:
            self._rhocp_f[3] = self.rhocpd
            return 1.0 / self._rhocp_f

    @property
    def lda_f(self) -> np.ndarray((6,), dtype=float):
        if self.vdt > 0.0:
            # coeff_d = min(self.vdt, self.ad*self.dx)/self.vdt
            # self._lda_f[3] = coeff_d * self.ldad + (1. - coeff_d) * self.ldag
            self._lda_f[3] = self.ldad
            return self._lda_f
        else:
            self._lda_f[3] = self.ldad
            return self._lda_f

    @property
    def rhocpT_f(self) -> np.ndarray((6,), dtype=float):
        if self.vdt > 0.0:
            # TODO: implémenter une méthode qui renvoie rho * cp * T intégré sur le volume qui va passer d'une cellule à
            #  l'autre. Cette précision n'est peut-être pas nécessaire
            rhocpTf = self.rhocp_f * self.T_f
            return rhocpTf
        else:
            return self.rhocp_f * self.T_f

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

    def compute_T_f_gradT_f_amont_centre(self):
        """
        Cellule type ::

                Tg, gradTg                          Tghost
                         0          1         2         3
               +---------+----------+---------+---------+
               |         |          |         |   |     |
               |    +   -|>   +    -|>   +   -|>  |+    |
               |    0    |    1     |    2    |   |3    |         4         5
               +---------+----------+---------+---------+---------+---------+---------+
                                              |   |     |         |         |         |
                                 Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                              |   |0    |    1    |    2    |    3    |
                                              +---------+---------+---------+---------+
                                                  Ti,                              |--|
                                                  lda_gradTi                         vdt

        Returns:

        """
        Tim32, _, _ = self._interp_lagrange_amont_centre(
            self.Tg[0],
            self.Tg[1],
            self.Tg[2],
            -2 * self.dx,
            -1.0 * self.dx,
            0.0 * self.dx,
            -0.5 * self.dx,
        )
        Tim12, _, _ = self._interp_lagrange_amont_centre(
            self.Tg[1],
            self.Tg[2],
            self.Ti,
            -2.0 * self.dx,
            -1.0 * self.dx,
            (self.ag - 0.5) * self.dx,
            -0.5 * self.dx,
        )
        _, dTdxim12, _ = self._interp_lagrange_centre_grad(
            self.Tg[-3],
            self.Tg[-2],
            self.Ti,
            self._dTdxg,
            -2 * self.dx,
            -1 * self.dx,
            (self.ag - 0.5) * self.dx,
            -0.5 * self.dx,
        )
        # Tip12, _, _ = self._interp_lagrange_amont_grad(self.Ti, self.Td[1], self._dTdxd,
        #                                                (0.5 - self.ad) * self.dx, 1. * self.dx,
        #                                                0.5 * self.dx)
        Tip12, _, _ = self._interp_amont_decentre(
            self.Ti, self._dTdxd, (0.5 - self.ad) * self.dx, 0.5 * self.dx
        )
        _, dTdxip12, _ = self._interp_lagrange_centre_grad(
            self.Td[2],
            self.Td[1],
            self.Ti,
            self._dTdxd,
            2.0 * self.dx,
            1.0 * self.dx,
            (0.5 - self.ad) * self.dx,
            0.5 * self.dx,
        )
        Tip32, _, _ = self._interp_lagrange_amont_centre(
            self.Ti,
            self.Td[1],
            self.Td[2],
            (0.5 - self.ad) * self.dx,
            1.0 * self.dx,
            2.0 * self.dx,
            1.5 * self.dx,
        )
        Tip52, _, _ = self._interp_lagrange_amont_centre(
            self.Td[1],
            self.Td[2],
            self.Td[3],
            1.0 * self.dx,
            2.0 * self.dx,
            3.0 * self.dx,
            2.5 * self.dx,
        )
        self._T_f[
            0
        ] = (
            np.nan
        )  # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[1] = Tim32
        self._T_f[2] = Tim12  # self._T_dlg(0.)
        self._T_f[3] = Tip12  # self._T_dld(self.dx)
        self._T_f[4] = Tip32
        self._T_f[5] = Tip52
        self._gradT_f[0] = np.nan
        self._gradT_f[1] = np.nan
        self._gradT_f[2] = dTdxim12
        self._gradT_f[3] = dTdxip12
        self._gradT_f[4] = np.nan
        self._gradT_f[5] = np.nan

    def compute_T_f_gradT_f_quick(self):
        """
        Cellule type ::

                Tg, gradTg                          Tghost
                         0          1         2         3
               +---------+----------+---------+---------+
               |         |          |         |   |     |
               |    +   -|>   +    -|>   +   -|>  |+    |
               |    0    |    1     |    2    |   |3    |         4         5
               +---------+----------+---------+---------+---------+---------+---------+
                                              |   |     |         |         |         |
                                 Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                              |   |0    |    1    |    2    |    3    |
                                              +---------+---------+---------+---------+
                                                  Ti,                              |--|
                                                  lda_gradTi                         vdt

        Returns:

        """
        Tim32, dTdxim32, _ = self._interp_lagrange_amont(
            self.Tg[0],
            self.Tg[1],
            self.Tg[2],
            -2 * self.dx,
            -1.0 * self.dx,
            0.0 * self.dx,
            -0.5 * self.dx,
        )
        Tim12, _, _ = self._interp_lagrange_amont(
            self.Tg[1],
            self.Tg[2],
            self.Ti,
            -2.0 * self.dx,
            -1.0 * self.dx,
            (self.ag - 0.5) * self.dx,
            -0.5 * self.dx,
        )
        _, dTdxim12, _ = self._interp_lagrange_centre_grad(
            self.Tg[-3],
            self.Tg[-2],
            self.Ti,
            self._dTdxg,
            -2 * self.dx,
            -1 * self.dx,
            (self.ag - 0.5) * self.dx,
            -0.5 * self.dx,
        )
        Tip12, _, _ = self._interp_lagrange_amont_grad(
            self.Ti,
            self.Td[1],
            self._dTdxd,
            (0.5 - self.ad) * self.dx,
            1.0 * self.dx,
            0.5 * self.dx,
        )
        _, dTdxip12, _ = self._interp_lagrange_centre_grad(
            self.Td[2],
            self.Td[1],
            self.Ti,
            self._dTdxd,
            2.0 * self.dx,
            1.0 * self.dx,
            (0.5 - self.ad) * self.dx,
            0.5 * self.dx,
        )
        Tip32, dTdxip32, _ = self._interp_lagrange_amont(
            self.Ti,
            self.Td[1],
            self.Td[2],
            (0.5 - self.ad) * self.dx,
            1.0 * self.dx,
            2.0 * self.dx,
            1.5 * self.dx,
        )
        Tip52, dTdxip52, _ = self._interp_lagrange_amont(
            self.Td[1],
            self.Td[2],
            self.Td[3],
            1.0 * self.dx,
            2.0 * self.dx,
            3.0 * self.dx,
            2.5 * self.dx,
        )
        self._T_f[
            0
        ] = (
            np.nan
        )  # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[1] = Tim32
        self._T_f[2] = Tim12  # self._T_dlg(0.)
        self._T_f[3] = Tip12  # self._T_dld(self.dx)
        self._T_f[4] = Tip32
        self._T_f[5] = Tip52
        self._gradT_f[0] = np.nan
        self._gradT_f[1] = dTdxim32
        self._gradT_f[2] = dTdxim12
        self._gradT_f[3] = dTdxip12
        self._gradT_f[4] = dTdxip32
        self._gradT_f[5] = dTdxip52

    def compute_T_f_gradT_f_quick_ghost(self):
        """
        Cellule type ::

                Tg, gradTg                          Tghost
                         0          1         2         3
               +---------+----------+---------+---------+
               |         |          |         |   |     |
               |    +   -|>   +    -|>   +   -|>  |+    |
               |    0    |    1     |    2    |   |3    |         4         5
               +---------+----------+---------+---------+---------+---------+---------+
                                              |   |     |         |         |         |
                                 Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                              |   |0    |    1    |    2    |    3    |
                                              +---------+---------+---------+---------+
                                                  Ti,                              |--|
                                                  lda_gradTi                         vdt

        Returns:
            Tf avec :
            - -3/2 en quick pas corrige
            - -1/2 en quick avec Tghost gauche
            -  1/2 en upwind avec Td ghost et gradTd_ghost
            -  3/2 en quick avec Td ghost, Td1 et Td2
            -  5/2 en quick pas corrige

        """
        # calcul Tghost

        _, _, _, Tim12, _ = interpolate_from_center_to_face_quick(self.Tg)
        Tip12 = (
            self.Td[0] + self.lda_gradTi / self.ldad * self.dx * 0.5
        )  # interpolation amont
        _, _, Tip32, _, _ = interpolate_from_center_to_face_quick(self.Td)
        # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[0] = np.nan
        # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[1] = np.nan
        self._T_f[2] = Tim12  # self._T_dlg(0.)
        self._T_f[3] = Tip12  # self._T_dld(self.dx)
        self._T_f[4] = Tip32
        # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[5] = np.nan

        self._gradT_f[0] = np.nan
        self._gradT_f[1] = np.nan
        self._gradT_f[2] = (self.Tg[-1] - self.Tg[-2]) / self.dx
        self._gradT_f[3] = (self.Td[1] - self.Td[0]) / self.dx
        self._gradT_f[4] = np.nan
        self._gradT_f[5] = np.nan

    def compute_T_f_gradT_f_quick_upwind_ghost(self):
        """
        Cellule type ::

                Tg, gradTg                          Tghost
                         0          1         2         3
               +---------+----------+---------+---------+
               |         |          |         |   |     |
               |    +   -|>   +    -|>   +   -|>  |+    |
               |    0    |    1     |    2    |   |3    |         4         5
               +---------+----------+---------+---------+---------+---------+---------+
                                              |   |     |         |         |         |
                                 Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                              |   |0    |    1    |    2    |    3    |
                                              +---------+---------+---------+---------+
                                                  Ti,                              |--|
                                                  lda_gradTi                         vdt

        Returns:
            Tf avec :
            - -3/2 en quick pas corrige
            - -1/2 en quick avec Tghost gauche
            -  1/2 en centre avec T ghost droit et Td1
            -  3/2 en quick avec Tghost droit Td1 et Td2
            -  5/2 en quick pas corrige

        """
        # calcul Tghost

        # _, _, Tim32, _, _ = interpolate_from_center_to_face_quick(self.Tg)
        Tim12 = self.Tg[-2]  # extrapolation amont
        Tip12 = self.Td[0]  # interpolation amont
        Tip32 = self.Td[1]  # interpolation amont

        # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[0] = np.nan
        # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[1] = np.nan
        self._T_f[2] = Tim12  # self._T_dlg(0.)
        self._T_f[3] = Tip12  # self._T_dld(self.dx)
        self._T_f[4] = Tip32
        # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[5] = np.nan

        self._gradT_f[0] = np.nan
        self._gradT_f[1] = np.nan
        self._gradT_f[2] = (self.Tg[-1] - self.Tg[-2]) / self.dx
        self._gradT_f[3] = (self.Td[1] - self.Td[0]) / self.dx
        self._gradT_f[4] = np.nan
        self._gradT_f[5] = np.nan

    def compute_T_f_gradT_f_upwind(self):
        """
        Cellule type ::

                Tg, gradTg                          Tghost
                         0          1         2         3
               +---------+----------+---------+---------+
               |         |          |         |   |     |
               |    +   -|>   +    -|>   +   -|>  |+    |
               |    0    |    1     |    2    |   |3    |         4         5
               +---------+----------+---------+---------+---------+---------+---------+
                                              |   |     |         |         |         |
                                 Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                              |   |0    |    1    |    2    |    3    |
                                              +---------+---------+---------+---------+
                                                  Ti,                              |--|
                                                  lda_gradTi                         vdt

        Returns:

        """
        Tim32, dTdxim32, _ = self._interp_upwind(
            self.Tg[1], self.Tg[2], -1.0 * self.dx, 0.0 * self.dx
        )
        Tim12, dTdxim12, _ = self._interp_upwind(
            self.Tg[-1], self.Tg[0], -1.0 * self.dx, 0.0 * self.dx
        )
        Tip12, dTdxip12, _ = self._interp_upwind(
            self.Td[0], self.Td[1], 0.0 * self.dx, 1.0 * self.dx
        )
        Tip32, dTdxip32, _ = self._interp_upwind(
            self.Td[1], self.Td[2], 1.0 * self.dx, 2.0 * self.dx
        )
        Tip52, dTdxip52, _ = self._interp_upwind(
            self.Td[2], self.Td[3], 2.0 * self.dx, 3.0 * self.dx
        )
        self._T_f[
            0
        ] = (
            np.nan
        )  # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[1] = Tim32
        self._T_f[2] = Tim12  # self._T_dlg(0.)
        self._T_f[3] = Tip12  # self._T_dld(self.dx)
        self._T_f[4] = Tip32
        self._T_f[5] = Tip52
        self._gradT_f[0] = np.nan
        self._gradT_f[1] = dTdxim32
        self._gradT_f[2] = dTdxim12
        self._gradT_f[3] = dTdxip12
        self._gradT_f[4] = dTdxip32
        self._gradT_f[5] = dTdxip52

    def compute_T_f_gradT_f_quick_vol(self):
        """
        Cellule type ::

                Tg, gradTg                          Tghost
                         0          1         2         3
               +---------+----------+---------+---------+
               |         |          |         |   |     |
               |    +   -|>   +    -|>   +   -|>  |+    |
               |    0    |    1     |    2    |   |3    |         4         5
               +---------+----------+---------+---------+---------+---------+---------+
                                              |   |     |         |         |         |
                                 Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                              |   |0    |    1    |    2    |    3    |
                                              +---------+---------+---------+---------+
                                                  Ti,                              |--|
                                                  lda_gradTi                         vdt

        Returns:

        """
        Tim32, dTdxim32, _ = self._interp_lagrange_amont_vol(
            self.Tg[0],
            self.Tg[1],
            self.Tg[2],
            -2 * self.dx,
            -1.0 * self.dx,
            0.0 * self.dx,
            -0.5 * self.dx,
        )
        Tim12, dTdxim12, _ = self._interp_lagrange_amont_vol(
            self.Tg[1],
            self.Tg[2],
            self._Ti,
            -2.0 * self.dx,
            -1.0 * self.dx,
            (self.ag - 0.5) * self.dx,
            -0.5 * self.dx,
        )
        Tip12, dTdxip12, _ = self._interp_lagrange_amont_grad_vol(
            self._Ti,
            self.Td[1],
            self._dTdxd,
            (0.5 - self.ad) * self.dx,
            1.0 * self.dx,
            0.5 * self.dx,
        )
        Tip32, dTdxip32, _ = self._interp_lagrange_amont_vol(
            self._Ti,
            self.Td[1],
            self.Td[2],
            (0.5 - self.ad) * self.dx,
            1.0 * self.dx,
            2.0 * self.dx,
            1.5 * self.dx,
        )
        Tip52, dTdxip52, _ = self._interp_lagrange_amont_vol(
            self.Td[1],
            self.Td[2],
            self.Td[3],
            1.0 * self.dx,
            2.0 * self.dx,
            3.0 * self.dx,
            2.5 * self.dx,
        )
        self._T_f[
            0
        ] = (
            np.nan
        )  # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[1] = Tim32
        self._T_f[2] = Tim12  # self._T_dlg(0.)
        self._T_f[3] = Tip12  # self._T_dld(self.dx)
        self._T_f[4] = Tip32
        self._T_f[5] = Tip52
        self._gradT_f[0] = np.nan
        self._gradT_f[1] = dTdxim32
        self._gradT_f[2] = dTdxim12
        self._gradT_f[3] = dTdxip12
        self._gradT_f[4] = dTdxip32
        self._gradT_f[5] = dTdxip52

    def compute_from_ldagradTi(self):
        """
        On commence par récupérer lda_grad_Ti, gradTg, gradTd par continuité à partir de gradTim32 et gradTip32.
        On en déduit Ti par continuité à gauche et à droite.

        Returns:
            Calcule les gradients g, I, d, et Ti. gradTig et gradTid sont les gradients centrés entre TI et Tim1 et TI
            et Tip1

        Warnings:
            Cette méthode est dépréciée, elle a recours à des extrapolation d'ordre 1
        """
        (
            lda_gradTg,
            lda_gradTig,
            self._lda_gradTi,
            lda_gradTid,
            lda_gradTd,
        ) = self._get_lda_grad_T_i_from_ldagradT_continuity(
            self.Tg[1],
            self.Tg[2],
            self.Td[1],
            self.Td[2],
            (1.0 + self.ag) * self.dx,
            (1.0 + self.ad) * self.dx,
        )

        self._Ti = self._get_Ti_from_lda_grad_Ti(
            self.Tg[-2],
            self.Td[1],
            (0.5 + self.ag) * self.dx,
            (0.5 + self.ad) * self.dx,
            lda_gradTig,
            lda_gradTid,
        )
        self.Tg[-1] = self.Tg[-2] + lda_gradTg / self.ldag * self.dx
        self.Td[0] = self.Td[1] - lda_gradTd / self.ldad * self.dx

    def compute_from_ldagradTi_ordre2(self):
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

        self._Ti = self._get_Ti_from_lda_grad_Ti(
            self.Tg[-2],
            self.Td[1],
            (0.5 + self.ag) * self.dx,
            (0.5 + self.ad) * self.dx,
            lda_gradTig,
            lda_gradTid,
        )
        self.Tg[-1] = self.Tg[-2] + lda_gradTig / self.ldag * self.dx
        self.Td[0] = self.Td[1] - lda_gradTid / self.ldad * self.dx

    def compute_from_Ti_ghost(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1.
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        C'est la méthode qui donne les résultats les plus stables. Probablement parce qu'elle donne un poids plus
        important aux valeurs des mailles monophasiques

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(
            self.Tg[-2],
            self.Td[1],
            (0.5 + self.ag) * self.dx,
            (0.5 + self.ad) * self.dx,
        )
        self._dTdxg = self._lda_gradTi / self.ldag
        self._dTdxd = self._lda_gradTi / self.ldad
        self.Tg[-1] = self._Ti + self._dTdxg * self.dx * (0.5 - self.ag)
        self.Td[0] = self._Ti + self._dTdxd * self.dx * (self.ad - 0.5)
        # self.grad_Tim12 = self.pid_interp(
        #     np.array([self.gradTg[1], self._lda_gradTi / self.ldag]),
        #     np.array([1.0, self.ag]) * self.dx,
        # )
        # self.grad_Tip12 = self.pid_interp(
        #     np.array([self._lda_gradTi / self.ldad, self.gradTd[1]]),
        #     np.array([self.ad, 1.0]) * self.dx,
        # )

    def compute_from_Ti(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1.
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        C'est la méthode qui donne les résultats les plus stables. Probablement parce qu'elle donne un poids plus
        important aux valeurs des mailles monophasiques

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(
            self.Tg[-2],
            self.Td[1],
            (1.0 / 2 + self.ag) * self.dx,
            (1.0 / 2 + self.ad) * self.dx,
        )
        self._dTdxg = self._lda_gradTi / self.ldag
        self._dTdxd = self._lda_gradTi / self.ldad
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

    def compute_from_Ti2(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1, Tim2 et Tip2
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
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
        self._dTdxg = self._lda_gradTi / self.ldag
        self._dTdxd = self._lda_gradTi / self.ldad
        self.Tg[-1] = self._T_dlg(0.5 * self.dx)
        self.Td[0] = self._T_dld(0.5 * self.dx)

    def compute_from_Ti2_vol(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1, Tim2 et Tip2
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
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
        self._dTdxg = self._lda_gradTi / self.ldag
        self._dTdxd = self._lda_gradTi / self.ldad
        self.Tg[-1] = self._T_dlg(0.5 * self.dx)
        self.Td[0] = self._T_dld(0.5 * self.dx)

    def compute_from_Ti3(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1, Tim2 et Tip2
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
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
        self._dTdxg = self._lda_gradTi / self.ldag
        self._dTdxd = self._lda_gradTi / self.ldad
        self.Tg[-1] = self._T_dlg(0.5 * self.dx)
        self.Td[0] = self._T_dld(0.5 * self.dx)

    def compute_from_Ti3_vol(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1, Tim2 et Tip2
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
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
        self._dTdxg = self._lda_gradTi / self.ldag
        self._dTdxd = self._lda_gradTi / self.ldad
        self.Tg[-1] = self._T_dlg(0.5 * self.dx)
        self.Td[0] = self._T_dld(0.5 * self.dx)

    def compute_from_Ti3_1_vol(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1, Tim2 et Tip2
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        (
            self._Ti,
            self._lda_gradTi,
            self._d2Tdx2g,
            self._d2Tdx2d,
        ) = self._get_T_i_and_lda_grad_T_i3_1_vol(
            self.Tg[-4],
            self.Tg[-3],
            self.Tg[-2],
            self.Td[1],
            (1.0 / 2 + self.ag) * self.dx,
            (1.0 / 2 + self.ad) * self.dx,
        )
        self._dTdxg = self._lda_gradTi / self.ldag
        self._dTdxd = self._lda_gradTi / self.ldad
        self.Tg[-1] = self._T_dlg(0.5 * self.dx)
        self.Td[0] = self._T_dld(0.5 * self.dx)

    def _compute_Tgc_Tdc(self, h: float, T_mean: float):
        """
        Résout le système d'équation entre la température moyenne et l'énergie de la cellule pour trouver les valeurs de
        Tgc et Tdc.
        Le système peut toujours être résolu car rhocpg != rhocpd

        Returns:
            Rien mais mets à jour Tgc et Tdc
        """
        system = np.array(
            [[self.ag * self.rhocpg, self.ad * self.rhocpd], [self.ag, self.ad]]
        )
        self.Tgc, self.Tdc = np.dot(np.linalg.inv(system), np.array([h, T_mean]))

    def compute_from_h_T(self, h: float, T_mean: float):
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
        # On commence par calculer T_I et lda_grad_Ti en fonction de Tgc et Tdc :
        EPSILON = 10**-6
        if self.ag < EPSILON:
            self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(
                self.Tg[-2],
                T_mean,
                (1.0 / 2.0 + self.ag) * self.dx,
                self.ad / 2.0 * self.dx,
            )
        elif self.ad < EPSILON:
            self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(
                T_mean,
                self.Td[1],
                self.ag / 2.0 * self.dx,
                (1.0 / 2.0 + self.ad) * self.dx,
            )
        else:
            self._compute_Tgc_Tdc(h, T_mean)
            self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(
                self.Tgc, self.Tdc, self.ag / 2.0 * self.dx, self.ad / 2.0 * self.dx
            )

        self._dTdxg = self._lda_gradTi / self.ldag
        self._dTdxd = self._lda_gradTi / self.ldad
        self.Tg[-1] = self._T_dlg(0.5 * self.dx)
        self.Td[0] = self._T_dld(0.5 * self.dx)
        # grad_Tg = self.pid_interp(np.array([self.gradTg[1], self._lda_gradTi/self.ldag]),
        #                           np.array([1., self.ag])*self.dx)
        # grad_Td = self.pid_interp(np.array([self._lda_gradTi/self.ldad, self.gradTd[1]]),
        #                           np.array([self.ad, 1.])*self.dx)
        # self.Tg[-1] = self.Tg[-2] + grad_Tg * self.dx
        # self.Td[0] = self.Td[1] - grad_Td * self.dx

    def compute_from_Ti_and_ip1(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Ti et Tip1 (ou Tim1 selon la position de l'interface).
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Cellule type ::


                       0          1         2         3
             +---------+----------+---------+---------+---------+---------+---------+
             |         |          |         |   |     |         |         |         |
             |    +   -|>   +    -|>   +   -|>  |+    |>   +   -|>   +   -|>   +    |
             |    0    |    1     |    2    |   |3    |    4    |    5    |    6    |
             +---------+----------+---------+---------+---------+---------+---------+
                                                Ti,
                                                lda_gradTi

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        (
            self._Ti,
            self._lda_gradTi,
        ) = self._get_T_i_and_lda_grad_T_i_from_integrated_temperature(
            self._T[2], self._T[3], self._T[4], self.ag, self.dx
        )
        self._dTdxg = self._lda_gradTi / self.ldag
        self._dTdxd = self._lda_gradTi / self.ldad
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

    def _get_T_i_and_lda_grad_T_i_limited(
        self, Tg: float, Td: float, Tgg: float, Tdd: float, dg: float, dd: float
    ) -> (float, float):
        """
        On utilise la continuité de lda_grad_T pour interpoler en Ti, on en déduit Ti et lda_grad_Ti
        Puis on limite lda_grad_Ti a partir de ce qui se passe de chaque côté
        On retourne les gradients suivants ::

                                          dgg    dg              dd      ddd
                            |---------------|-----------|-------------------|---------------|
                    +---------------+---------------+---------------+---------------+---------------+
                    |               |               |   |           |               |               |
                   -|>      +      -|>      +      -|>  |   +      -|>      +      -|>      +      -|>
                    |               |               |   |           |               |               |
                    +---------------+---------------+---------------+---------------+---------------+

        Returns:
            Ti, lda_grad_T
        """
        T_i = (self.ldag / dg * Tg + self.ldad / dd * Td) / (
            self.ldag / dg + self.ldad / dd
        )
        lda_grad_T_ig = self.ldag * (T_i - Tg) / dg
        lda_grad_T_id = self.ldad * (Td - T_i) / dd
        lda_grad_T = (lda_grad_T_ig + lda_grad_T_id) / 2.0
        lda_grad_Tg = self.ldag * (Tg - Tgg) / self.dx
        lda_grad_Td = self.ldad * (Tdd - Td) / self.dx
        if lda_grad_Tg < lda_grad_Td:
            lda_grad_Tmin = lda_grad_Tg
            lda_grad_Tmax = lda_grad_Td
        else:
            lda_grad_Tmax = lda_grad_Tg
            lda_grad_Tmin = lda_grad_Td
        recalcul = True
        if lda_grad_T < lda_grad_Tmin:
            lda_grad_T = lda_grad_Tmin
        elif lda_grad_T > lda_grad_Tmax:
            lda_grad_T = lda_grad_Tmax
        else:
            recalcul = False
        # recalcul de Ti ? comment ?
        # TODO: vérifier que ce n'est pas completement arbitraire...
        if recalcul:
            T_ig = Tg + lda_grad_T / self.ldag * (0.5 + self.ag) * self.dx
            T_id = Td - lda_grad_T / self.ldad * (0.5 + self.ad) * self.dx
            T_i = (T_id + T_ig) / 2.0

        return T_i, lda_grad_T

    def _get_T_i_and_lda_grad_T_i2(
        self, Tg: float, Td: float, Tgg: float, Tdd: float, dg: float, dd: float
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

    def _get_T_i_and_lda_grad_T_i2_vol(
        self, Tg: float, Td: float, Tgg: float, Tdd: float, dg: float, dd: float
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
                [1.0, -dg, dg**2 * 2.0 / 3.0, 0.0, -(dg**3) * 1.0 / 3.0, 0.0],
                [1.0, -dgg, dgg**2 * 2.0 / 3.0, 0.0, -(dgg**3) * 1.0 / 3.0, 0.0],
                [1.0, -dggg, dggg**2 * 2.0 / 3.0, 0.0, -(dggg**3) * 1.0 / 3.0, 0.0],
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

    def _get_T_i_and_lda_grad_T_i3_1_vol(
        self, Tggg: float, Tgg: float, Tg: float, Td: float, dg: float, dd: float
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

    def _T_dlg(self, x: float) -> float:
        xi = self.ag * self.dx
        return (
            self.Ti
            + (x - xi) * self._dTdxg
            + (x - xi) ** 2 / 2.0 * self._d2Tdx2g
            + (x - xi) ** 3 / 6.0 * self._d3Tdx3g
        )

    def _T_dld(self, x: float) -> float:
        xi = self.ag * self.dx
        return (
            self.Ti
            + (x - xi) * self._dTdxd
            + (x - xi) ** 2 / 2.0 * self._d2Tdx2d
            + (x - xi) ** 3 / 6.0 * self._d3Tdx3d
        )

    def _gradT_dlg(self, x: float) -> float:
        xi = self.ag * self.dx
        return (
            self._dTdxg + (x - xi) * self._d2Tdx2g + (x - xi) ** 2 / 2.0 * self._d3Tdx3g
        )

    def _gradT_dld(self, x: float) -> float:
        xi = self.ag * self.dx
        return (
            self._dTdxd + (x - xi) * self._d2Tdx2d + (x - xi) ** 2 / 2.0 * self._d3Tdx3d
        )

    @staticmethod
    def _interp_upwind(
        T0: float, T1: float, x0: float, x1: float
    ) -> (float, float, float):
        """
        Dans cette méthode on veux que T0 et T1 soient amont de x_int
        C'est une interpolation d'ordre 0

        Args:
            T0:
            T1:
            x0:
            x1:

        Returns:

        """

        Tint = T0
        dTdx_int = (T1 - T0) / (x1 - x0)
        d2Tdx2_int = 0.0
        return Tint, dTdx_int, d2Tdx2_int

    @staticmethod
    def _interp_amont_decentre(
        T0: float, gradT0: float, x0: float, xint: float
    ) -> (float, float, float):
        """
        Dans cette méthode on veux que T0 soit amont de xint, et gradT0 soit le gradient en T0.
        C'est une interpolation d'ordre 1 décentrée amont.

        """

        Tint = T0 + gradT0 * (xint - x0)
        d2Tdx2_int = 0.0
        return Tint, gradT0, d2Tdx2_int

    def _interp_lagrange_amont_vol(
        self,
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
        d00 = x0 - x_int - 0.5 * self.dx
        d01 = x0 - x_int + 0.5 * self.dx
        d10 = x1 - x_int - 0.5 * self.dx
        d11 = x1 - x_int + 0.5 * self.dx
        d20 = x2 - x_int - 0.5 * self.dx
        d21 = x2 - x_int + 0.5 * self.dx

        mat = np.array(
            [
                [
                    1.0,
                    (d01**2 - d00**2) / 2.0 / self.dx,
                    (d01**3 - d00**3) / 6.0 / self.dx,
                ],
                [
                    1.0,
                    (d11**2 - d10**2) / 2.0 / self.dx,
                    (d11**3 - d10**3) / 6.0 / self.dx,
                ],
                [
                    1.0,
                    (d21**2 - d20**2) / 2.0 / self.dx,
                    (d21**3 - d20**3) / 6.0 / self.dx,
                ],
            ],
            dtype=np.float_,
        )
        inv_mat = np.linalg.inv(mat)
        Tint, dTdx_int, d2Tdx2_int = np.dot(inv_mat, np.array([T0, T1, T2]))
        return Tint, dTdx_int, d2Tdx2_int

    def _interp_lagrange_amont_grad_vol(
        self, T0: float, T1: float, gradT0: float, x0: float, x1: float, x_int: float
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
        d00 = x0 - x_int - 0.5 * self.dx
        d01 = x0 - x_int + 0.5 * self.dx
        d10 = x1 - x_int - 0.5 * self.dx
        d11 = x1 - x_int + 0.5 * self.dx

        mat = np.array(
            [
                [
                    1.0,
                    (d01**2 - d00**2) / 2.0 / self.dx,
                    (d01**3 - d00**3) / 6.0 / self.dx,
                ],
                [
                    1.0,
                    (d11**2 - d10**2) / 2.0 / self.dx,
                    (d11**3 - d10**3) / 6.0 / self.dx,
                ],
                [0.0, 1.0, (d01**2 - d00**2) / 2.0 / self.dx],
            ],
            dtype=np.float_,
        )
        Tint, dTdx_int, d2Tdx2_int = np.dot(
            np.linalg.inv(mat), np.array([T0, T1, gradT0])
        )
        return Tint, dTdx_int, d2Tdx2_int

    @staticmethod
    def _interp_lagrange_aval(
        T0: float, T1: float, T2: float, x0: float, x1: float, x2: float, x_int: float
    ) -> (float, float, float):
        """
        Dans cette méthode on veux que T0 seulement soit amont de x_int.
        C'est une interpolation d'ordre 3, attention elle ne peut être utilisée que pour calculer le gradient de
        température, la température elle même doit impérativement être calculée en amont.

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

    @staticmethod
    def _interp_lagrange_amont_centre(
        T0: float, T1: float, T2: float, x0: float, x1: float, x2: float, x_int: float
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

    @staticmethod
    def _interp_lagrange_amont(
        T0: float, T1: float, T2: float, x0: float, x1: float, x2: float, x_int: float
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

    @staticmethod
    def _interp_lagrange_amont_grad(
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
        Tint, dTdx_int, d2Tdx2_int = np.dot(
            np.linalg.inv(mat), np.array([T0, T1, gradT0])
        )
        return Tint, dTdx_int, d2Tdx2_int

    @staticmethod
    def _interp_lagrange_centre_grad(
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

    def _get_lda_grad_T_i_from_ldagradT_continuity(
        self, Tim2: float, Tim1: float, Tip1: float, Tip2: float, dg: float, dd: float
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
            np.array([ldagradTgg, ldagradTdd]), np.array([self.dx, 2.0 * self.dx])
        )
        lda_gradTgi = self.pid_interp(
            np.array([ldagradTgg, ldagradTdd]),
            np.array(
                [dg - (dg - 0.5 * self.dx) / 2.0, dd + (dg - 0.5 * self.dx) / 2.0]
            ),
        )
        lda_gradTi = self.pid_interp(
            np.array([ldagradTgg, ldagradTdd]), np.array([dg, dd])
        )
        lda_gradTdi = self.pid_interp(
            np.array([ldagradTgg, ldagradTdd]),
            np.array(
                [dg + (dd - 0.5 * self.dx) / 2.0, dd - (dd - 0.5 * self.dx) / 2.0]
            ),
        )
        lda_gradTd = self.pid_interp(
            np.array([ldagradTgg, ldagradTdd]), np.array([2.0 * self.dx, self.dx])
        )
        return lda_gradTg, lda_gradTgi, lda_gradTi, lda_gradTdi, lda_gradTd

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
            np.array([lda_gradTim32, lda_gradTi]), np.array([self.dx, ag * self.dx])
        )
        lda_gradTd = self.pid_interp(
            np.array([lda_gradTi, lda_gradTip32]), np.array([ad * self.dx, self.dx])
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


class CellsSuiviInterface:
    """
    Cette classe contient des cellules j qui suivent l'interface ::

             Tg, gradTg                          Tghost
            +---------+----------+---------+---------+
            |         |          |         |   |     |
            |    +   -|>   +    -|>   +   -|>  |+    |
            |    0    |    1     |    2    |   |3    |
            +---------+----------+---------+---------+---------+---------+---------+
                                           |   |     |         |         |         |
                              Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                           |   |0    |    1    |    2    |    3    |
               +----------+----------+-----+---+-----+---+-----+---+-----+---+-----+
               |          |          |         |         |         |         |
               |    +     |    +     |    +    |    +    |    +    |    +    |
               |    jm2   |    jm1   |    j    |    jp1  |    jp2  |    jp3  |
               +----------+----------+---------+---------+---------+---------+
                                              T_I

    """

    def __init__(
        self,
        ldag=1.0,
        ldad=1.0,
        ag=1.0,
        dx=1.0,
        T=None,
        rhocpg=1.0,
        rhocpd=1.0,
        vdt=0.0,
        interp_type=None,
    ):
        self.cells_fixe = CellsInterface(
            ldag=ldag,
            ldad=ldad,
            ag=ag,
            dx=dx,
            T=T,
            rhocpg=rhocpg,
            rhocpd=rhocpd,
            vdt=vdt,
            interp_type=interp_type,
        )
        self.dx = dx
        self.Tj = np.zeros((6,))
        self.Tjnp1 = np.zeros((4,))
        # Ici on calcule Tg et Td pour ensuite interpoler les valeurs à proximité de l'interface
        if self.cells_fixe.interp_type == "Ti":
            self.cells_fixe.compute_from_Ti()
        elif self.cells_fixe.interp_type == "Ti2":
            self.cells_fixe.compute_from_Ti2()
        elif self.cells_fixe.interp_type == "Ti3":
            self.cells_fixe.compute_from_Ti3()
        elif self.cells_fixe.interp_type == "gradTi":
            self.cells_fixe.compute_from_ldagradTi()
        else:
            self.cells_fixe.compute_from_ldagradTi_ordre2()

        # le zéro correspond à la position du centre de la maille i
        x_I = (ag - 1.0 / 2) * dx
        self.xj = np.linspace(-2, 3, 6) * dx + x_I - 1.0 / 2 * dx

        self.Tj[:3] = self._interp_from_i_to_j_g(self.cells_fixe.Tg, self.cells_fixe.dx)
        self.Tj[3:] = self._interp_from_i_to_j_d(self.cells_fixe.Td, self.cells_fixe.dx)

    def _interp_from_i_to_j_g(self, Ti, dx):
        """
        On récupère un tableau de taille Ti - 1

        Args:
            Ti: la température à gauche
            dx (float):

        Returns:

        """
        Tj = np.empty((len(Ti) - 1,))
        xi = np.linspace(-3, 0, 4) * dx
        for j in range(len(Tj)):
            i = j + 1
            d_im1_j_i = np.abs(xi[[i - 1, i]] - self.xj[:3][j])
            Tj[j] = self.cells_fixe.pid_interp(Ti[[i - 1, i]], d_im1_j_i)
        return Tj

    def _interp_from_i_to_j_d(self, Ti, dx):
        """
        On récupère un tableau de taille Ti - 1

        Args:
            Ti: la température à droite
            dx (float):

        Returns:

        """
        Tj = np.empty((len(Ti) - 1,))
        xi = np.linspace(0, 3, 4) * dx
        for j in range(len(Tj)):
            i = j + 1
            d_im1_j_i = np.abs(xi[[i - 1, i]] - self.xj[3:][j])
            Tj[j] = self.cells_fixe.pid_interp(Ti[[i - 1, i]], d_im1_j_i)
        return Tj

    def timestep(self, diff, dt):
        gradT = (self.Tj[1:] - self.Tj[:-1]) / self.cells_fixe.dx
        lda_grad_T = gradT * np.array(
            [
                self.cells_fixe.ldag,
                self.cells_fixe.ldag,
                np.nan,
                self.cells_fixe.ldad,
                self.cells_fixe.ldad,
            ]
        )  # taille 5
        lda_grad_T[2] = self.cells_fixe.lda_gradTi
        rho_cp_center = np.array(
            [
                self.cells_fixe.rhocpg,
                self.cells_fixe.rhocpg,
                self.cells_fixe.rhocpd,
                self.cells_fixe.rhocpd,
            ]
        )  # taille 4
        # le pas de temps de diffusion
        self.Tjnp1 = (
            self.Tj[1:-1]
            + dt * 1 / rho_cp_center * integrale_vol_div(lda_grad_T, self.dx) * diff
        )

    def interp_T_from_j_to_i(self):
        """
        Ici on récupère un tableau de température centré en i. La valeur de température en i correspond à une
        interpolation entre la valeur à l'interface et la valeur du bon côté (j et j+1)

        Args:

        Returns:

        """
        Tj = self.Tjnp1
        Ti = np.empty((len(Tj) - 1,))
        xi = np.linspace(-1, 1, 3) * self.dx
        x_I = (self.cells_fixe.ag - 1.0 / 2) * self.dx + self.cells_fixe.vdt
        xj = np.linspace(-1, 2, 4) * self.dx + x_I - 1.0 / 2 * self.dx
        for i in range(len(Ti)):
            j = i + 1
            d_jm1_i_j = np.abs(xj[[j - 1, j]] - xi[i])
            Ti[i] = self.cells_fixe.pid_interp(Tj[[j - 1, j]], d_jm1_i_j)

        # Il faut traiter à part le cas de la cellule qu'on doit interpoler avec I
        # 3 possibilités se présentent :
        # soit l'interface est à gauche du milieu de la cellule i
        if x_I < 0.0:
            # dans ce cas on fait l'interpolation entre I et j+1
            d = np.abs(np.array([x_I, xj[2]]))
            Ti[1] = self.cells_fixe.pid_interp(np.array([self.cells_fixe.Ti, Tj[2]]), d)
        # soit l'interface est à droite du milieu de la cellule i, mais toujours dans la cellule i
        elif x_I < 0.5 * self.dx:
            # dans ce cas on fait l'interpolation entre j et I
            d = np.abs(np.array([xj[1], x_I]))
            Ti[1] = self.cells_fixe.pid_interp(np.array([Tj[1], self.cells_fixe.Ti]), d)
        # soit l'interface est passée à droite de la face i+1/2
        else:
            # dans ce cas on fait l'interpolation entre I et j+1 pour la température i+1
            d = np.abs(np.array([x_I, xj[3]]) - xi[2])
            Ti[2] = self.cells_fixe.pid_interp(np.array([self.cells_fixe.Ti, Tj[3]]), d)
        return Ti


class ProblemDiscontinuEnergieTemperature(Problem):
    bulles: BulleTemperature

    """
    Cette classe résout le problème en couplant une équation sur la température et une équation sur l'énergie
    interne au niveau des interfaces.
    On a donc un tableau T et un tableau h

        - on calcule dans les mailles diphasiques Tgc et Tdc les températures au centres de la partie remplie par la
        phase à gauche et la partie remplie par la phase à droite.
        - on en déduit en interpolant des flux aux faces
        - on met à jour T et h avec des flux exprimés de manière monophasique.

    Le problème de cette formulation est qu'elle fait intervenir l'équation sur la température alors qu'on sait
    que cette équation n'est pas terrible.

    """

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, **kwargs):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        self.h = self.rho_cp_a * self.T
        self.flux_conv_energie = Flux(np.zeros_like(self.flux_conv))

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(markers=markers, phy_prop=self.phy_prop)
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def copy(self, pb):
        super().copy(pb)
        self.h = pb.h.copy()
        self.flux_conv_energie = pb.flux_conv_energie.copy()

    def _corrige_interface(self):
        """
        Dans cette approche on calcule Ti et lda_gradTi à partir du système énergie température

        Returns:
            Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
            jour T_old
        """
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(self.bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(self.T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    self.T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type="energie_temperature",
                    schema_conv="quick",
                    vdt=self.phy_prop.v * self.dt,
                )
                cells.compute_from_h_T(self.h[i0], self.T[i0])
                cells.compute_T_f_gradT_f_quick()
                # print(cells.rhocp_f)

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                rhocp_T_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                lda_grad_T = cells.lda_f * cells.gradT

                # Correction des cellules
                # ind_to_change = [im2, im1, i0, ip1, ip2]
                ind_flux_conv = [
                    im1,
                    i0,
                    ip1,
                    ip2,
                    ip3,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
                self.flux_conv[ind_flux_conv] = cells.T_f[1:] * self.phy_prop.v
                self.flux_conv_ener[ind_flux_conv] = rhocp_T_u[1:]
                self.flux_diff[ind_flux_diff] = lda_grad_T[2:4]
                # on écrit l'équation en température, et en energie
                # Tnp1 = Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T
                #                 - delta * I2 * (rhocp2 - rhocpa) - [rhocp] * int_S_Ti_v_n2_dS) / rhocpa

    def _euler_timestep(self, debug=None, bool_debug=False):
        self.flux_conv = (
            interpolate(self.T, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        self.flux_conv_ener = (
            interpolate(self.h, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        self.flux_diff = interpolate(
            self.Lda_h, I=self.I, schema=self.num_prop.schema
        ) * grad(self.T, self.num_prop.dx)
        self._corrige_interface()
        int_div_T_u = integrale_vol_div(self.flux_conv, self.num_prop.dx)
        int_div_rho_cp_T_u = integrale_vol_div(self.flux_conv_ener, self.num_prop.dx)
        int_div_lda_grad_T = integrale_vol_div(self.flux_diff, self.num_prop.dx)

        if (debug is not None) and bool_debug:
            debug.plot(
                self.num_prop.x,
                1.0 / self.rho_cp_h,
                label="rho_cp_inv_h, time = %f" % self.time,
            )
            debug.plot(
                self.num_prop.x,
                int_div_lda_grad_T,
                label="div_lda_grad_T, time = %f" % self.time,
            )
            debug.xticks(self.num_prop.x_f)
            debug.grid(which="major")
            maxi = max(np.max(int_div_lda_grad_T), np.max(1.0 / self.rho_cp_h))
            mini = min(np.min(int_div_lda_grad_T), np.min(1.0 / self.rho_cp_h))
            for markers in self.bulles():
                debug.plot([markers[0]] * 2, [mini, maxi], "--")
                debug.plot([markers[1]] * 2, [mini, maxi], "--")
            debug.legend()
        rho_cp_inv_h = 1.0 / self.rho_cp_h
        self.T += self.dt * (
            -int_div_T_u + self.phy_prop.diff * rho_cp_inv_h * int_div_lda_grad_T
        )
        self.h += self.dt * (
            -int_div_rho_cp_T_u + self.phy_prop.diff * int_div_lda_grad_T
        )
        # dT/dt = -inv_rho_cp * div_rho_cp_T_u + corr + rho_cp_inv * div_lda_grad_T
        # self.T += self.dt * (-inv_rho_cp_h * int_div_rho_cp_T_u + self.phy_prop.diff * rho_cp_inv_h * int_div_lda_grad_T)

    @property
    def name_cas(self):
        return "Energie température "


class ProblemDiscontinuEnergieTemperatureInt(Problem):
    bulles: BulleTemperature

    """
    Cette classe résout le problème en couplant une équation sur la température et une équation sur l'énergie
    interne au niveau des interfaces.
    On a donc un tableau T et un tableau h

        - on calcule dans les mailles diphasiques Tgc et Tdc les températures au centres de la partie remplie par la
        phase à gauche et la partie remplie par la phase à droite.
        - on en déduit en interpolant des flux aux faces
        - on met à jour T et h avec des flux exprimés de manière monophasique.

    Le problème de cette formulation est qu'elle fait intervenir l'équation sur la température alors qu'on sait
    que cette équation n'est pas terrible.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul

    """

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, **kwargs):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        self.h = self.rho_cp_a * self.T
        self.flux_conv_ener = Flux(np.zeros_like(self.flux_conv))
        self.flux_diff_temp = Flux(np.zeros_like(self.flux_conv))
        self.ind_interf = np.zeros_like(self.T)

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(markers=markers, phy_prop=self.phy_prop)
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def copy(self, pb):
        super().copy(pb)
        self.h = pb.h.copy()
        self.flux_conv_ener = pb.flux_conv_ener.copy()
        self.flux_diff_temp = pb.flux_diff_temp.copy()
        self.ind_interf = pb.ind_interf.copy()

    def _corrige_interface(self):
        """
        Dans cette approche on calclue Ti et lda_gradTi à partir du système énergie température

        Returns:
            Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
            jour T_old
        """
        dx = self.num_prop.dx
        self.ind_interf = np.zeros_like(self.T)

        for i_int, (i1, i2) in enumerate(self.bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(self.T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    self.T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type="energie_temperature",
                    schema_conv="quick",
                    vdt=self.phy_prop.v * self.dt,
                )
                cells.compute_from_h_T(self.h[i0], self.T[i0])
                cells.compute_T_f_gradT_f_quick()
                # print(cells.rhocp_f)

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                rhocp_T_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                lda_grad_T = cells.lda_f * cells.gradT

                # Correction des cellules
                # ind_to_change = [im2, im1, i0, ip1, ip2]
                ind_flux_conv = [
                    im1,
                    i0,
                    ip1,
                    ip2,
                    ip3,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
                self.flux_conv[ind_flux_conv] = cells.T_f[1:] * self.phy_prop.v
                self.flux_conv_ener[ind_flux_conv] = rhocp_T_u[1:]
                self.flux_diff[ind_flux_diff] = lda_grad_T[2:4]
                self.flux_diff_temp[ind_flux_diff] = (
                    lda_grad_T[2:4] * cells.inv_rhocp_f[2:4]
                )
                self.ind_interf[i0] = (1.0 / rhocpg - 1.0 / rhocpd) * cells.lda_gradTi
                # on écrit l'équation en température, et en energie
                # Tnp1 = Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T
                #                 - delta * I2 * (rhocp2 - rhocpa) - [rhocp] * int_S_Ti_v_n2_dS) / rhocpa

    def _euler_timestep(self, debug=None, bool_debug=False):
        self.flux_conv = (
            interpolate(self.T, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        self.flux_conv_ener = (
            interpolate(self.h, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        self.flux_diff = interpolate(
            self.Lda_h, I=self.I, schema=self.num_prop.schema
        ) * grad(self.T, self.num_prop.dx)
        # Attention, l'interpolation suivante n'est valide que dans le cas de deux cellules monophasiques adjacentes
        # elle nécessite impérativement une correction aux faces mitoyennes de l'interface.
        self.flux_diff_temp = interpolate(
            self.Lda_h / self.rho_cp_a, I=self.I, schema=self.num_prop.schema
        ) * grad(self.T, self.num_prop.dx)

        self._corrige_interface()
        self._echange_flux()
        self.flux_diff_temp.perio()
        self.flux_conv_ener.perio()

        int_div_T_u = integrale_vol_div(self.flux_conv, self.num_prop.dx)
        int_inv_rhocpf_div_ldaf_grad_T = integrale_vol_div(
            self.flux_diff_temp, self.num_prop.dx
        )
        int_div_rho_cp_T_u = integrale_vol_div(self.flux_conv_ener, self.num_prop.dx)
        int_div_lda_grad_T = integrale_vol_div(self.flux_diff, self.num_prop.dx)

        self.T += self.dt * (
            -int_div_T_u
            + self.phy_prop.diff * int_inv_rhocpf_div_ldaf_grad_T
            + self.phy_prop.diff / self.num_prop.dx * self.ind_interf
        )
        self.h += self.dt * (
            -int_div_rho_cp_T_u + self.phy_prop.diff * int_div_lda_grad_T
        )

    @property
    def name_cas(self):
        return "Energie température couplé"


class ProblemDiscontinuE(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Résolution en énergie.
    Cette classe résout le problème en 3 étapes :

        - on calcule le nouveau T comme avant (avec un stencil de 1 à proximité des interfaces par simplicité)
        - on calcule précisemment T1 et T2 ansi que les bons flux aux faces, on met à jour T
        - on met à jour T_i et lda_grad_T_i

    Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
    continuité de lda_grad_T et T à l'interface.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if num_prop.time_scheme == "rk3":
            print("RK3 is not implemented, changes to Euler")
            self.num_prop._time_scheme = "euler"
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf

    def copy(self, pb):
        super().copy(pb)
        self.conv_interf = pb.conv_interf
        self.interp_type = pb.interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.dt * self.phy_prop.v,
                )
                self.bulles.cells[2 * i_int + ist] = cells

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                lda_grad_T = cells.lda_f * cells.gradT
                self.bulles.post(cells, i_int, ist)

                # Correction des flux cellules
                ind_flux_conv = [
                    im1,
                    i0,
                    ip1,
                    ip2,
                    ip3,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
                flux_conv[ind_flux_conv] = rhocpT_u[1:]
                flux_diff[ind_flux_diff] = lda_grad_T[2:4]
                # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
            I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self._compute_convection_flux(self.rho_cp_a * self.T, self.bulles, debug)
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )

        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff
        )
        self._echange_flux()
        drhocpTdt = -integrale_vol_div(
            self.flux_conv, dx
        ) + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        self.T = (self.T * self.rho_cp_a + self.dt * drhocpTdt) / rho_cp_a_np1

    @property
    def name_cas(self):
        return "ESP"  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


class ProblemDiscontinuE_CN(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Résolution en énergie.
    Cette classe résout le problème en 3 étapes :

        - on calcule le nouveau T comme avant (avec un stencil de 1 à proximité des interfaces par simplicité)
        - on calcule précisemment T1 et T2 ansi que les bons flux aux faces, on met à jour T
        - on met à jour T_i et lda_grad_T_i

    Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
    continuité de lda_grad_T et T à l'interface.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if num_prop.time_scheme == "rk3":
            print("RK3 is not implemented, changes to Euler")
            self.num_prop._time_scheme = "euler"
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf

    def copy(self, pb):
        super().copy(pb)
        self.conv_interf = pb.conv_interf
        self.interp_type = pb.interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.dt * self.phy_prop.v,
                    time_integral="CN",
                )
                self.bulles.cells[2 * i_int + ist] = cells

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                lda_grad_T = cells.lda_f * cells.gradT
                self.bulles.post(cells, i_int, ist)

                # Correction des flux cellules
                ind_flux_conv = [
                    im1,
                    i0,
                    ip1,
                    ip2,
                    ip3,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
                flux_conv[ind_flux_conv] = rhocpT_u[1:]
                flux_diff[ind_flux_diff] = lda_grad_T[2:4]
                # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
            I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self._compute_convection_flux(self.rho_cp_a * self.T, self.bulles, debug)
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )

        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff
        )
        self._echange_flux()
        drhocpTdt = -integrale_vol_div(
            self.flux_conv, dx
        ) + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        self.T = (self.T * self.rho_cp_a + self.dt * drhocpTdt) / rho_cp_a_np1

    @property
    def name_cas(self):
        return "ESP CN"  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


class ProblemDiscontinuEsansq(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Résolution en énergie.
    Cette classe résout le problème en 3 étapes :

        - on calcule le nouveau T comme avant (avec un stencil de 1 à proximité des interfaces par simplicité)
        - on calcule précisemment T1 et T2 ansi que les bons flux aux faces, on met à jour T
        - on met à jour T_i et lda_grad_T_i

    Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
    continuité de lda_grad_T et T à l'interface.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if num_prop.time_scheme == "rk3":
            print("RK3 is not implemented, changes to Euler")
            self.num_prop._time_scheme = "euler"
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf

    def copy(self, pb):
        super().copy(pb)
        self.conv_interf = pb.conv_interf
        self.interp_type = pb.interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.dt * self.phy_prop.v,
                )

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                self.bulles.post(cells, i_int, ist)

                # Correction des flux cellules
                ind_flux_conv = [
                    im1,
                    i0,
                    ip1,
                    ip2,
                    ip3,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
                flux_conv[ind_flux_conv] = rhocpT_u[1:]
                # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
            I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self._compute_convection_flux(self.rho_cp_a * self.T, self.bulles, debug)
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )

        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff
        )
        self._echange_flux()
        drhocpTdt = -integrale_vol_div(
            self.flux_conv, dx
        ) + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        self.T = (self.T * self.rho_cp_a + self.dt * drhocpTdt) / rho_cp_a_np1

    @property
    def name_cas(self):
        return "ESP sans corr. diff."  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


# TODO: à supprimer, n'a aucun intérêt
class ProblemDiscontinuEcomme3D(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if num_prop.time_scheme == "rk3":
            print("RK3 is not implemented, changes to Euler")
            self.num_prop._time_scheme = "euler"
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf

    def copy(self, pb):
        super().copy(pb)
        self.conv_interf = pb.conv_interf
        self.interp_type = pb.interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.dt * self.phy_prop.v,
                    time_integral="CN",
                )

                # Correction des cellules i0 - 1 à i0 + 1 inclue

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                self.bulles.post(cells, i_int, ist)

                # Correction des flux cellules
                ind_flux_conv = [
                    ip1,
                ]
                flux_conv[ind_flux_conv] = rhocpT_u[3]
                # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
            I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(self.T, self.bulles, debug)
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )

        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff
        )
        self._echange_flux()
        drhocpTdt = -integrale_vol_div(
            self.flux_conv, dx
        ) + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        self.T = (self.T * self.rho_cp_a + self.dt * drhocpTdt) / rho_cp_a_np1

    @property
    def name_cas(self):
        return "ESP 3D"  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


# Pas besoin d'equivalent, il suffit de donner le bon num_prop a un StateProblemDiscontinuEsansq
class ProblemDiscontinuEcomme3D_ghost(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Résolution en énergie.
    Cette classe résout le problème en 3 étapes :

        - on calcule le nouveau T comme avant (avec un stencil de 1 à proximité des interfaces par simplicité)
        - on calcule précisemment T1 et T2 ansi que les bons flux aux faces, on met à jour T
        - on met à jour T_i et lda_grad_T_i

    Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
    continuité de lda_grad_T et T à l'interface.

    Comme on veut faire comme en 3D, on n'utilise pas l'intégrale en temps de rho cp à travers la face, on fait
    du CN.
    En plus de cela, on ne corrige pas le flux qui est calculé pour les faces qui sont monophasiques. Donc cela revient
    à utiliser la température diphasique de la maille dans l'interpolation.
    Pour les cellules a proximité de l'interface, au lieu de faire l'interpolation qu'on avait avant, on fait une interpolation
    quick avec des valeurs ghost à la place de Ti (et de Tim1 pour Tip12).
    Les ghosts sont interpolés à partir de l'interface seulement. Cela dit, ça ne change rien de les interpoler à
    partir d'ailleurs, il faut juste que ce soit pertinent.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if num_prop.time_scheme == "rk3":
            print("RK3 is not implemented, changes to Euler")
            self.num_prop._time_scheme = "euler"
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf
        if not conv_interf.endswith("ghost"):
            raise (Exception("Le schema conv_interf doit etre du type ghost."))

    def copy(self, pb):
        super().copy(pb)
        self.conv_interf = pb.conv_interf
        self.interp_type = pb.interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.dt * self.phy_prop.v,
                    time_integral="CN",
                )

                # Correction des cellules i0 - 1 à i0 + 1 inclue

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v

                self.bulles.post(cells, i_int, ist)

                # Correction des flux cellules
                ind_flux_conv = [
                    i,
                    ip1,
                    ip2,
                ]  # on corrige les flux de i-1/2 a i+3/2 (en QUICK ça va jusqu'a 3/2)
                flux_conv[ind_flux_conv] = rhocpT_u[2:-1]
                # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
            I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(self.T, self.bulles, debug)
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )

        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff
        )
        self._echange_flux()
        drhocpTdt = -integrale_vol_div(
            self.flux_conv, dx
        ) + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        self.T = (self.T * self.rho_cp_a + self.dt * drhocpTdt) / rho_cp_a_np1

    def _rk3_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        T_int = self.T.copy()
        markers_int = self.bulles.copy()
        markers_int_kp1 = self.bulles.copy()
        K = 0.0
        coeff_h = np.array([1.0 / 3, 5.0 / 12, 1.0 / 4])
        coeff_dTdtm1 = np.array([0.0, -5.0 / 9, -153.0 / 128])
        coeff_dTdt = np.array([1.0, 4.0 / 9, 15.0 / 32])
        for step, h in enumerate(coeff_h):
            I_f = markers_int.indicatrice_liquide(self.num_prop.x_f)
            I = markers_int.indicatrice_liquide(self.num_prop.x)
            rho_cp_f = I_f * self.phy_prop.rho_cp1 + (1.0 - I_f) * self.phy_prop.rho_cp2
            rho_cp_a = I * self.phy_prop.rho_cp1 + (1.0 - I) * self.phy_prop.rho_cp2

            markers_int_kp1.shift(self.phy_prop.v * h * self.dt)
            I_kp1 = markers_int_kp1.indicatrice_liquide(self.num_prop.x)
            rho_cp_a_kp1 = (
                I_kp1 * self.phy_prop.rho_cp1 + (1.0 - I_kp1) * self.phy_prop.rho_cp2
            )

            flux_conv = rho_cp_f * self._compute_convection_flux(T_int, markers_int, debug)
            flux_diff = self._compute_diffusion_flux(
                T_int, markers_int, bool_debug, debug
            )

            self._corrige_flux_coeff_interface(T_int, markers_int, flux_conv, flux_diff)
            self._echange_flux()
            flux_diff.perio()
            flux_conv.perio()
            drhocpTdt = -integrale_vol_div(
                flux_conv, dx
            ) + self.phy_prop.diff * integrale_vol_div(flux_diff, dx)
            # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
            K = K * coeff_dTdtm1[step] + drhocpTdt
            T_int = (
                T_int * rho_cp_a + self.dt * h * K / coeff_dTdt[step]
            ) / rho_cp_a_kp1
            markers_int.shift(self.phy_prop.v * h * self.dt)

        self.T = T_int

    @property
    def name_cas(self):
        return "ESP 3D ghost, CN"  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


# Pas besoin d'equivalent, il suffit de donner le bon num_prop a un StateProblemDiscontinuEsansq
class ProblemDiscontinuEcomme3D_ghost_exactSf(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Résolution en énergie.
    Cette classe résout le problème en 3 étapes :

        - on calcule le nouveau T comme avant (avec un stencil de 1 à proximité des interfaces par simplicité)
        - on calcule précisemment T1 et T2 ansi que les bons flux aux faces, on met à jour T
        - on met à jour T_i et lda_grad_T_i

    Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
    continuité de lda_grad_T et T à l'interface.

    Comme on veut faire comme en 3D, on n'utilise pas l'intégrale en temps de rho cp à travers la face, on fait
    du CN.
    En plus de cela, on ne corrige pas le flux qui est calculé pour les faces qui sont monophasiques. Donc cela revient
    à utiliser la température diphasique de la maille dans l'interpolation.
    Pour les cellules a proximité de l'interface, au lieu de faire l'interpolation qu'on avait avant, on fait une interpolation
    quick avec des valeurs ghost à la place de Ti (et de Tim1 pour Tip12).
    Les ghosts sont interpolés à partir de l'interface seulement. Cela dit, ça ne change rien de les interpoler à
    partir d'ailleurs, il faut juste que ce soit pertinent.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if num_prop.time_scheme == "rk3":
            print("RK3 is not implemented, changes to Euler")
            self.num_prop._time_scheme = "euler"
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf
        if not conv_interf.endswith("ghost"):
            raise (Exception("Le schema conv_interf doit etre du type ghost."))

    def copy(self, pb):
        super().copy(pb)
        self.conv_interf = pb.conv_interf
        self.interp_type = pb.interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.dt * self.phy_prop.v,
                    time_integral="exact",
                )

                # Correction des cellules i0 - 1 à i0 + 1 inclue

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v

                self.bulles.post(cells, i_int, ist)

                # Correction des flux cellules
                ind_flux_conv = [
                    i,
                    ip1,
                    ip2,
                ]  # on corrige les flux de i-1/2 a i+3/2 (en QUICK ça va jusqu'a 3/2)
                flux_conv[ind_flux_conv] = rhocpT_u[2:-1]
                # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
            I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(self.T, self.bulles, debug)
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )

        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff
        )
        self._echange_flux()
        drhocpTdt = -integrale_vol_div(
            self.flux_conv, dx
        ) + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        self.T = (self.T * self.rho_cp_a + self.dt * drhocpTdt) / rho_cp_a_np1

    @property
    def name_cas(self):
        return "ESP 3D ghost, exact Sf"  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


# Pas besoin d'equivalent, il suffit de donner le bon num_prop a un StateProblemDiscontinuE
class ProblemDiscontinuEcomme3Davecq_ghost(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Résolution en énergie.
    Cette classe résout le problème en 3 étapes :

        - on calcule le nouveau T comme avant (avec un stencil de 1 à proximité des interfaces par simplicité)
        - on calcule précisemment T1 et T2 ansi que les bons flux aux faces, on met à jour T
        - on met à jour T_i et lda_grad_T_i

    Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
    continuité de lda_grad_T et T à l'interface.

    Comme on veut faire comme en 3D, on n'utilise pas l'intégrale en temps de rho cp à travers la face, on fait
    du CN.
    En plus de cela, on ne corrige pas le flux qui est calculé pour les faces qui sont monophasiques. Donc cela revient
    à utiliser la température diphasique de la maille dans l'interpolation.
    Pour les cellules a proximité de l'interface, au lieu de faire l'interpolation qu'on avait avant, on fait une interpolation
    quick avec des valeurs ghost à la place de Ti (et de Tim1 pour Tip12).
    Les ghosts sont interpolés à partir de l'interface seulement. Cela dit, ça ne change rien de les interpoler à
    partir d'ailleurs, il faut juste que ce soit pertinent.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if num_prop.time_scheme == "rk3":
            print("RK3 is not implemented, changes to Euler")
            self.num_prop._time_scheme = "euler"
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf
        if not conv_interf.endswith("ghost"):
            raise (Exception("Le schema conv_interf doit etre du type ghost."))

    def copy(self, pb):
        super().copy(pb)
        self.conv_interf = pb.conv_interf
        self.interp_type = pb.interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.dt * self.phy_prop.v,
                    time_integral="CN",
                )

                # Correction des cellules i0 - 1 à i0 + 1 inclue

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                lda_grad_T = cells.lda_f * cells.gradT

                self.bulles.post(cells, i_int, ist)

                # Correction des flux cellules
                ind_flux_conv = [
                    i0,
                    ip1,
                    ip2,
                ]  # on corrige les flux de i-1/2 a i+3/2 (en QUICK ça va jusqu'a 3/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
                flux_conv[ind_flux_conv] = rhocpT_u[2:-1]
                flux_diff[ind_flux_diff] = lda_grad_T[2:4]
                # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
            I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(self.T, self.bulles, debug)
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )

        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff
        )
        self._echange_flux()
        drhocpTdt = -integrale_vol_div(
            self.flux_conv, dx
        ) + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        self.T = (self.T * self.rho_cp_a + self.dt * drhocpTdt) / rho_cp_a_np1

    @property
    def name_cas(self):
        return "ESP 3D ghost, corr. diff."  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


# Pas besoin d'equivalent, il suffit de donner le bon num_prop a un StateProblemDiscontinuE
class ProblemDiscontinuEcomme3Davecq_I(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Résolution en énergie.
    Cette classe résout le problème en 3 étapes :

        - on calcule le nouveau T comme avant (avec un stencil de 1 à proximité des interfaces par simplicité)
        - on calcule précisemment T1 et T2 ansi que les bons flux aux faces, on met à jour T
        - on met à jour T_i et lda_grad_T_i

    Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
    continuité de lda_grad_T et T à l'interface.

    Comme on veut faire comme en 3D, on n'utilise pas l'intégrale en temps de rho cp à travers la face, on fait
    du CN.
    En plus de cela, on ne corrige pas le flux qui est calculé pour les faces qui sont monophasiques. Donc cela revient
    à utiliser la température diphasique de la maille dans l'interpolation.
    Pour les cellules a proximité de l'interface, au lieu de faire l'interpolation qu'on avait avant, on fait une interpolation
    quick avec des valeurs ghost à la place de Ti (et de Tim1 pour Tip12).
    Les ghosts sont interpolés à partir de l'interface seulement. Cela dit, ça ne change rien de les interpoler à
    partir d'ailleurs, il faut juste que ce soit pertinent.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if num_prop.time_scheme == "rk3":
            print("RK3 is not implemented, changes to Euler")
            self.num_prop._time_scheme = "euler"
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf
        if not conv_interf.endswith("ghost"):
            raise (Exception("Le schema conv_interf doit etre du type ghost."))

    def copy(self, pb):
        super().copy(pb)
        self.conv_interf = pb.conv_interf
        self.interp_type = pb.interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.dt * self.phy_prop.v,
                    time_integral="CN",
                )
                # Correction des cellules i0 - 1 à i0 + 1 inclue

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                lda_grad_T_I = cells.lda_gradTi
                # lda_grad_T = cells.lda_f * cells.gradT

                self.bulles.post(cells, i_int, ist)

                # Correction des flux cellules
                ind_flux_conv = [
                    i0,
                    ip1,
                    ip2,
                ]  # on corrige les flux de i-1/2 a i+3/2 (en QUICK ça va jusqu'a 3/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
                flux_conv[ind_flux_conv] = rhocpT_u[2:-1]
                flux_diff[ind_flux_diff] = lda_grad_T_I
                # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
            I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(self.T, self.bulles, debug)
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )

        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff
        )
        self._echange_flux()
        drhocpTdt = -integrale_vol_div(
            self.flux_conv, dx
        ) + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        self.T = (self.T * self.rho_cp_a + self.dt * drhocpTdt) / rho_cp_a_np1

    @property
    def name_cas(self):
        return "ESP 3D ghost, CN, corr. diff. qI"  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


# Pas besoin d'equivalent, il suffit de donner le bon num_prop a un StateProblemDiscontinuE
class ProblemDiscontinuEcomme3D_ghost_avecq_I_exactSf(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Résolution en énergie.
    Cette classe résout le problème en 3 étapes :

        - on calcule le nouveau T comme avant (avec un stencil de 1 à proximité des interfaces par simplicité)
        - on calcule précisemment T1 et T2 ansi que les bons flux aux faces, on met à jour T
        - on met à jour T_i et lda_grad_T_i

    Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
    continuité de lda_grad_T et T à l'interface.

    Comme on veut faire comme en 3D, on n'utilise pas l'intégrale en temps de rho cp à travers la face, on fait
    du CN.
    En plus de cela, on ne corrige pas le flux qui est calculé pour les faces qui sont monophasiques. Donc cela revient
    à utiliser la température diphasique de la maille dans l'interpolation.
    Pour les cellules a proximité de l'interface, au lieu de faire l'interpolation qu'on avait avant, on fait une interpolation
    quick avec des valeurs ghost à la place de Ti (et de Tim1 pour Tip12).
    Les ghosts sont interpolés à partir de l'interface seulement. Cela dit, ça ne change rien de les interpoler à
    partir d'ailleurs, il faut juste que ce soit pertinent.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if num_prop.time_scheme == "rk3":
            print("RK3 is not implemented, changes to Euler")
            self.num_prop._time_scheme = "euler"
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf
        if not conv_interf.endswith("ghost"):
            raise (Exception("Le schema conv_interf doit etre du type ghost."))

    def copy(self, pb):
        super().copy(pb)
        self.conv_interf = pb.conv_interf
        self.interp_type = pb.interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.dt * self.phy_prop.v,
                    time_integral="exact",
                )

                # Correction des cellules i0 - 1 à i0 + 1 inclue

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                lda_grad_T_I = cells.lda_gradTi
                # lda_grad_T = cells.lda_f * cells.gradT
                self.bulles.post(cells, i_int, ist)

                # Correction des flux cellules
                ind_flux_conv = [
                    i0,
                    ip1,
                    ip2,
                ]  # on corrige les flux de i-1/2 a i+3/2 (en QUICK ça va jusqu'a 3/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
                flux_conv[ind_flux_conv] = rhocpT_u[2:-1]
                flux_diff[ind_flux_diff] = lda_grad_T_I
                # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
            I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(self.T, self.bulles, debug)
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )

        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff
        )
        self._echange_flux()
        drhocpTdt = -integrale_vol_div(
            self.flux_conv, dx
        ) + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        self.T = (self.T * self.rho_cp_a + self.dt * drhocpTdt) / rho_cp_a_np1

    @property
    def name_cas(self):
        return "ESP 3D ghost, exact Sf, corr. diff. qI"  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


class ProblemDiscontinuT(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Cette classe résout le problème en 3 étapes :

        - on calcule le nouveau T comme avant (avec un stencil de 1 à proximité des interfaces par simplicité)
        - on calcule précisemment T1 et T2 ansi que les bons flux aux faces, on met à jour T
        - on met à jour T_i et lda_grad_T_i

    Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
    continuité de lda_grad_T et T à l'interface.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf

    def copy(self, pb):
        super().copy(pb)
        self.conv_interf = pb.conv_interf
        self.interp_type = pb.interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4

        Args:
            flux_conv:
            flux_diff:
            coeff_diff:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.phy_prop.v * self.dt,
                )

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                T_u = cells.T_f * self.phy_prop.v
                lda_grad_T = cells.lda_f * cells.gradT

                self.bulles.post(cells, i_int, ist)

                # Correction des cellules
                ind_flux_conv = [
                    im1,
                    i0,
                    ip1,
                    ip2,
                    ip3,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
                flux_conv[ind_flux_conv] = T_u[1:]
                flux_diff[ind_flux_diff] = lda_grad_T[2:4]
                # Tnp1 = Tn + dt (- int_S_T_u + 1/rhocp * int_S_lda_grad_T)

    @property
    def name_cas(self):
        return "TSP"  # + self.interp_type.replace('_', '-') + ', ' + self.conv_interf.replace('_', '-')

    def _euler_timestep(self, debug=None, bool_debug=False):
        """
        Dans ce euler timestep on calcule 1/rhocp^{n+1/2} = (1/rhocp^{n} + 1/rhocp^{n+1})/2.
        Args:
            debug:
            bool_debug:

        Returns:

        """
        dx = self.num_prop.dx
        bulles = self.bulles.copy()
        bulles.shift(self.phy_prop.v * self.dt)
        Inp1 = bulles.indicatrice_liquide(self.num_prop.x)
        self.flux_conv = self._compute_convection_flux(self.T, self.bulles, debug)
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )
        rho_cp_inv_h = 1.0 / self.rho_cp_h
        rho_cp_inv_h_np1 = (
            Inp1 / self.phy_prop.rho_cp1 + (1.0 - Inp1) / self.phy_prop.rho_cp2
        )
        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff
        )
        self._echange_flux()
        dTdt = -integrale_vol_div(self.flux_conv, dx) + self.phy_prop.diff * (
            rho_cp_inv_h + rho_cp_inv_h_np1
        ) / 2.0 * integrale_vol_div(self.flux_diff, dx)
        self.T += self.dt * dTdt


class ProblemDiscontinuSautdTdt(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        deb=False,
        delta_diff=1.0,
        delta_conv=1.0,
        int_Ti=1.0,
        delta_conv2=0.0,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        self.deb = deb
        self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        self.delta_diff = delta_diff
        self.delta_conv = delta_conv
        self.delta_conv2 = delta_conv2
        self.int_Ti = int_Ti
        if num_prop.time_scheme != "euler":
            print(
                "%s time scheme not implemented, falling back to euler time scheme"
                % num_prop.time_scheme
            )
            self.num_prop._time_scheme = "euler"

    def copy(self, pb):
        super().copy(pb)
        self.interp_type = pb.interp_type
        self.deb = pb.deb
        self.T_old = pb.T.copy()
        self.delta_diff = pb.delta_diff
        self.delta_conv = pb.delta_conv
        self.delta_conv2 = pb.delta_conv2
        self.int_Ti = pb.int_Ti

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_interface(self):
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(self.bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(self.T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    self.T_old[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.num_prop.schema,
                )

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                rhocp_T_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                int_div_rhocp_T_u = integrale_vol_div(rhocp_T_u, dx)
                lda_grad_T = cells.lda_f * cells.gradT
                int_div_lda_grad_T = integrale_vol_div(lda_grad_T, dx)

                # propre à cette version particulière, on calule le saut de dT/dt à l'interface et int_S_Ti_v_n2_dS
                delta0 = (
                    self.delta_diff
                    * (
                        cells.grad_lda_gradT_n_d / rhocpd
                        - cells.grad_lda_gradT_n_g / rhocpg
                    )
                    - self.delta_conv
                    * cells.lda_gradTi
                    * (1 / ldad - 1 / ldag)
                    * self.phy_prop.v
                    - self.delta_conv2
                    * cells.lda_gradTi
                    * (1 / ldad + 1 / ldag)
                    * self.phy_prop.v
                )

                # pour rappel, ici on a divisé l'intégrale par le volume de la cellule comme toutes les intégrales
                # le signe - vient du fait qu'on calcule pour V2, avec le vecteur normal à I qui est donc dirigé en -x
                int_S_Ti_v_n2_dS_0 = (
                    -self.int_Ti * cells.Ti * self.phy_prop.v / self.num_prop.dx
                )

                delta = np.array([0.0, 0.0, delta0, 0.0, 0.0])
                int_S_Ti_v_n2_dS = np.array([0.0, 0.0, int_S_Ti_v_n2_dS_0, 0.0, 0.0])

                # Correction des cellules
                ind_to_change = [im2, im1, i0, ip1, ip2]
                ind_flux_conv = [
                    im1,
                    i0,
                    ip1,
                    ip2,
                    ip3,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
                self.flux_conv[ind_flux_conv] = (
                    rhocp_T_u[1:] / self.rho_cp_a[ind_flux_conv]
                )
                self.flux_diff[ind_flux_diff] = lda_grad_T[2:4]
                if self.deb:
                    print(
                        "delta conv : ",
                        cells.lda_gradTi * (1 / ldad - 1 / ldag) * self.phy_prop.v,
                    )
                    print(
                        "delta cond : ",
                        (
                            cells.grad_lda_gradT_n_d / rhocpd
                            - cells.grad_lda_gradT_n_g / rhocpg
                        ),
                    )
                    print("delta * ... : ", delta0 * ad * (rhocpd - self.rho_cp_a[i0]))
                    print("int_I... : ", (rhocpd - rhocpg) * int_S_Ti_v_n2_dS_0)
                    print(
                        "int_I... + delta * ... : ",
                        (rhocpd - rhocpg) * int_S_Ti_v_n2_dS_0
                        + delta0 * ad * (rhocpd - self.rho_cp_a[i0]),
                    )
                    print(
                        "(int_I... + delta * ...)/rho_cp_a : ",
                        (
                            (rhocpd - rhocpg) * int_S_Ti_v_n2_dS_0
                            + delta0 * ad * (rhocpd - self.rho_cp_a[i0])
                        )
                        / self.rho_cp_a[i0],
                    )
                    print(
                        "int_div_lda_grad_T/rho_cp_a : ",
                        int_div_lda_grad_T[2] / self.rho_cp_a[i0],
                    )
                    print(
                        "int_div_rhocp_T_u/rho_cp_a : ",
                        int_div_rhocp_T_u[2] / self.rho_cp_a[i0],
                    )

                # on écrit l'équation en température, ça me semble peut être mieux ?
                # Tnp1 = Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T
                #                 - delta * I2 * (rhocp2 - rhocpa) - [rhocp] * int_S_Ti_v_n2_dS) / rhocpa
                self.T[ind_to_change] = (
                    self.T_old[ind_to_change]
                    + self.dt
                    * (
                        -int_div_rhocp_T_u
                        + self.phy_prop.diff * int_div_lda_grad_T
                        - delta * ad * (rhocpd - self.rho_cp_a[i0])
                        - (rhocpd - rhocpg) * int_S_Ti_v_n2_dS
                    )
                    / self.rho_cp_a[ind_to_change]
                )
        self.T_old = self.T.copy()

    def _euler_timestep(self, debug=None, bool_debug=False):
        super()._euler_timestep(debug=debug, bool_debug=bool_debug)
        self._corrige_interface()

    @property
    def name_cas(self):
        return "SEFC "


class ProblemDiscontinuSepIntT(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf
        # ce tableau sert à ajouter des contributions localement dans les cellules traversées par l'interface.
        # Il est très creux.
        self.ind_interf = np.zeros_like(self.T)

    def copy(self, pb):
        super().copy(pb)
        self.interp_type = pb.interp_type
        self.conv_interf = pb.conv_interf
        self.ind_interf = pb.ind_interf.copy()

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(markers=markers, phy_prop=self.phy_prop)
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff = args

        dx = self.num_prop.dx
        self.ind_interf = np.zeros_like(T)

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.phy_prop.v * self.dt,
                )

                # post-traitements

                self.bulles.post(cells, i_int, ist)

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                # rhocp_T_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                inv_rhocp_f_lda_f_grad_T = cells.lda_f * cells.gradT * cells.inv_rhocp_f

                # Correction des cellules
                # ind_to_change = [im2, im1, i0, ip1, ip2]
                ind_flux_conv = [
                    im1,
                    i0,
                    ip1,
                    ip2,
                    ip3,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
                flux_conv[ind_flux_conv] = cells.T_f[1:] * self.phy_prop.v
                flux_diff[ind_flux_diff] = inv_rhocp_f_lda_f_grad_T[2:4]
                self.ind_interf[i0] = (
                    (1.0 / rhocpg - 1.0 / rhocpd) * cells.lda_gradTi * cells.coeff_d
                )
                self.ind_interf[ip1] = (
                    (1.0 / rhocpg - 1.0 / rhocpd)
                    * cells.lda_gradTi
                    * (1.0 - cells.coeff_d)
                )
                # on écrit l'équation en température, et en energie
                # Tnp1 = Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T
                #                 - delta * I2 * (rhocp2 - rhocpa) - [rhocp] * int_S_Ti_v_n2_dS) / rhocpa

    def _euler_timestep(self, debug=None, bool_debug=False):
        self.flux_conv = (
            interpolate(self.T, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        self.flux_diff = interpolate(self.Lda_h, I=self.I, schema="center_h") * grad(
            self.T, self.num_prop.dx
        )
        # Attention, l'interpolation suivante n'est valide que dans le cas de deux cellules monophasiques adjacentes
        # elle nécessite impérativement une correction aux faces mitoyennes de l'interface.
        flux_diff = self.flux_diff / interpolate(
            self.rho_cp_a, I=self.I, schema="center_h"
        )
        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, flux_diff
        )
        self._echange_flux()
        flux_diff.perio()
        int_div_T_u = integrale_vol_div(self.flux_conv, self.num_prop.dx)
        int_inv_rhocpf_div_ldaf_grad_T = integrale_vol_div(flux_diff, self.num_prop.dx)

        # if (debug is not None) and bool_debug:
        #     debug.plot(self.num_prop.x, 1. / self.rho_cp_h, label='rho_cp_inv_h, time = %f' % self.time)
        #     debug.plot(self.num_prop.x, int_div_lda_grad_T, label='div_lda_grad_T, time = %f' % self.time)
        #     debug.xticks(self.num_prop.x_f)
        #     debug.grid(which='major')
        #     maxi = max(np.max(int_div_lda_grad_T), np.max(1. / self.rho_cp_h))
        #     mini = min(np.min(int_div_lda_grad_T), np.min(1. / self.rho_cp_h))
        #     for markers in self.bulles():
        #         debug.plot([markers[0]] * 2, [mini, maxi], '--')
        #         debug.plot([markers[1]] * 2, [mini, maxi], '--')
        #     debug.legend()
        self.T += self.dt * (
            -int_div_T_u
            + self.phy_prop.diff * int_inv_rhocpf_div_ldaf_grad_T
            + self.phy_prop.diff / self.num_prop.dx * self.ind_interf
        )

    @property
    def name_cas(self):
        return "TSV "


# Pas validé, verifier la formule
class ProblemDiscontinuECorrige(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        """
        Ici on corrige l'approximation :maht:`\\overline{h} = rhoCp_a * \\overline{T}` grace au DL de la température
        à l'ordre 1 :
        :math:`T(x) = T_I + \\left.\\nabla T\\right|_{I^{+/-}} \\cdot (\\underline{x} - \\underline{x}_I)`
        Cette classe est expérimentale, les réultats ne sont pas validés.

        Args:
            T0: temperature function
            markers (BulleTemperature|np.ndarray):
            num_prop (NumericalProperties):
            phy_prop (PhysicalProperties):
            interp_type (str): interpolation of the interfacial values. Default is Ti
                Ex: Ti, Ti2, Ti3, Tivol, energie_temperature, ...
            conv_interf (str): interpolation type convection scheme (upwind or quick). Default is self.num_prop.schema
        """
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf
        # ce tableau sert à ajouter des contributions localement dans les cellules traversées par l'interface.
        # Il est très creux.
        self.ind_interf = np.zeros_like(self.T)

    def copy(self, pb):
        super().copy(pb)
        self.interp_type = pb.interp_type
        self.conv_interf = pb.conv_interf
        self.ind_interf = pb.ind_interf.copy()

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(markers=markers, phy_prop=self.phy_prop)
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_interface(self):
        """
        Dans  cette approche on doit non seulement corriger les flux :math:`\rho C_p T_f` et :math:`\\lambda \\nabla T_f`
        mais aussi le produit :math:`\\overline{h} = {\\rho C_p}_a \\overline{T} + (\\rho C_{p1} - \\rho C_{p2})\\alpha_1\\alpha_2[\\left.\\nabla T\\right|_{I1} \\cdot (\\underline{x}_1 - \\underline{x}_I) - \\left.\\nabla T\\right|_{I2} \\cdot (\\underline{x}_2 - \\underline{x}_I)`

        Returns:
            Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
            jour T_old
        """
        dx = self.num_prop.dx
        self.ind_interf = np.zeros_like(self.T)

        for i_int, (i1, i2) in enumerate(self.bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                    sign = -1.0
                else:
                    from_liqu_to_vap = False
                    sign = 1.0
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(self.T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                a1 = self.I[i0]
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    self.T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.phy_prop.v * self.dt,
                )

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                rhocp_T_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                lda_gradT = cells.lda_f * cells.gradT

                # Correction des cellules
                # ind_to_change = [im2, im1, i0, ip1, ip2]
                ind_flux_conv = [
                    im1,
                    i0,
                    ip1,
                    ip2,
                    ip3,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement (i-1/2 et i+1/2)
                self.flux_conv[ind_flux_conv] = rhocp_T_u[1:]
                self.flux_diff[ind_flux_diff] = lda_gradT[2:4]
                self.ind_interf[i0] = (
                    sign
                    * (self.phy_prop.rho_cp1 - self.phy_prop.rho_cp2)
                    * cells.lda_gradTi
                    * self.phy_prop.v
                    * (
                        ag * ad * (1.0 / self.phy_prop.lda1 - 1.0 / self.phy_prop.lda2)
                        - a1**2 / (2.0 * self.phy_prop.lda1)
                        + (1 - a1) ** 2 / (2 * self.phy_prop.lda2)
                    )
                )
                # self.ind_interf[i0] = 0.
                # on écrit l'équation en energie et on fait un DL a l'odre 1 sur la température de chaque côté pour
                # corriger le produit :
                # rhocp_a(np1)T(np1) = rhocp_a(n)T(n) + dt/dV (- int_S_rho_cp_T_u + int_S_lda_grad_T) + dt*reste

    def _euler_timestep(self, debug=None, bool_debug=False):
        bulles = self.bulles.copy()
        bulles.shift(self.phy_prop.v * self.dt)
        Inp1 = bulles.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = self.phy_prop.rho_cp1 * Inp1 + self.phy_prop.rho_cp2 * (1 - Inp1)
        self.flux_conv = (
            interpolate(self.rho_cp_a * self.T, I=self.I, schema=self.num_prop.schema)
            * self.phy_prop.v
        )
        self.flux_diff = interpolate(
            self.Lda_h, I=self.I, schema=self.num_prop.schema
        ) * grad(self.T, self.num_prop.dx)
        self._corrige_interface()
        self._echange_flux()
        int_div_rho_cp_T_u = integrale_vol_div(self.flux_conv, self.num_prop.dx)
        int_div_lda_grad_T = integrale_vol_div(self.flux_diff, self.num_prop.dx)

        self.T = (
            self.T * self.rho_cp_a
            + self.dt
            * (
                -int_div_rho_cp_T_u
                + self.phy_prop.diff * int_div_lda_grad_T
                + self.phy_prop.diff * self.ind_interf
            )
        ) / rho_cp_a_np1

    @property
    def name_cas(self):
        return "EFCPC "


class ProblemRhoCpDiscontinuE(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Résolution en énergie.
    Dans cette résolution on ne corrige pas la température et le gradient pour les flux, on corrige seulement les 
    valeurs des propriétés discontinues à l'interface.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        # if self.num_prop.schema != 'upwind':
        #     raise Exception('Cette version ne marche que pour un schéma upwind')
        if num_prop.time_scheme == "rk3":
            print("RK3 is not implemented, changes to Euler")
            self.num_prop._time_scheme = "euler"
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "Ti"
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf

    def copy(self, pb):
        super().copy(pb)
        self.conv_interf = pb.conv_interf
        self.interp_type = pb.interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff, T_f = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                # Ici on ne prend pas en compte la température, seule la correction des coefficients nous intéresse
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    np.empty((7,)),
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type=self.interp_type,
                    schema_conv=self.conv_interf,
                    vdt=self.dt * self.phy_prop.v,
                )

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation

                T_f_ = T_f[[im2, im1, i0, ip1, ip2, ip3]]
                rhocpT_u = cells.rhocp_f * T_f_ * self.phy_prop.v

                self.bulles.post(cells, i_int, ist)

                # Correction des flux cellules
                ind_flux_conv = [
                    im1,
                    i0,
                    ip1,
                    ip2,
                    ip3,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
                flux_conv[ind_flux_conv] = rhocpT_u[1:]

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
            I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self._compute_convection_flux(self.rho_cp_a * self.T, self.bulles, debug)
        T_f = self._compute_convection_flux(self.T, self.bulles, debug)
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )
        self._corrige_flux_coeff_interface(
            self.T, self.bulles, self.flux_conv, self.flux_diff, T_f
        )
        self._echange_flux()
        drhocpTdt = -integrale_vol_div(
            self.flux_conv, dx
        ) + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        self.T = (self.T * self.rho_cp_a + self.dt * drhocpTdt) / rho_cp_a_np1

    @property
    def name_cas(self):
        return "ESPconvOFdiff"  # + self.interp_type.replace('_', '-')


# Pas validé, vérifier la formule
class ProblemDiscontinuCoupleConserv(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Attention cette classe n'est probablement pas finalisée

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(
        self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None, **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        self.T_old = self.T.copy()
        self.flux_conv_ener = self.flux_conv.copy()
        self.h = self.rho_cp_a * self.T
        self.h_old = self.h.copy()
        if interp_type is None:
            self.interp_type = "energie_temperature"
        else:
            self.interp_type = interp_type
        print(self.interp_type)

    def copy(self, pb):
        super().copy(pb)
        self.interp_type = pb.interp_type
        self.T_old = pb.T_old.copy()
        self.flux_conv_ener = pb.flux_conv_ener.copy()
        self.h = pb.h.copy()
        self.h_old = pb.h_old.copy()

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_interface(self):
        """
        Dans cette approche on calclue Ti et lda_gradTi à partir du système énergie température

        Returns:
            Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
            jour T_old
        """
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(self.bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(self.T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells = CellsInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    self.T_old[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    interp_type="energie_temperature",
                    schema_conv="quick",
                    vdt=self.phy_prop.v * self.dt,
                )
                cells.compute_from_h_T(self.h_old[i0], self.T_old[i0])
                cells.compute_T_f_gradT_f_quick()

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                rhocp_T_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                int_div_rhocp_T_u = integrale_vol_div(rhocp_T_u, dx)
                lda_grad_T = cells.lda_f * cells.gradT
                int_div_lda_grad_T = integrale_vol_div(lda_grad_T, dx)

                # propre à cette version particulière, on calule le saut de dT/dt à l'interface et int_S_Ti_v_n2_dS
                delta0 = (
                    cells.grad_lda_gradT_n_d / rhocpd
                    - cells.grad_lda_gradT_n_g / rhocpg
                ) - cells.lda_gradTi * (1 / ldad - 1 / ldag) * self.phy_prop.v

                # pour rappel, ici on a divisé l'intégrale par le volume de la cellule comme toutes les intégrales
                # le signe - vient du fait qu'on calcule pour V2, avec le vecteur normal à I qui est donc dirigé en -x
                int_S_Ti_v_n2_dS_0 = -cells.Ti * self.phy_prop.v / self.num_prop.dx

                delta = np.array([0.0, 0.0, delta0, 0.0, 0.0])
                int_S_Ti_v_n2_dS = np.array([0.0, 0.0, int_S_Ti_v_n2_dS_0, 0.0, 0.0])

                # Correction des cellules
                ind_to_change = [im2, im1, i0, ip1, ip2]
                # ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                ind_flux_conv = [
                    im1,
                    i0,
                    ip1,
                    ip2,
                    ip3,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
                ind_flux_diff = [
                    i0,
                    ip1,
                ]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
                self.flux_conv[ind_flux_conv] = cells.T_f[1:] * self.phy_prop.v
                self.flux_conv_ener[ind_flux_conv] = rhocp_T_u[1:]
                self.flux_diff[ind_flux_diff] = lda_grad_T[2:4]
                # on écrit l'équation en température, et en energie
                # Tnp1 = Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T
                #                 - delta * I2 * (rhocp2 - rhocpa) - [rhocp] * int_S_Ti_v_n2_dS) / rhocpa
                self.T[ind_to_change] = (
                    self.T_old[ind_to_change]
                    + self.dt
                    * (
                        -int_div_rhocp_T_u
                        + self.phy_prop.diff * int_div_lda_grad_T
                        - delta * ad * (rhocpd - self.rho_cp_a[i0])
                        - (rhocpd - rhocpg) * int_S_Ti_v_n2_dS
                    )
                    / self.rho_cp_a[ind_to_change]
                )
                self.h[ind_to_change] = self.h_old[ind_to_change] + self.dt * (
                    -int_div_rhocp_T_u + self.phy_prop.diff * int_div_lda_grad_T
                )
        self.T_old = self.T.copy()
        self.h_old = self.h.copy()

    def _euler_timestep(self, debug=None, bool_debug=False):
        self.flux_conv = (
            interpolate(self.T, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        self.flux_conv_ener = (
            interpolate(self.h, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        int_div_T_u = integrale_vol_div(self.flux_conv, self.num_prop.dx)
        int_div_rho_cp_T_u = integrale_vol_div(self.flux_conv_ener, self.num_prop.dx)
        self.flux_diff = interpolate(
            self.Lda_h, I=self.I, schema=self.num_prop.schema
        ) * grad(self.T, self.num_prop.dx)
        int_div_lda_grad_T = integrale_vol_div(self.flux_diff, self.num_prop.dx)

        if (debug is not None) and bool_debug:
            debug.plot(
                self.num_prop.x,
                1.0 / self.rho_cp_h,
                label="rho_cp_inv_h, time = %f" % self.time,
            )
            debug.plot(
                self.num_prop.x,
                int_div_lda_grad_T,
                label="div_lda_grad_T, time = %f" % self.time,
            )
            debug.xticks(self.num_prop.x_f)
            debug.grid(which="major")
            maxi = max(np.max(int_div_lda_grad_T), np.max(1.0 / self.rho_cp_h))
            mini = min(np.min(int_div_lda_grad_T), np.min(1.0 / self.rho_cp_h))
            for markers in self.bulles():
                debug.plot([markers[0]] * 2, [mini, maxi], "--")
                debug.plot([markers[1]] * 2, [mini, maxi], "--")
            debug.legend()
        rho_cp_inv_h = 1.0 / self.rho_cp_h
        self.T += self.dt * (
            -int_div_T_u + self.phy_prop.diff * rho_cp_inv_h * int_div_lda_grad_T
        )
        self.h += self.dt * (
            -int_div_rho_cp_T_u + self.phy_prop.diff * int_div_lda_grad_T
        )

        self._corrige_interface()

    @property
    def name_cas(self):
        return "CL température saut dTdt "


# L'objectif était de résoudre le problème sans convection dans 5 cellules autour de l'interface
# dans un repère mobile avec l'interface, puis de réinterpoler sur le maillage fixe, mais ça ne
# marche pas a priori
class ProblemDiscontinuFT(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    def __init__(
        self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None, **kwargs
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
        if self.num_prop.schema != "upwind":
            raise Exception("Cette version ne marche que pour un schéma upwind")
        self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = "gradTi"
            print("interp type is :", self.interp_type)
        else:
            self.interp_type = interp_type

    def copy(self, pb):
        super().copy(pb)
        self.interp_type = pb.interp_type
        self.T_old = pb.T_old.copy()

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(
                markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(
                markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x
            )
        else:
            print(markers)
            raise NotImplementedError

    def _corrige_interface_ft(self):
        """
        Dans cette correction, on calcule l'évolution de la température dans des cellules qui suivent l'interface (donc
        sans convection). Ensuite on réinterpole sur la grille fixe des température.

        Returns:
            Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
            jour T_old
        """
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(self.bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(self.T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=from_liqu_to_vap
                )
                cells_ft = CellsSuiviInterface(
                    ldag,
                    ldad,
                    ag,
                    dx,
                    self.T_old[[im3, im2, im1, i0, ip1, ip2, ip3]],
                    rhocpg=rhocpg,
                    rhocpd=rhocpd,
                    vdt=self.dt * self.phy_prop.v,
                    interp_type=self.interp_type,
                )
                # On commence par interpoler Ti sur Tj avec TI et lda_gradTi
                # On calcule notre pas de temps avec lda_gradTj entre j et jp1 (à l'interface)
                # On interpole Tj sur la grille i

                # Correction des cellules
                ind_to_change = [im1, i0, ip1]
                ind_flux = [im1, i0, ip1, ip2]
                self.flux_conv[ind_flux] = np.nan
                self.flux_diff[ind_flux] = np.nan

                cells_ft.timestep(self.dt, self.phy_prop.diff)
                T_i_np1_interp = cells_ft.interp_T_from_j_to_i()
                self.T[ind_to_change] = T_i_np1_interp

                # post-traitements

                self.bulles.T[i_int, ist] = cells_ft.cells_fixe.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells_ft.cells_fixe.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells_ft.cells_fixe.Tg[-1]
                self.bulles.Td[i_int, ist] = cells_ft.cells_fixe.Td[0]
                self.bulles.cells[2 * i_int + ist] = cells_ft

        self.T_old = self.T.copy()

    def _euler_timestep(self, debug=None, bool_debug=False):
        super()._euler_timestep(debug=debug, bool_debug=bool_debug)
        self._corrige_interface_ft()

    @property
    def name_cas(self):
        return "TFF "  # température front-fitting
