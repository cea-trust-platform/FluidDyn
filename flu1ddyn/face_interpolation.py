"""
Different FaceInterpolation implementations.
"""


from abc import ABC, abstractmethod
import numpy as np
from flu1ddyn.cells_interface import InterfaceInterpolationBase
from flu1ddyn.interpolation_methods import (
    interpolate_from_center_to_face_quick,
)
import flu1ddyn.local_accelerated_interp_methods as interp


class FaceInterpolationBase(ABC):
    def __init__(
        self,
        vdt=0.0,
        time_integral="exact",
    ):
        self.interface_cells = None  # InterfaceInterpolationBase()
        self.rhocpg = 1.0
        self.rhocpd = 1.0
        self.time_integral = time_integral
        self.vdt = vdt
        self._rhocp_f = np.zeros(6)
        self._inv_rhocp_f = np.zeros(6)
        self._lda_f = np.zeros(6)
        self._T_f = np.empty((6,), dtype=np.float_)
        self._gradT_f = np.empty((6,), dtype=np.float_)
        self._name = "FaceInterpolationBase"
        # On fait tout de suite le calcul qui nous intéresse, il est nécessaire pour la suite

    @property
    def name(self):
        return self._name + ", " + self.time_integral

    def interpolate_on_faces(
        self, interface_cells: InterfaceInterpolationBase, rhocpg, rhocpd
    ):
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
            interpolates temperatures and gradients at the faces

        """
        self.interface_cells = interface_cells
        self.rhocpg = rhocpg
        self.rhocpd = rhocpd
        self._compute_Tf_gradTf()

    @abstractmethod
    def _compute_Tf_gradTf(self):
        raise NotImplementedError

    @property
    def dx(self):
        return self.interface_cells.dx

    @property
    def ad(self):
        return self.interface_cells.ad

    @property
    def ag(self):
        return self.interface_cells.ag

    @property
    def ldad(self):
        return self.interface_cells.ldad

    @property
    def ldag(self):
        return self.interface_cells.ldag

    @property
    def T_f(self):
        return self._T_f

    @property
    def gradT(self) -> np.ndarray((6,), dtype=float):
        return self._gradT_f

    @property
    def rhocp_f(self) -> np.ndarray((6,), dtype=float):
        self._rhocp_f[:3] = self.rhocpg
        self._rhocp_f[-3:] = self.rhocpd
        if self.time_integral == "exact":
            self._rhocp_f[3] = self._exactSf(self.rhocpg, self.rhocpd)
        elif self.time_integral == "CN":
            self._rhocp_f[3] = self._crank_nicolson(self.rhocpg, self.rhocpd)
        else:
            raise Exception(
                "L'attribut time_integral : %s n'est pas reconnu" % self.time_integral
            )
        return self._rhocp_f

    @property
    def coeff_d(self) -> float:
        if self.vdt > 0.0:
            return min(self.vdt, self.ad * self.dx) / self.vdt
        else:
            return 1.0

    def _exactSf(self, valg, vald):
        return self.coeff_d * vald + (1.0 - self.coeff_d) * valg

    def _crank_nicolson(self, valg, vald):
        if self.ad * self.dx > self.vdt:
            return vald
        else:
            return (valg + vald) / 2.0

    @property
    def inv_rhocp_f(self) -> np.ndarray((6,), dtype=float):
        self._inv_rhocp_f[:3] = 1.0 / self.rhocpg
        self._inv_rhocp_f[4:] = 1.0 / self.rhocpd
        self._inv_rhocp_f[3] = (
            self.coeff_d * 1.0 / self.rhocpd + (1.0 - self.coeff_d) * 1.0 / self.rhocpg
        )
        return self._inv_rhocp_f

    @property
    def lda_f(self) -> np.ndarray((6,), dtype=float):
        self._lda_f[:3] = self.ldag
        self._lda_f[3:] = self.ldad
        return self._lda_f

    @property
    def rhocpT_f(self) -> np.ndarray((6,), dtype=float):
        return self.rhocp_f * self.T_f


class FaceInterpolationUpwind(FaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "Upwind"

    def _compute_Tf_gradTf(self):
        Tim32, dTdxim32, _ = interp.upwind(
            self.interface_cells.Tg[1],
            self.interface_cells.Tg[2],
            -1.0 * self.dx,
            0.0 * self.dx,
        )
        Tim12, dTdxim12, _ = interp.upwind(
            self.interface_cells.Tg[-1],
            self.interface_cells.Tg[0],
            -1.0 * self.dx,
            0.0 * self.dx,
        )
        Tip12, dTdxip12, _ = interp.upwind(
            self.interface_cells.Td[0],
            self.interface_cells.Td[1],
            0.0 * self.dx,
            1.0 * self.dx,
        )
        Tip32, dTdxip32, _ = interp.upwind(
            self.interface_cells.Td[1],
            self.interface_cells.Td[2],
            1.0 * self.dx,
            2.0 * self.dx,
        )
        Tip52, dTdxip52, _ = interp.upwind(
            self.interface_cells.Td[2],
            self.interface_cells.Td[3],
            2.0 * self.dx,
            3.0 * self.dx,
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
        self._gradT_f[1] = np.nan  # dTdxim32
        self._gradT_f[2] = dTdxim12
        self._gradT_f[3] = dTdxip12
        self._gradT_f[4] = np.nan  # dTdxip32
        self._gradT_f[5] = np.nan  # dTdxip52


class FaceInterpolationDiphOnlyQuick(FaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "Downwind only, quick"

    def _compute_Tf_gradTf(self):
        Tip12, _, _ = interp.lagrange_amont_grad(
            self.interface_cells.Ti,
            self.interface_cells.Td[1],
            self.interface_cells.dTdxd,
            (0.5 - self.ad) * self.dx,
            1.0 * self.dx,
            0.5 * self.dx,
        )
        _, dTdxip12, _ = interp.lagrange_centre_grad(
            self.interface_cells.Td[2],
            self.interface_cells.Td[1],
            self.interface_cells.Ti,
            self.interface_cells.dTdxd,
            2.0 * self.dx,
            1.0 * self.dx,
            (0.5 - self.ad) * self.dx,
            0.5 * self.dx,
        )
        self._T_f[0] = np.nan
        self._T_f[1] = np.nan
        self._T_f[2] = np.nan  # self._T_dlg(0.)
        self._T_f[3] = Tip12  # self._T_dld(self.dx)
        self._T_f[4] = np.nan
        self._T_f[5] = np.nan
        self._gradT_f[0] = np.nan
        self._gradT_f[1] = np.nan  # dTdxim32
        self._gradT_f[2] = np.nan
        self._gradT_f[3] = dTdxip12
        self._gradT_f[4] = np.nan  # dTdxip32
        self._gradT_f[5] = np.nan  # dTdxip52


class FaceInterpolationQuick(FaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "Quick et amont 1"

    def _compute_Tf_gradTf(self):
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
        Tim32, dTdxim32, _ = interp.lagrange_amont(
            self.interface_cells.Tg[0],
            self.interface_cells.Tg[1],
            self.interface_cells.Tg[2],
            -2 * self.dx,
            -1.0 * self.dx,
            0.0 * self.dx,
            -0.5 * self.dx,
        )
        Tim12, _, _ = interp.lagrange_amont(
            self.interface_cells.Tg[1],
            self.interface_cells.Tg[2],
            self.interface_cells.Ti,
            -2.0 * self.dx,
            -1.0 * self.dx,
            (self.ag - 0.5) * self.dx,
            -0.5 * self.dx,
        )
        _, dTdxim12, _ = interp.lagrange_centre_grad(
            self.interface_cells.Tg[-3],
            self.interface_cells.Tg[-2],
            self.interface_cells.Ti,
            self.interface_cells.dTdxg,
            -2 * self.dx,
            -1 * self.dx,
            (self.ag - 0.5) * self.dx,
            -0.5 * self.dx,
        )
        Tip12, _, _ = interp.lagrange_amont_grad(
            self.interface_cells.Ti,
            self.interface_cells.Td[1],
            self.interface_cells.dTdxd,
            (0.5 - self.ad) * self.dx,
            1.0 * self.dx,
            0.5 * self.dx,
        )
        _, dTdxip12, _ = interp.lagrange_centre_grad(
            self.interface_cells.Td[2],
            self.interface_cells.Td[1],
            self.interface_cells.Ti,
            self.interface_cells.dTdxd,
            2.0 * self.dx,
            1.0 * self.dx,
            (0.5 - self.ad) * self.dx,
            0.5 * self.dx,
        )
        Tip32, dTdxip32, _ = interp.lagrange_amont(
            self.interface_cells.Ti,
            self.interface_cells.Td[1],
            self.interface_cells.Td[2],
            (0.5 - self.ad) * self.dx,
            1.0 * self.dx,
            2.0 * self.dx,
            1.5 * self.dx,
        )
        Tip52, dTdxip52, _ = interp.lagrange_amont(
            self.interface_cells.Td[1],
            self.interface_cells.Td[2],
            self.interface_cells.Td[3],
            1.0 * self.dx,
            2.0 * self.dx,
            3.0 * self.dx,
            2.5 * self.dx,
        )
        self._T_f[0] = np.nan
        self._T_f[1] = Tim32
        self._T_f[2] = Tim12
        self._T_f[3] = Tip12
        self._T_f[4] = Tip32
        self._T_f[5] = Tip52
        self._gradT_f[0] = np.nan
        self._gradT_f[1] = np.nan  # dTdxim32
        self._gradT_f[2] = dTdxim12
        self._gradT_f[3] = dTdxip12
        self._gradT_f[4] = np.nan  # dTdxip32
        self._gradT_f[5] = np.nan  # dTdxip52


class FaceInterpolationQuickGhost(FaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "QuickGhost et amont 1"

    def _compute_Tf_gradTf(self):
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

        _, _, _, Tim12, _ = interpolate_from_center_to_face_quick(
            self.interface_cells.Tg
        )
        Tip12 = (
            self.interface_cells.Td[0] + self.interface_cells.dTdxd * self.dx * 0.5
        )  # interpolation amont
        _, _, Tip32, _, _ = interpolate_from_center_to_face_quick(
            self.interface_cells.Td
        )
        # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[0] = np.nan
        self._T_f[1] = np.nan
        self._T_f[2] = Tim12
        self._T_f[3] = Tip12
        self._T_f[4] = Tip32
        self._T_f[5] = np.nan

        self._gradT_f[0] = np.nan
        self._gradT_f[1] = np.nan
        self._gradT_f[2] = self.interface_cells.gradTg[-1]
        self._gradT_f[3] = self.interface_cells.gradTd[0]
        self._gradT_f[4] = np.nan
        self._gradT_f[5] = np.nan


class FaceInterpolationQuickGhostLdaGradTi(FaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "QuickGhost et amont 1, flux q_I"

    def _compute_Tf_gradTf(self):
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

        _, _, _, Tim12, _ = interpolate_from_center_to_face_quick(
            self.interface_cells.Tg
        )
        Tip12 = (
            self.interface_cells.Td[0] + self.interface_cells.dTdxd * self.dx * 0.5
        )  # interpolation amont
        _, _, Tip32, _, _ = interpolate_from_center_to_face_quick(
            self.interface_cells.Td
        )
        # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[0] = np.nan
        self._T_f[1] = np.nan
        self._T_f[2] = Tim12
        self._T_f[3] = Tip12
        self._T_f[4] = Tip32
        self._T_f[5] = np.nan

        self._gradT_f[0] = np.nan
        self._gradT_f[1] = np.nan
        self._gradT_f[2] = self.interface_cells.dTdxg
        self._gradT_f[3] = self.interface_cells.dTdxd
        self._gradT_f[4] = np.nan
        self._gradT_f[5] = np.nan


class FaceInterpolationQuickUpwindGhost(FaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "QuickGhost et amont 0"

    def _compute_Tf_gradTf(self):
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
        Tim12 = self.interface_cells.Tg[-2]
        Tip12 = self.interface_cells.Td[0]
        Tip32 = self.interface_cells.Td[1]

        self._T_f[0] = np.nan
        self._T_f[1] = np.nan
        self._T_f[2] = Tim12
        self._T_f[3] = Tip12
        self._T_f[4] = Tip32
        self._T_f[5] = np.nan

        self._gradT_f[0] = np.nan
        self._gradT_f[1] = np.nan
        self._gradT_f[2] = self.interface_cells.gradTg[-1]
        self._gradT_f[3] = self.interface_cells.gradTd[0]
        self._gradT_f[4] = np.nan
        self._gradT_f[5] = np.nan


class FaceInterpolationAmontCentre(FaceInterpolationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "mélange centré2, amont1 et amont1"

    def _compute_Tf_gradTf(self):
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
        self._compute_Tf()
        self._compute_gradTf()

    def _compute_Tf(self):
        Tim32, _, _ = interp.lagrange_amont_centre(
            self.interface_cells.Tg[0],
            self.interface_cells.Tg[1],
            self.interface_cells.Tg[2],
            -2 * self.dx,
            -1.0 * self.dx,
            0.0 * self.dx,
            -0.5 * self.dx,
        )
        Tim12, _, _ = interp.lagrange_amont_centre(
            self.interface_cells.Tg[1],
            self.interface_cells.Tg[2],
            self.interface_cells.Ti,
            -2.0 * self.dx,
            -1.0 * self.dx,
            (self.ag - 0.5) * self.dx,
            -0.5 * self.dx,
        )
        Tip12, _, _ = interp.amont_decentre(
            self.interface_cells.Ti,
            self.interface_cells.dTdxd,
            (0.5 - self.ad) * self.dx,
            0.5 * self.dx,
        )
        Tip32, _, _ = interp.lagrange_amont_centre(
            self.interface_cells.Ti,
            self.interface_cells.Td[1],
            self.interface_cells.Td[2],
            (0.5 - self.ad) * self.dx,
            1.0 * self.dx,
            2.0 * self.dx,
            1.5 * self.dx,
        )
        Tip52, _, _ = interp.lagrange_amont_centre(
            self.interface_cells.Td[1],
            self.interface_cells.Td[2],
            self.interface_cells.Td[3],
            1.0 * self.dx,
            2.0 * self.dx,
            3.0 * self.dx,
            2.5 * self.dx,
        )
        self._T_f[0] = np.nan
        self._T_f[1] = Tim32
        self._T_f[2] = Tim12
        self._T_f[3] = Tip12
        self._T_f[4] = Tip32
        self._T_f[5] = Tip52

    def _compute_gradTf(self):
        _, dTdxim12, _ = interp.lagrange_centre_grad(
            self.interface_cells.Tg[-3],
            self.interface_cells.Tg[-2],
            self.interface_cells.Ti,
            self.interface_cells.dTdxg,
            -2 * self.dx,
            -1 * self.dx,
            (self.ag - 0.5) * self.dx,
            -0.5 * self.dx,
        )
        _, dTdxip12, _ = interp.lagrange_centre_grad(
            self.interface_cells.Td[2],
            self.interface_cells.Td[1],
            self.interface_cells.Ti,
            self.interface_cells.dTdxd,
            2.0 * self.dx,
            1.0 * self.dx,
            (0.5 - self.ad) * self.dx,
            0.5 * self.dx,
        )
        self._gradT_f[0] = np.nan
        self._gradT_f[1] = np.nan
        self._gradT_f[2] = dTdxim12
        self._gradT_f[3] = dTdxip12
        self._gradT_f[4] = np.nan
        self._gradT_f[5] = np.nan
