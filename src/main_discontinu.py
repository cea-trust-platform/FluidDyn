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
from src.main import *
from src.cells_interface import *


class BulleTemperature(Bulles):
    """
    On ajoute des champs afin de post-traiter les données calculées à l'interface.
    On peut maintenant savoir quelles sont les mailles diphasiques.

    Args:
        markers:
        phy_prop:
        n_bulle:
        Delta:
        x:
    """

    def __init__(self, markers=None, phy_prop=None, n_bulle=None, Delta=1.0, x=None):
        if x is not None:
            self.x = x
        else:
            raise Exception("x est un argument obligatoire")
        super().__init__(markers, phy_prop, n_bulle, Delta)
        self.T = np.zeros_like(self.markers)
        self.Tg = np.zeros_like(self.markers)
        self.Td = np.zeros_like(self.markers)
        self.gradTg = np.zeros_like(self.markers)
        self.gradTd = np.zeros_like(self.markers)
        self.xg = np.zeros_like(self.markers)
        self.xd = np.zeros_like(self.markers)
        self.lda_grad_T = np.zeros_like(self.markers)
        self.Ti = np.zeros_like(self.markers)
        self.cells = [0.0] * (2 * len(self.markers))  # type: list
        self.ind = None
        self._set_indices_markers(x)

    def _set_indices_markers(self, x):
        """
        Retourne les indices des cellules traveresées par l'interface.
        Il serait beaucoup plus économe de considérer que les marqueurs ne se déplacent pas de plus d'une cellule entre
        deux mise à jour et donc qu'il est facile de vérifier la position des marqueurs sur les 2 cellules voisines
        plutôt que partout dans le domaine. Cela dit en python il n'existe pas de moyen optimisé de faire ça.

        Args:
            x: le tableau des positions

        Returns:
            Met à jour self.ind, le tableau d'indices de la meme forme que self.markers
        """
        res = []
        dx = x[1] - x[0]
        for marks in self.markers:
            ind1 = (np.abs(marks[0] - x) < dx / 2.0).nonzero()[0][0]
            ind2 = (np.abs(marks[1] - x) < dx / 2.0).nonzero()[0][0]
            res.append([ind1, ind2])
        self.ind = np.array(res, dtype=np.int)

    def copy(self):
        cls = self.__class__
        copie = cls(markers=self.markers.copy(), Delta=self.Delta, x=self.x.copy())
        copie.diam = self.diam
        return copie

    def shift(self, dx):
        super().shift(dx)
        self._set_indices_markers(self.x)

    def post(self, cells: InterfaceCellsBase, i_int: int, ist: int):
        self.cells[2 * i_int + ist] = cells
        self.lda_grad_T[i_int, ist] = cells.lda_gradTi
        self.T[i_int, ist] = cells.Ti
        self.Tg[i_int, ist] = cells.Tg[-1]
        self.Td[i_int, ist] = cells.Td[0]
        self.gradTg[i_int, ist] = cells.gradTg[-1]
        self.gradTd[i_int, ist] = cells.gradTd[0]


def get_prop(prob: StateProblem or Problem, i, liqu_a_gauche=True):
    if isinstance(prob, StateProblem):
        if liqu_a_gauche:
            ldag = prob.lda.l
            rhocpg = prob.rho_cp.l
            ldad = prob.lda.v
            rhocpd = prob.rho_cp.v
            ag = prob.I[i]
            ad = 1.0 - prob.I[i]
        else:
            ldag = prob.lda.v
            rhocpg = prob.rho_cp.v
            ldad = prob.lda.l
            rhocpd = prob.rho_cp.l
            ag = 1.0 - prob.I[i]
            ad = prob.I[i]
    elif isinstance(prob, Problem):
        if liqu_a_gauche:
            ldag = prob.phy_prop.lda1
            rhocpg = prob.phy_prop.rho_cp1
            ldad = prob.phy_prop.lda2
            rhocpd = prob.phy_prop.rho_cp2
            ag = prob.I[i]
            ad = 1.0 - prob.I[i]
        else:
            ldag = prob.phy_prop.lda2
            rhocpg = prob.phy_prop.rho_cp2
            ldad = prob.phy_prop.lda1
            rhocpd = prob.phy_prop.rho_cp1
            ag = 1.0 - prob.I[i]
            ad = prob.I[i]
    else:
        raise NotImplementedError
    return ldag, rhocpg, ag, ldad, rhocpd, ad


class ProblemConserv2(Problem):
    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, **kwargs):
        super().__init__(
            T0, markers=markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs
        )
        if num_prop.time_scheme == "rk3":
            print("RK3 is not implemented, changes to Euler")
            self.num_prop._time_scheme = "euler"

    @property
    def name_cas(self):
        return "EOFm"

    def _euler_timestep(self, debug=None, bool_debug=False):
        rho_cp_u = (
            interpolate(self.rho_cp_a, I=self.I, schema=self.num_prop.schema)
            * self.phy_prop.v
        )
        int_div_rho_cp_u = integrale_vol_div(rho_cp_u, self.num_prop.dx)
        rho_cp_etoile = self.rho_cp_a + self.dt * int_div_rho_cp_u

        self.flux_conv = (
            interpolate(self.rho_cp_a * self.T, I=self.I, schema=self.num_prop.schema)
            * self.phy_prop.v
        )
        int_div_rho_cp_T_u = integrale_vol_div(self.flux_conv, self.num_prop.dx)

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
        self.T += (
            self.dt
            / rho_cp_etoile
            * (
                int_div_rho_cp_u * self.T
                + -int_div_rho_cp_T_u
                + self.phy_prop.diff * int_div_lda_grad_T
            )
        )

    def _rk4_timestep(self, debug=None, bool_debug=False):
        K = [np.array(0.0)]  # type: list[np.ndarray]
        K_rhocp = [0.0]
        rho_cp_T_u_l = []
        lda_gradT_l = []
        pas_de_temps = np.array([0, 0.5, 0.5, 1.0])
        for h in pas_de_temps:
            markers_int = self.bulles.copy()
            markers_int.shift(self.phy_prop.v * self.dt * h)
            temp_I = markers_int.indicatrice_liquide(self.num_prop.x)

            # On s'occupe de calculer d_rho_cp

            rho_cp = self.rho_cp_a + h * self.dt * K_rhocp[-1]
            rho_cp_u = (
                interpolate(rho_cp, I=temp_I, schema=self.num_prop.schema)
                * self.phy_prop.v
            )
            int_div_rho_cp_u = integrale_vol_div(rho_cp_u, self.num_prop.dx)

            rho_cp_etoile = rho_cp - int_div_rho_cp_u * self.dt * h

            # On s'occupe de calculer d_rho_cp_T

            T = self.T + h * self.dt * K[-1]
            rho_cp_T_u = (
                interpolate(rho_cp * T, I=temp_I, schema=self.num_prop.schema)
                * self.phy_prop.v
            )
            rho_cp_T_u_l.append(rho_cp_T_u)
            int_div_rho_cp_T_u = integrale_vol_div(rho_cp_T_u, self.num_prop.dx)

            Lda_h = 1.0 / (
                temp_I / self.phy_prop.lda1 + (1.0 - temp_I) / self.phy_prop.lda2
            )
            lda_grad_T = interpolate(Lda_h, I=temp_I, schema="center_h") * grad(
                T, self.num_prop.dx
            )
            lda_gradT_l.append(lda_grad_T)
            int_div_lda_grad_T = integrale_vol_div(lda_grad_T, self.num_prop.dx)

            K.append(
                1.0
                / rho_cp_etoile
                * (
                    T * int_div_rho_cp_u
                    - int_div_rho_cp_T_u
                    + self.phy_prop.diff * int_div_lda_grad_T
                )
            )

        coeff = np.array([1.0 / 6, 1 / 3.0, 1 / 3.0, 1.0 / 6])
        self.flux_conv = np.sum(coeff * Flux(rho_cp_T_u_l).T, axis=-1)
        self.flux_diff = np.sum(coeff * Flux(lda_gradT_l).T, axis=-1)
        d_rhocpT = np.sum(self.dt * coeff * np.array(K[1:]).T, axis=-1)
        self.T += d_rhocpT

    def _rk3_timestep(self, debug=None, bool_debug=False):
        # TODO: a implémenter
        raise NotImplementedError


class StateProblemDiscontinu(StateProblem, ABC):
    bulles: BulleTemperature

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        interp_type=None,
        conv_interf=None,
        time_integral=None,
    ):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        if num_prop.interp_type is None:
            if interp_type is None:
                self.interp_type = "Ti"
            else:
                self.interp_type = interp_type
        else:
            self.interp_type = num_prop.interp_type
        if num_prop.conv_interf is None:
            if conv_interf is None:
                self.conv_interf = num_prop.schema
            else:
                self.conv_interf = conv_interf
        else:
            self.conv_interf = num_prop.conv_interf
        if num_prop.time_integral is None:
            if time_integral is None:
                self.time_integral = "exact"
            else:
                self.time_integral = time_integral
        else:
            self.time_integral = num_prop.time_integral

        print("Interface interp type : ", self.interp_type)
        print("Face interp : ", self.conv_interf)
        print("Time integration method for surfaces :", self.time_integral)

        if isinstance(self.interp_type, InterfaceInterpolationBase):
            self.interpolation_interface = self.interp_type
        # Le reste est hérité de l'ancienne manière de faire. À supprimer à terme.
        elif self.interp_type == "Ti":
            self.interpolation_interface = InterfaceInterpolation1_0(dx=num_prop.dx)
        elif self.interp_type == "Ti2":
            self.interpolation_interface = InterfaceInterpolation2(dx=num_prop.dx)
        elif self.interp_type == "Ti2_vol":
            self.interpolation_interface = InterfaceInterpolation2(
                dx=num_prop.dx, volume_integration=True
            )
        elif self.interp_type == "Ti3":
            self.interpolation_interface = InterfaceInterpolation3(dx=num_prop.dx)
        elif self.interp_type == "Ti3_vol":
            self.interpolation_interface = InterfaceInterpolation3(
                dx=num_prop.dx, volume_integration=True
            )
        elif self.interp_type == "Ti3_1_vol":
            raise NotImplementedError
        elif self.interp_type == "gradTi":
            self.interpolation_interface = InterfaceInterpolationContinuousFlux1(
                dx=num_prop.dx
            )
        elif self.interp_type == "gradTi2":
            self.interpolation_interface = InterfaceInterpolationContinuousFlux2(
                dx=num_prop.dx
            )
        elif self.interp_type == "energie_temperature":
            self.interpolation_interface = InterfaceInterpolationEnergieTemperature(
                dx=num_prop.dx
            )
        elif self.interp_type == "integrale":
            self.interpolation_interface = InterfaceInterpolationIntegral(
                dx=num_prop.dx
            )
        else:
            raise NotImplementedError

        if isinstance(self.conv_interf, FaceInterpolationBase):
            self.face_interpolation = self.conv_interf
        elif self.interp_type.endswith("_vol"):
            self.face_interpolation = FaceInterpolationQuick(
                vdt=self.phy_prop.v * self.dt, time_integral=self.time_integral
            )
        elif self.interp_type == "energie_temperature":
            self.face_interpolation = FaceInterpolationQuick(
                vdt=self.phy_prop.v * self.dt, time_integral=self.time_integral
            )
        elif self.conv_interf == "weno":
            self.face_interpolation = FaceInterpolationQuick(
                vdt=self.phy_prop.v * self.dt, time_integral=self.time_integral
            )
        elif self.conv_interf == "quick":
            self.face_interpolation = FaceInterpolationQuick(
                vdt=self.phy_prop.v * self.dt, time_integral=self.time_integral
            )
        elif self.conv_interf == "quick_ghost":
            self.face_interpolation = FaceInterpolationQuickGhost(
                vdt=self.phy_prop.v * self.dt, time_integral=self.time_integral
            )
        elif self.conv_interf == "quick_ghost_qi":
            self.face_interpolation = FaceInterpolationQuickGhostLdaGradTi(
                vdt=self.phy_prop.v * self.dt, time_integral=self.time_integral
            )
        elif self.conv_interf == "quick_upwind_ghost":
            self.face_interpolation = FaceInterpolationQuickUpwindGhost(
                vdt=self.phy_prop.v * self.dt, time_integral=self.time_integral
            )
        elif self.conv_interf == "downwind_only_quick":
            self.face_interpolation = FaceInterpolationDiphOnlyQuick(
                vdt=self.phy_prop.v * self.dt, time_integral=self.time_integral
            )
        elif self.conv_interf == "upwind":
            self.face_interpolation = FaceInterpolationUpwind(
                vdt=self.phy_prop.v * self.dt, time_integral=self.time_integral
            )
        elif self.conv_interf == "amont_centre":
            self.face_interpolation = FaceInterpolationAmontCentre(
                vdt=self.phy_prop.v * self.dt, time_integral=self.time_integral
            )
        else:
            raise NotImplementedError

    def copy(self, pb):
        super().copy(pb)
        self.interp_type = pb.inter_type
        self.conv_interf = pb.conv_interf
        try:
            self.interpolation_interface = deepcopy(pb.interpolation_interface)
            self.face_interpolation = deepcopy(pb.face_interpolation)
        except AttributeError:
            print(
                "Pas d'interpolation chargée, attention la sauvegarde est probablement trop vieille"
            )

    def _init_bulles(self, markers=None) -> BulleTemperature:
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

    def _corrige_flux_coeff_interface(self, **kwargs):
        for i_bulle, (i_amont, i_aval) in enumerate(self.bulles.ind):
            # i_bulle sert à aller chercher les valeurs aux interfaces,
            # i_amont et i_aval servent à aller chercher les valeurs sur
            # le maillage cartésien
            for ist, i in enumerate((i_amont, i_aval)):
                centre_stencil_interf = list(cl_perio(len(self.T), i))
                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche=(i == i_amont)
                )
                self.interpolation_interface.interpolate(
                    self.T[centre_stencil_interf], ldag, ldad, ag
                )
                self.face_interpolation.interpolate_on_faces(
                    self.interpolation_interface, rhocpg, rhocpd
                )
                face_stencil_interf = centre_stencil_interf[1:]
                self._corrige_flux_une_interface(face_stencil_interf)
                self.bulles.post(self.interpolation_interface, i_bulle, ist)

    @property
    def name(self):
        return (
            self.name_cas
            + ", "
            + self.interpolation_interface.name
            + ", "
            + self.face_interpolation.name
        )

    @staticmethod
    def _corrige_flux_local(flux: Flux, get_new_flux, stencil_interf: list):
        new_flux = (
            get_new_flux()
        )  # array de 6, le premier flux est à droite du premier point
        stencil_new_flux = get_stencil(
            new_flux
        )  # liste des indices non nan du flux de correction
        arr_stencil_interf = np.array(stencil_interf)
        ind_flux_corrige = arr_stencil_interf[stencil_new_flux]
        flux[ind_flux_corrige] = new_flux[stencil_new_flux]

    def _corrige_flux_une_interface(self, stencil_interf, *args):
        # Correction des cellules i0 - 1 à i0 + 1 inclue
        self._corrige_flux_local(
            self.flux_conv, self._get_new_flux_conv, stencil_interf
        )
        self._corrige_flux_local(
            self.flux_diff, self._get_new_flux_diff, stencil_interf
        )

    @abstractmethod
    def _get_new_flux_conv(self):
        raise NotImplementedError

    @abstractmethod
    def _get_new_flux_diff(self):
        raise NotImplementedError

    @abstractmethod
    def compute_time_derivative(self, debug=None, bool_debug=False, **kwargs):
        raise NotImplementedError

    @property
    def name_cas(self):
        raise NotImplementedError


class StateProblemDiscontinuEnergieTemperatureBase(StateProblemDiscontinu, ABC):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = self.rho_cp.a(self.I) * self.T
        self.flux_conv_energie = Flux(np.zeros_like(self.flux_conv))  # type: Flux

    def copy(self, pb):
        super().copy(pb)
        self.h = pb.h.copy()
        self.flux_conv_energie = pb.flux_conv_energie.copy()

    def _corrige_flux_une_interface(self, stencil_interf, *args):
        self._corrige_flux_local(
            self.flux_conv, self._get_new_flux_conv, stencil_interf
        )
        self._corrige_flux_local(
            self.flux_diff, self._get_new_flux_diff, stencil_interf
        )
        self._corrige_flux_local(
            self.flux_conv_energie, self._get_new_flux_conv_energie, stencil_interf
        )

    def _echange_flux(self):
        super()._echange_flux()
        self.flux_conv_energie.perio()

    def _get_new_flux_conv(self):
        return self.face_interpolation.T_f * self.phy_prop.v

    def _get_new_flux_conv_energie(self):
        return self.face_interpolation.rhocp_f * self.face_interpolation.T_f * self.v

    def _get_new_flux_diff(self):
        return self.face_interpolation.lda_f * self.face_interpolation.gradT

    @abstractmethod
    def compute_time_derivative(self, debug=None, bool_debug=False, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def name_cas(self):
        raise NotImplementedError


class StateProblemDiscontinuEnergieTemperature(
    StateProblemDiscontinuEnergieTemperatureBase
):
    def compute_time_derivative(self, debug=None, bool_debug=False, **kwargs):
        self.flux_conv = (
            interpolate(self.T, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        self.flux_conv_energie = (
            interpolate(self.h, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        self.flux_diff = interpolate(
            self.lda.h(self.I), I=self.I, schema=self.num_prop.schema
        ) * grad(self.T, self.num_prop.dx)

        self._corrige_flux_coeff_interface()
        self._echange_flux()

        int_div_T_u = integrale_vol_div(self.flux_conv, self.num_prop.dx)
        int_div_rho_cp_T_u = integrale_vol_div(self.flux_conv_energie, self.num_prop.dx)
        int_div_lda_grad_T = integrale_vol_div(self.flux_diff, self.num_prop.dx)
        rho_cp_inv_h = 1.0 / self.rho_cp.h(self.I)

        dTdt = -int_div_T_u + self.active_diff * rho_cp_inv_h * int_div_lda_grad_T
        dhdt = -int_div_rho_cp_T_u + self.active_diff * int_div_lda_grad_T
        return dTdt, dhdt

    @property
    def name_cas(self):
        return "Energie-Température"


class StateProblemDiscontinuEnergieTemperatureInt(
    StateProblemDiscontinuEnergieTemperatureBase
):
    bulles: BulleTemperature

    """
    Cette classe résout le problème en couplant une équation sur la température et une équation sur l'énergie
    interne au niveau des interfaces.
    On a donc un tableau T et un tableau h

        - on calcule dans les mailles diphasiques Tgc et Tdc les températures au centres de la partie remplie par la
        phase à gauche et la partie remplie par la phase à droite.
        - on en déduit en interpolant des flux aux faces
        - on met à jour T et h avec des flux exprimés de manière monophasique.

    """

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        self.flux_diff_temp = Flux(np.zeros_like(self.flux_conv))
        self.ind_interf = np.zeros_like(self.T)

    def copy(self, pb):
        super().copy(pb)
        self.flux_diff_temp = pb.flux_diff_temp.copy()
        self.ind_interf = pb.ind_interf.copy()

    def _corrige_flux_une_interface(self, stencil_interf, *args):
        # Tnp1 = Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T
        #                 - delta * I2 * (rhocp2 - rhocpa) - [rhocp] * int_S_Ti_v_n2_dS) / rhocpa
        self._corrige_flux_local(
            self.flux_conv, self._get_new_flux_conv, stencil_interf
        )
        self._corrige_flux_local(
            self.flux_diff, self._get_new_flux_diff, stencil_interf
        )
        self._corrige_flux_local(
            self.flux_conv_energie, self._get_new_flux_conv_energie, stencil_interf
        )
        self._corrige_flux_local(
            self.flux_diff_temp, self._get_new_flux_diff_temp, stencil_interf
        )
        self.ind_interf[stencil_interf[3]] = self._get_terme_saut()

    def _get_terme_saut(self):
        return (
            1.0 / self.face_interpolation.rhocpg - 1.0 / self.face_interpolation.rhocpd
        ) * self.interpolation_interface.lda_gradTi

    def _get_new_flux_diff_temp(self):
        return (
            self.face_interpolation.lda_f
            / self.face_interpolation.rhocp_f
            * self.face_interpolation.T_f
        )

    def compute_time_derivative(self, debug=None, bool_debug=False, **kwargs):
        self.flux_conv = (
            interpolate(self.T, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        self.flux_conv_energie = (
            interpolate(self.h, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        self.flux_diff = interpolate(
            self.lda.h(self.I), I=self.I, schema=self.num_prop.schema
        ) * grad(self.T, self.num_prop.dx)
        # Attention, l'interpolation suivante n'est valide que dans le cas de deux cellules monophasiques adjacentes
        # elle nécessite impérativement la correction aux faces mitoyennes de l'interface.
        self.flux_diff_temp = interpolate(
            self.lda.h(self.I) / self.rho_cp.a(self.I),
            I=self.I,
            schema=self.num_prop.schema,
        ) * grad(self.T, self.num_prop.dx)

        self.ind_interf[:] = 0.0
        self._corrige_flux_coeff_interface()
        self._echange_flux()
        self.flux_diff_temp.perio()
        self.flux_conv_energie.perio()

        int_div_T_u = integrale_vol_div(self.flux_conv, self.num_prop.dx)
        int_inv_rhocpf_div_ldaf_grad_T = integrale_vol_div(
            self.flux_diff_temp, self.num_prop.dx
        )
        int_div_rho_cp_T_u = integrale_vol_div(self.flux_conv_energie, self.num_prop.dx)
        int_div_lda_grad_T = integrale_vol_div(self.flux_diff, self.num_prop.dx)

        dTdt = (
            -int_div_T_u
            + self.phy_prop.diff * int_inv_rhocpf_div_ldaf_grad_T
            + self.phy_prop.diff / self.num_prop.dx * self.ind_interf
        )
        dhdt = -int_div_rho_cp_T_u + self.phy_prop.diff * int_div_lda_grad_T
        return dTdt, dhdt

    @property
    def name_cas(self):
        return "Energie-TSV"


class StateProblemDiscontinuE(StateProblemDiscontinu):
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

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, **kwargs):
        if num_prop is None:
            num_prop = NumericalProperties()
        if num_prop.interp_type is None:
            num_prop.interp_type = "Ti"
        if num_prop.conv_interf is None:
            num_prop.conv_interf = "quick"
        super().__init__(
            T0, markers=markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs
        )

    def _corrige_flux_une_interface(self, stencil_interf, *args):
        # Correction des cellules i0 - 1 à i0 + 1 inclue
        self._corrige_flux_local(
            self.flux_conv, self._get_new_flux_conv, stencil_interf
        )
        self._corrige_flux_local(
            self.flux_diff, self._get_new_flux_diff, stencil_interf
        )

    def _get_new_flux_conv(self):
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        return (
            self.face_interpolation.rhocp_f
            * self.face_interpolation.T_f
            * self.phy_prop.v
        )

    def _get_new_flux_diff(self):
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        return self.face_interpolation.lda_f * self.face_interpolation.gradT

    def compute_time_derivative(self, debug=None, bool_debug=False, **kwargs):
        self.flux_conv = self.rho_cp.a(self.If) * self._compute_convection_flux()
        self.flux_diff = self._compute_diffusion_flux()
        self._corrige_flux_coeff_interface()
        self._echange_flux()
        drhocpTdt = -integrale_vol_div(
            self.flux_conv, self.num_prop.dx
        ) + self.phy_prop.diff * integrale_vol_div(self.flux_diff, self.num_prop.dx)
        return drhocpTdt

    @property
    def name_cas(self):
        return "Energie"


class StateProblemDiscontinuEsansq(StateProblemDiscontinuE):
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

    def _corrige_flux_une_interface(self, stencil_interf, *args):
        # Correction des cellules i0 - 1 à i0 + 1 inclue
        self._corrige_flux_local(
            self.flux_conv, self._get_new_flux_conv, stencil_interf
        )

    @property
    def name_cas(self):
        return "Energie sans corr. diff."


class StateProblemDiscontinuT(StateProblemDiscontinu):
    def _get_new_flux_conv(self):
        return self.face_interpolation.T_f * self.v

    def _get_new_flux_diff(self):
        return self.face_interpolation.lda_f * self.face_interpolation.gradT

    def compute_time_derivative(self, debug=None, bool_debug=False, **kwargs):
        rho_cp_inv_h = 1.0 / self.rho_cp.h(self.I)
        bulles = self.bulles.copy()
        bulles.shift(self.phy_prop.v * self.dt)
        Inp1 = bulles.indicatrice_liquide(self.num_prop.x)
        rho_cp_inv_h_np1 = self.rho_cp_inv(Inp1)

        self.flux_conv = self._compute_convection_flux()
        self.flux_diff = self._compute_diffusion_flux()
        self._corrige_flux_coeff_interface()
        self._echange_flux()
        dTdt = -integrale_vol_div(self.flux_conv, self.dx) + self.phy_prop.diff * (
            rho_cp_inv_h + rho_cp_inv_h_np1
        ) / 2.0 * integrale_vol_div(self.flux_diff, self.dx)
        return dTdt

    @property
    def name_cas(self):
        return "Temperature"


class StateProblemDiscontinuSepIntT(StateProblemDiscontinu):
    def __init__(self, *args, **kwargs):
        super(StateProblemDiscontinuSepIntT, self).__init__(*args, **kwargs)
        self.ind_interf = np.zeros_like(self.T)

    def copy(self, pb):
        super(StateProblemDiscontinuSepIntT, self).copy(pb)
        self.ind_interf = pb.ind_interf.copy()

    def _get_new_flux_conv(self):
        return self.face_interpolation.T_f * self.v

    def _get_new_flux_diff(self):
        return (
            self.face_interpolation.lda_f
            * self.face_interpolation.inv_rhocp_f
            * self.face_interpolation.gradT
        )

    def _get_new_terme_interfacial(self):
        int_i0 = (
            (
                1.0 / self.face_interpolation.rhocpg
                - 1.0 / self.face_interpolation.rhocpd
            )
            * self.interpolation_interface.lda_gradTi
            * self.face_interpolation.coeff_d
        )
        int_i1 = (
            (
                1.0 / self.face_interpolation.rhocpg
                - 1.0 / self.face_interpolation.rhocpd
            )
            * self.interpolation_interface.lda_gradTi
            * (1.0 - self.face_interpolation.coeff_d)
        )
        return np.array([int_i0, int_i1])

    def _corrige_flux_une_interface(self, stencil_interf, *args):
        # Tnp1 = Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T
        #                 - delta * I2 * (rhocp2 - rhocpa) - [rhocp] * int_S_Ti_v_n2_dS) / rhocpa
        self._corrige_flux_local(
            self.flux_conv, self._get_new_flux_conv, stencil_interf
        )
        self._corrige_flux_local(
            self.flux_diff, self._get_new_flux_diff, stencil_interf
        )
        self.ind_interf[stencil_interf[2:3]] = self._get_new_terme_interfacial()

    def compute_time_derivative(self, debug=None, bool_debug=False, **kwargs):
        self.flux_conv = (
            interpolate(self.T, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        )
        flux_diff = interpolate(self.lda.h(self.I), I=self.I, schema="center_h") * grad(
            self.T, self.num_prop.dx
        )
        # Attention, l'interpolation suivante n'est valide que dans le cas de deux cellules monophasiques adjacentes
        # elle nécessite impérativement une correction aux faces mitoyennes de l'interface.
        self.flux_diff = flux_diff / interpolate(
            self.rho_cp.a(self.I), I=self.I, schema="center_h"
        )
        self.ind_interf[:] = 0.0
        self._corrige_flux_coeff_interface()
        self._echange_flux()
        int_div_T_u = integrale_vol_div(self.flux_conv, self.dx)
        int_inv_rhocpf_div_ldaf_grad_T = integrale_vol_div(self.flux_diff, self.dx)
        dTdt = (
            -int_div_T_u
            + self.phy_prop.diff * int_inv_rhocpf_div_ldaf_grad_T
            + self.phy_prop.diff / self.num_prop.dx * self.ind_interf
        )
        return dTdt

    @property
    def name_cas(self):
        return "Temperature separe vol"


def cl_perio(n, i):
    im1 = (i - 1) % n
    im2 = (i - 2) % n
    im3 = (i - 3) % n
    i0 = i % n
    ip1 = (i + 1) % n
    ip2 = (i + 2) % n
    ip3 = (i + 3) % n
    return im3, im2, im1, i0, ip1, ip2, ip3


def get_stencil(a: np.ndarray):
    return list(np.argwhere(np.logical_not(np.isnan(a))).flatten())
