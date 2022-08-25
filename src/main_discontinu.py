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

    def post(self, cells, i_int: int, ist: int):
        self.cells[2 * i_int + ist] = cells
        self.lda_grad_T[i_int, ist] = cells.lda_gradTi
        self.T[i_int, ist] = cells.Ti
        self.Tg[i_int, ist] = cells.Tg[-1]
        self.Td[i_int, ist] = cells.Td[0]
        self.gradTg[i_int, ist] = cells.gradTg[-1]
        self.gradTd[i_int, ist] = cells.gradTd[0]


def get_prop(prop, i, liqu_a_gauche=True):
    if liqu_a_gauche:
        ldag = prop.phy_prop.lda1
        rhocpg = prop.phy_prop.rho_cp1
        ldad = prop.phy_prop.lda2
        rhocpd = prop.phy_prop.rho_cp2
        ag = prop.I[i]
        ad = 1.0 - prop.I[i]
    else:
        ldag = prop.phy_prop.lda2
        rhocpg = prop.phy_prop.rho_cp2
        ldad = prop.phy_prop.lda1
        rhocpd = prop.phy_prop.rho_cp1
        ag = 1.0 - prop.I[i]
        ad = prop.I[i]
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


class ProblemDiscontinu(Problem):
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
        if not conv_interf.endswith("ghost"):
            raise (Exception("Le schema conv_interf doit etre du type ghost."))

        if isinstance(self.interp_type, InterfaceInterpolationBase):
            self.interpolation_interface = self.interp_type
        # Le reste est hérité de l'ancienne manière de faire. À supprimer à terme.
        elif self.interp_type == 'Ti':
            self.interpolation_interface = InterfaceInterpolation1_0(dx=self.num_prop.dx)
        elif self.interp_type == 'Ti2':
            self.interpolation_interface = InterfaceInterpolation2(dx=self.num_prop.dx)
        elif self.interp_type == "Ti2_vol":
            self.interpolation_interface = InterfaceInterpolation2(dx=self.num_prop.dx, volume_integration=True)
        elif self.interp_type == "Ti3":
            self.interpolation_interface = InterfaceInterpolation3(dx=self.num_prop.dx)
        elif self.interp_type == "Ti3_vol":
            self.interpolation_interface = InterfaceInterpolation3(dx=self.num_prop.dx, volume_integration=True)
        elif self.interp_type == "Ti3_1_vol":
            raise NotImplementedError
        elif self.interp_type == "gradTi":
            self.interpolation_interface = InterfaceInterpolationContinuousFlux1(dx=self.num_prop.dx)
        elif self.interp_type == "gradTi2":
            self.interpolation_interface = InterfaceInterpolationContinuousFlux2(dx=self.num_prop.dx)
        elif self.interp_type == "energie_temperature":
            self.interpolation_interface = InterfaceInterpolationEnergieTemperature(dx=self.num_prop.dx)
        elif self.interp_type == "integrale":
            self.interpolation_interface = InterfaceInterpolationIntegral(dx=self.num_prop.dx)
        else:
            raise NotImplementedError

        if isinstance(conv_interf, FaceInterpolationBase):
            self.face_interpolation = conv_interf
        # Le reste est hérité de l'ancienne manière de faire. À supprimer à terme.
        elif self.interp_type.endswith("_vol"):
            self.face_interpolation = FaceInterpolationQuick(vdt=self.phy_prop.v * self.dt, time_integral='exact')
        elif self.interp_type == "energie_temperature":
            self.face_interpolation = FaceInterpolationQuick(vdt=self.phy_prop.v * self.dt, time_integral='exact')
        elif self.conv_interf == "weno":
            self.face_interpolation = FaceInterpolationQuick(vdt=self.phy_prop.v * self.dt, time_integral='exact')
        elif self.conv_interf == "quick":
            self.face_interpolation = FaceInterpolationQuick(vdt=self.phy_prop.v * self.dt, time_integral='exact')
        elif self.conv_interf == "quick_ghost":
            self.face_interpolation = FaceInterpolationQuickGhost(vdt=self.phy_prop.v * self.dt, time_integral='exact')
        elif self.conv_interf == "quick_upwind_ghost":
            self.face_interpolation = FaceInterpolationQuickUpwindGhost(vdt=self.phy_prop.v * self.dt, time_integral='exact')
        elif self.conv_interf == "upwind":
            self.face_interpolation = FaceInterpolationUpwind(vdt=self.phy_prop.v * self.dt, time_integral='exact')
        elif self.conv_interf == "amont_centre":
            self.face_interpolation = FaceInterpolationAmontCentre(vdt=self.phy_prop.v * self.dt, time_integral='exact')
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
            print("Pas d'interpolation chargée, attention la sauvegarde est probablement trop vieille")

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

    def _corrige_flux_interfaces(self, T, bulles, *args):
        for i_bulle, (i_amont, i_aval) in enumerate(bulles.ind):
            # i_bulle sert à aller chercher les valeurs aux interfaces, i_amont et i_aval servent à aller chercher les valeurs sur
            # le maillage cartésien
            for ist, i in enumerate((i_amont, i_aval)):
                stencil_interf = list(cl_perio(len(T), i))
                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                    self, i, liqu_a_gauche= (i == i_amont)
                )
                self.interpolation_interface.interpolate(T[stencil_interf], ag, ldag, ldad)
                self.face_interpolation.interpolate_on_faces(self.interpolation_interface, rhocpg, rhocpd)
                self._corrige_flux_une_interface(stencil_interf, *args)
                self.bulles.post(self.interpolation_interface, i_bulle, ist)

    def _corrige_flux_une_interface(self, stencil_interf, flux_conv, flux_diff, *args):
        # Correction des cellules i0 - 1 à i0 + 1 inclue
        self._corrige_flux_local(flux_conv, self._get_new_flux_conv, stencil_interf)
        self._corrige_flux_local(flux_diff, self._get_new_flux_diff, stencil_interf)

    @staticmethod
    def _corrige_flux_local(flux, get_new_flux, stencil_interf):
        new_flux = get_new_flux()
        stencil_new_flux = get_stencil(new_flux)
        ind_flux_corrige = stencil_interf[stencil_new_flux]
        flux[ind_flux_corrige] = new_flux[stencil_new_flux]

    def _get_new_flux_conv(self):
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        return self.face_interpolation.rhocp_f * self.face_interpolation.T_f * self.phy_prop.v

    def _get_new_flux_diff(self):
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        return self.face_interpolation.lda_f * self.face_interpolation.gradT

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
                I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(
            self.T, self.bulles, bool_debug, debug
        )
        self.flux_diff = self._compute_diffusion_flux(
            self.T, self.bulles, bool_debug, debug
        )
        self._corrige_flux_interfaces(
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

            flux_conv = rho_cp_f * self._compute_convection_flux(
                T_int, markers_int, bool_debug, debug
            )
            flux_diff = self._compute_diffusion_flux(
                T_int, markers_int, bool_debug, debug
            )

            self._corrige_flux_interfaces(T_int, markers_int, flux_conv, flux_diff)
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
        return "Energie, " + self.interpolation_interface.name + ", " + self.face_interpolation.name


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

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
        
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
        self.flux_conv = self._compute_convection_flux(
            self.rho_cp_a * self.T, self.bulles, bool_debug, debug
        )
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
        self.flux_conv = self._compute_convection_flux(
            self.rho_cp_a * self.T, self.bulles, bool_debug, debug
        )
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
                # flux_diff[ind_flux_diff] = lda_grad_T[2:4]
                # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = (
            I_np1 * self.phy_prop.rho_cp1 + (1.0 - I_np1) * self.phy_prop.rho_cp2
        )
        self.flux_conv = self._compute_convection_flux(
            self.rho_cp_a * self.T, self.bulles, bool_debug, debug
        )
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


class ProblemDiscontinuEcomme3D(Problem):
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

                # Correction des cellules i0 - 1 à i0 + 1 inclue

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                self.bulles.post(cells, i_int, ist)

                # Correction des flux cellules
                ind_flux_conv = [
                    ip1,
                ]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
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
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(
            self.T, self.bulles, bool_debug, debug
        )
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
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(
            self.T, self.bulles, bool_debug, debug
        )
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

            flux_conv = rho_cp_f * self._compute_convection_flux(
                T_int, markers_int, bool_debug, debug
            )
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
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(
            self.T, self.bulles, bool_debug, debug
        )
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
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(
            self.T, self.bulles, bool_debug, debug
        )
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
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(
            self.T, self.bulles, bool_debug, debug
        )
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
        self.flux_conv = self.rho_cp_f * self._compute_convection_flux(
            self.T, self.bulles, bool_debug, debug
        )
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
        self.flux_conv = self._compute_convection_flux(
            self.rho_cp_a * self.T, self.bulles, bool_debug, debug
        )
        T_f = self._compute_convection_flux(self.T, self.bulles, bool_debug, debug)
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
        self.flux_conv = self._compute_convection_flux(
            self.T, self.bulles, bool_debug, debug
        )
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
        """
        Dans cette approche on calclue Ti et lda_gradTi soit en utilisant la continuité avec Tim1 et Tip1, soit en
        utilisant la continuité des lda_grad_T calculés avec Tim2, Tim1, Tip1 et Tip2.
        Dans les deux cas il est à noter que l'on utilise pas les valeurs présentes dans la cellule de l'interface.
        On en déduit ensuite les gradients de température aux faces, et les températures aux faces.

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


class ProblemDiscontinuFT(Problem):
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

