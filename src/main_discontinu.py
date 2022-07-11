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
        if x is not None:
            self.x = x
            self._set_indices_markers(x)
        else:
            raise Exception("x est un argument obligatoire")

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
        self.flux_conv_energie = np.zeros_like(self.flux_conv)

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
        self.flux_conv_ener = np.zeros_like(self.flux_conv)
        self.flux_diff_temp = np.zeros_like(self.flux_conv)
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
        self.flux_diff_temp[-1] = self.flux_diff_temp[0]
        self.flux_conv_ener[-1] = self.flux_conv_ener[0]

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
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

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

    # # TODO: finir cette méthode, attention il faut trouver une solution pour mettre à jour T de manière cohérente
    # def _rk3_timestep(self, debug=None, bool_debug=False):
    #     T_int = self.T.copy()
    #     markers_int = self.bulles.copy()
    #     K = 0.
    #     # pas_de_temps = np.array([0, 1/3., 3./4])
    #     coeff_h = np.array([1./3, 5./12, 1./4])
    #     coeff_dTdtm1 = np.array([0., -5./9, -153./128])
    #     coeff_dTdt = np.array([1., 4./9, 15./32])
    #     for step, h in enumerate(coeff_h):
    #         I_step = markers_int.indicatrice_liquide(self.num_prop.x)
    #         rho_cp_a_step = I_step * self.phy_prop.rho_cp1 + (1. - I_step) * self.phy_prop.rho_cp2
    #         # convection, conduction, dTdt = self.compute_dT_dt(T_int, markers_int, bool_debug, debug)
    #         convection = self._compute_convection_flux(T_int, markers_int, bool_debug, debug)
    #         conduction = self._compute_diffusion_flux(T_int, markers_int, bool_debug, debug)
    #         self._corrige_flux_coeff_interface(T_int, markers_int, convection, conduction)
    #         markers_int_np1 = markers_int.copy()
    #         markers_int_np1.shift(self.phy_prop.v * h * self.dt)
    #         I_step_p1 = markers_int_np1.indicatrice_liquide(self.num_prop.x)
    #         rho_cp_a_step_p1 = I_step_p1 * self.phy_prop.rho_cp1 + (1. - I_step_p1) * self.phy_prop.rho_cp2
    #         drhocpTdt = - integrale_vol_div(convection, self.num_prop.dx) \
    #             + self.phy_prop.diff * integrale_vol_div(conduction, self.num_prop.dx)
    #         # On a dT = (- (rhocp_np1 - rhocp_n) * Tn + dt * (-conv + diff)) / rhocp_np1
    #         dTdt = (- (rho_cp_a_step_p1 - rho_cp_a_step) / (h * self.dt) * T_int + drhocpTdt) / rho_cp_a_step_p1
    #         K = K * coeff_dTdtm1[step] + dTdt
    #         if bool_debug and (debug is not None):
    #             print('step : ', step)
    #             print('dTdt : ', dTdt)
    #             print('K    : ', K)
    #         T_int += h * self.dt * K / coeff_dTdt[step]  # coeff_dTdt est calculé de
    #         # sorte à ce que le coefficient total devant les dérviées vale 1.
    #         markers_int.shift(self.phy_prop.v * h * self.dt)
    #     self.T = T_int
    #
    # def _rk4_timestep(self, debug=None, bool_debug=False):
    #     # T_int = self.T.copy()
    #     K = [0.]
    #     T_u_l = []
    #     lda_gradT_l = []
    #     pas_de_temps = np.array([0., 0.5, 0.5, 1.])
    #     dx = self.num_prop.dx
    #     for h in pas_de_temps:
    #         markers_int = self.bulles.copy()
    #         markers_int.shift(self.phy_prop.v * h * self.dt)
    #         I_step = markers_int.indicatrice_liquide(self.num_prop.x)
    #         rho_cp_a_step = I_step * self.phy_prop.rho_cp1 + (1. - I_step) * self.phy_prop.rho_cp2
    #         T = self.T + h * self.dt * K[-1]
    #         convection = self._compute_convection_flux(T, markers_int, bool_debug, debug)
    #         conduction = self._compute_diffusion_flux(T, markers_int, bool_debug, debug)
    #         self._corrige_flux_coeff_interface(T, markers_int, convection, conduction)
    #         T_u_l.append(convection)
    #         lda_gradT_l.append(conduction)
    #         # On a dT = (- (rhocp_np1 - rhocp_n) * Tn + dt * (-conv + diff)) / rhocp_np1
    #         # Probleme pour h = 0., on ne peut pas calculer drhocp/dt par différence de temps
    #         raise NotImplementedError
    #         K.append(- integrale_vol_div(convection, dx)
    #                  + self.phy_prop.diff * coeff_conduction * integrale_vol_div(conduction, dx))
    #     coeff = np.array([1. / 6, 1 / 3., 1 / 3., 1. / 6])
    #     self.flux_conv = np.sum(coeff * np.array(T_u_l).T, axis=-1)
    #     self.flux_diff = np.sum(coeff * np.array(lda_gradT_l).T, axis=-1)
    #     self.T += np.sum(self.dt * coeff * np.array(K[1:]).T, axis=-1)

    @property
    def name_cas(self):
        return "ESP"  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


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
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

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
        return "ESP sans correction flux"  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


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

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

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
        return "ESP sans correction flux"  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


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
                    schema_conv="quick_ghost",
                    vdt=self.dt * self.phy_prop.v,
                    time_integral="CN",
                )
                self.bulles.cells[2 * i_int + ist] = cells

                # Correction des cellules i0 - 1 à i0 + 1 inclue

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

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
        return "ESP sans correction flux"  # + self.interp_type.replace('_', '-') + self.conv_interf.replace('_', '-')


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
                self.bulles.cells[2 * i_int + ist] = cells

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation

                T_f_ = T_f[[im2, im1, i0, ip1, ip2, ip3]]
                rhocpT_u = cells.rhocp_f * T_f_ * self.phy_prop.v
                self.bulles.Ti[i_int, ist] = cells.Ti

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
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Ti[i_int, ist] = cells.Ti

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


# class ProblemDiscontinuT2(Problem):
#     T: np.ndarray
#     I: np.ndarray
#     bulles: BulleTemperature

# """
# Cette classe résout le problème en 3 étapes :

# - on calcule le nouveau T comme avant (avec un stencil de 1 à proximité des interfaces par simplicité)
# - on calcule précisemment T1 et T2 ansi que les bons flux aux faces, on met à jour T
# - on met à jour T_i et lda_grad_T_i

# Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
# continuité de lda_grad_T et T à l'interface.

# Args:
#     T0: la fonction initiale de température
#     markers: les bulles
#     num_prop: les propriétés numériques du calcul
#     phy_prop: les propriétés physiques du calcul
# """

# def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None, conv_interf=None, **kwargs):
#     super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop, **kwargs)
#     if interp_type is None:
#         self.interp_type = 'Ti'
#     else:
#         self.interp_type = interp_type
#     print(self.interp_type)
#     if conv_interf is None:
#         conv_interf = self.num_prop.schema
#     self.conv_interf = conv_interf

# def _init_bulles(self, markers=None):
#     if markers is None:
#         return BulleTemperature(markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x)
#     elif isinstance(markers, BulleTemperature):
#         return markers.copy()
#     elif isinstance(markers, Bulles):
#         return BulleTemperature(markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x)
#     else:
#         print(markers)
#         raise NotImplementedError

# def _corrige_flux_coeff_interface(self, T, bulles, *args):
#     """
#     Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4

# Args:
#     flux_conv:
#     flux_diff:
#     coeff_diff:

# Returns:

# """
# flux_conv, flux_diff = args
# dx = self.num_prop.dx

# for i_int, (i1, i2) in enumerate(bulles.ind):
#     # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
#     # le maillage cartésien

# for ist, i in enumerate((i1, i2)):
#     if i == i1:
#         from_liqu_to_vap = True
#     else:
#         from_liqu_to_vap = False
#     im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

#                 # On calcule gradTg, gradTi, Ti, gradTd

# ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(self, i, liqu_a_gauche=from_liqu_to_vap)
# cells = CellsInterface(ldag, ldad, ag, dx, T[[im3, im2, im1, i0, ip1, ip2, ip3]],
#                        rhocpg=rhocpg, rhocpd=rhocpd, interp_type=self.interp_type,
#                        schema_conv=self.conv_interf, vdt=self.phy_prop.v * self.dt)

# # Correction des cellules i0 - 1 à i0 + 1 inclue
# # DONE: l'écrire en version flux pour être sûr de la conservation
# dx = self.num_prop.dx
# T_u = cells.T_f * self.phy_prop.v
# lda_over_rhocp_grad_T = cells.lda_f / cells.rhocp_f * cells.gradT
# self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
# self.bulles.Ti[i_int, ist] = cells.Ti

# # Correction des cellules
# ind_flux_conv = [im1, i0, ip1, ip2,
#                  ip3]  # on corrige les flux de i-3/2 a i+5/2 (en WENO ça va jusqu'a 5/2)
# ind_flux_diff = [i0, ip1]  # on corrige les flux diffusifs des faces de la cellule diphasique seulement
# flux_conv[ind_flux_conv] = T_u[1:]
# flux_diff[ind_flux_diff] = lda_over_rhocp_grad_T[2:4]
# # Tnp1 = Tn + dt (- int_S_T_u + 1/rhocp * int_S_lda_grad_T)

# def _euler_timestep(self, debug=None, bool_debug=False):
#     dx = self.num_prop.dx
#     self.flux_conv = self._compute_convection_flux(self.T, self.bulles, bool_debug, debug)
#     self.flux_diff = self._compute_diffusion_flux(1. / self.rho_cp_a * self.T, self.bulles, bool_debug, debug)
#     self._corrige_flux_coeff_interface(self.T, self.bulles, self.flux_conv, self.flux_diff)
#     dTdt = - integrale_vol_div(self.flux_conv, dx) \
#            + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
#     self.T += self.dt * dTdt

# @property
# def name_cas(self):
#     return 'TFC2, ' + self.interp_type.replace('_', '-')


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

    def _corrige_interface_aymeric1(self):
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
        self._corrige_interface_aymeric1()

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

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

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
        flux_diff[-1] = flux_diff[0]
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
