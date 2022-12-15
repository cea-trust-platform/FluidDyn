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

import pickle
from copy import deepcopy

from src.problem_definition import *
from src.interpolation_methods import interpolate, integrale_vol_div, grad, Flux
from src.temperature_initialisation_functions import *


class StateProblem:
    phy_prop: PhysicalProperties
    num_prop: NumericalProperties
    bulles: Bulles

    def __init__(
        self,
        T0,
        markers=None,
        num_prop=None,
        phy_prop=None,
        name=None,
        fonction_source=None,
    ):
        self._imposed_name = name
        if phy_prop is None:
            print("Attention, les propriétés physiques par défaut sont utilisées")
            phy_prop = PhysicalProperties()
        if num_prop is None:
            print("Attention, les propriétés numériques par défaut sont utilisées")
            num_prop = NumericalProperties()

        self._init_from_phy_prop(phy_prop)
        self._init_from_num_prop(num_prop)
        self.bulles = self._init_bulles(markers)

        print()
        print(self.name_cas)
        print("=" * len(self.name_cas))
        self.T = T0(self.x, markers=self.bulles, phy_prop=self.phy_prop)
        self.dt = self.get_time()
        self.time = 0.0
        self.I = self._update_I()
        self.If = self._update_If()
        self.iter = 0
        self.flux_conv = Flux(np.zeros_like(self.x_f))
        self.flux_diff = Flux(np.zeros_like(self.x_f))
        print("Db / dx = %.2i" % (self.bulles.diam / self.dx))
        print("Monofluid convection : ", self.num_prop.schema)
        self._T_final = self.T_final_prevu
        self.fonction_source = fonction_source

    def _init_from_phy_prop(self, phy_prop: PhysicalProperties):
        self.phy_prop = deepcopy(phy_prop)
        self.Delta = self.phy_prop.Delta  # type: float
        self.v = self.phy_prop.v  # type: float
        self.active_diff = self.phy_prop.diff  # type: float
        self.lda = MonofluidVar(phy_prop.lda1, phy_prop.lda2)
        self.rho_cp = MonofluidVar(phy_prop.rho_cp1, phy_prop.rho_cp2)
        self.rho_cp_inv = MonofluidVar(1.0, 1.0) / MonofluidVar(
            phy_prop.rho_cp1, phy_prop.rho_cp2
        )

    def _init_from_num_prop(self, num_prop: NumericalProperties):
        self.num_prop = deepcopy(num_prop)
        self.x = self.num_prop.x
        self.x_f = self.num_prop.x_f
        self.dx = self.num_prop.dx

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
        init_bulles = self.bulles.copy()
        init_bulles.shift(-self.time * self.v)
        pb_bulles = pb.bulles.copy()
        arrive_bulles = pb.bulles.copy()
        arrive_bulles.shift(-pb.time * pb.phy_prop.v)
        tolerance = 10**-6
        equal_init_markers = np.all(
            np.abs(init_bulles.markers - arrive_bulles.markers) < tolerance
        )
        if not equal_init_markers:
            print("Init markers : ", init_bulles.markers)
            print("Arrive bulle markers : ", arrive_bulles.markers)
            raise Exception(
                "Impossible de copier le Problème, il n'a pas les mm markers de départ"
            )

        self.bulles = deepcopy(pb_bulles)
        self.T = pb.T.copy()
        self.dt = pb.dt
        self.time = pb.time
        self.I = self._update_I()
        self.If = self._update_If()
        self.iter = pb.iter
        self.flux_conv = pb.flux_conv.copy()
        self.flux_diff = pb.flux_diff.copy()

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
        if self.v == 0.0:
            return "%s, %s, dx = %g, dt = %.2g" % (
                self.num_prop.time_scheme,
                self.num_prop.schema,
                self.dx,
                self.dt,
            )
        elif self.phy_prop.diff == 0.0:
            return "%s, %s, dx = %g, cfl = %g" % (
                self.num_prop.time_scheme,
                self.num_prop.schema,
                self.dx,
                self.cfl,
            )
        else:
            return "%s, %s, dx = %g, dt = %.2g, cfl = %g" % (
                self.num_prop.time_scheme,
                self.num_prop.schema,
                self.dx,
                self.dt,
                self.cfl,
            )

    @property
    def cfl(self):
        return self.v * self.dt / self.dx

    def _update_I(self):
        i = self.bulles.indicatrice_liquide(self.x)
        return i

    def _update_If(self):
        i_f = self.bulles.indicatrice_liquide(self.x_f)
        return i_f

    def get_time(self) -> float:
        # nombre CFL = 1. par défaut
        if self.v > 10 ** (-12):
            dt_cfl = self.dx / self.v * self.num_prop.cfl_lim
        else:
            dt_cfl = 10**15
        # nombre de fourier = 1. par défaut
        dt_fo = (
            self.dx**2
            / max(self.lda.l, self.lda.v)
            * min(self.rho_cp.l, self.rho_cp.v)
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
        return np.sum(self.rho_cp.a(self.I) * self.T * self.phy_prop.dS * self.dx)

    @property
    def T_final_prevu(self):
        return np.sum(self.rho_cp.a(self.I) * self.T) / np.sum(self.rho_cp.a(self.I))

    @property
    def T_final(self):
        return self._T_final

    @property
    def energy_m(self):
        return np.sum(self.rho_cp.a(self.I) * self.T * self.dx) / self.phy_prop.Delta

    def update_markers(self, h=1.0):
        self.bulles.shift(h * self.v * self.dt)
        self.I = self._update_I()
        self.If = self._update_If()

    def _echange_flux(self):
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

    def _compute_convection_flux(self):
        T_u = interpolate(self.T, I=self.I, schema=self.num_prop.schema) * self.v
        return T_u

    def _compute_diffusion_flux(self):
        lda_grad_T = interpolate(
            self.lda.h(self.I), I=self.I, schema="center_h"
        ) * grad(self.T, self.dx)
        return lda_grad_T

    def compute_time_derivative(self, bool_debug=False, debug=None, *args, **kwargs):
        rho_cp_inv_h = 1.0 / self.rho_cp.h(self.I)
        self.flux_conv = self._compute_convection_flux()
        self.flux_diff = self._compute_diffusion_flux()
        self._corrige_flux_coeff_interface(self.T, self.bulles)
        self._echange_flux()
        dTdt = -integrale_vol_div(
            self.flux_conv, self.dx
        ) + self.active_diff * rho_cp_inv_h * integrale_vol_div(self.flux_diff, self.dx)
        dTdt += self.compute_source()
        return dTdt

    def compute_source(self):
        if self.fonction_source is not None:
            source = self.fonction_source(self.time, self.x, self.T)
        else:
            source = 0.0
        return source


"""
Gardée pour des raisons de compatibilité avec certaines études, mais obsolète.
La classe problème est remplacée par la classe TimeProblem, qui sépare la gestion
temporelle, la gestion des statistiques et celle d'un état.
"""
class Problem:
    bulles: Bulles
    num_prop: NumericalProperties
    phy_prop: PhysicalProperties

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, name=None):
        self._imposed_name = name
        if phy_prop is None:
            print("Attention, les propriétés physiques par défaut sont utilisées")
            phy_prop = PhysicalProperties()
        if num_prop is None:
            print("Attention, les propriétés numériques par défaut sont utilisées")
            num_prop = NumericalProperties()
        self.phy_prop = deepcopy(phy_prop)  # type: PhysicalProperties
        self.num_prop = deepcopy(num_prop)  # type: NumericalProperties
        self.bulles = self._init_bulles(markers)
        print()
        print(self.name)
        print("=" * len(self.name))
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
            equal_init_markers = np.all(
                self.bulles.init_markers == pb.bulles.init_markers
            )
        except:
            equal_init_markers = True
            print(
                "Attention, les markers initiaux ne sont pas enregistrés dans la référence"
            )
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
        return T_u

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
        return lda_grad_T

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
        # TODO: a retirer en mm temps que Problem
        from src.time_problem import SimuName

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


class MonofluidVar:
    def __init__(self, val_liquid, val_vapeur):
        self._val_liquid = val_liquid
        self._val_vapeur = val_vapeur

    def a(self, indicatrice_liquide):
        return self._val_liquid * indicatrice_liquide + self._val_vapeur * (
            1.0 - indicatrice_liquide
        )

    def h(self, indicatrice_liquide):
        return 1.0 / (
            indicatrice_liquide / self._val_liquid
            + (1.0 - indicatrice_liquide) / self._val_vapeur
        )

    @property
    def l(self):
        return self._val_liquid

    @property
    def v(self):
        return self._val_vapeur

    def __mul__(self, other):
        return MonofluidVar(self.l * other.l, self.v * other.v)

    def __rmul__(self, other):
        return MonofluidVar(self.l * other.l, self.v * other.v)

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            assert (other.l != 0.0) and (other.v != 0.0)
            return MonofluidVar(self.l / other.l, self.v / other.v)
        if isinstance(other, float) or isinstance(other, int):
            assert other != 0.0
            return MonofluidVar(self.l / other, self.v / other)
        if isinstance(other, tuple):
            assert (other[0] != 0.0) and (other[1] != 0.0)
            return MonofluidVar(self.l / other[0], self.v / other[1])

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            assert (other.l != 0.0) and (other.v != 0.0)
            return MonofluidVar(other.l / self.l, other.v / self.v)
        if isinstance(other, float) or isinstance(other, int):
            assert other != 0.0
            return MonofluidVar(other / self.l, other / self.v)
        if isinstance(other, tuple):
            assert (other[0] != 0.0) and (other[1] != 0.0)
            return MonofluidVar(other[0] / self.l, other[1] / self.v)

    def __add__(self, other):
        return MonofluidVar(self.l + other.l, self.v + other.v)

    def __radd__(self, other):
        return MonofluidVar(self.l + other.l, self.v + other.v)

    def __sub__(self, other):
        return MonofluidVar(self.l - other.l, self.v - other.v)

    def __rsub__(self, other):
        return MonofluidVar(other.l - self.l, other.v - self.v)
