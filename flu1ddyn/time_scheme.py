import numpy as np
from abc import ABC, abstractmethod
from flu1ddyn.main import StateProblem
from flu1ddyn.main_discontinu import (
    StateProblemDiscontinuEnergieTemperatureBase,
)


class TimestepBase(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, state: StateProblem, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _sub_step(self, *args, **kwargs):
        raise NotImplementedError


class EulerTimestep(TimestepBase):
    def step(self, pb: StateProblem, debug=None, bool_debug=False):
        self._sub_step(pb, debug, bool_debug)
        pb.time += pb.dt
        pb.iter += 1

    def _sub_step(self, pb: StateProblem, *args, **kwargs):
        dTdt = pb.compute_time_derivative(*args, **kwargs)
        pb.T += pb.dt * dTdt
        pb.update_markers()


class RK3Timestep(TimestepBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.K = 0.0
        self.coeff_h = np.array([1.0 / 3, 5.0 / 12, 1.0 / 4])
        self.coeff_dTdtm1 = np.array([0.0, -5.0 / 9, -153.0 / 128])
        self.coeff_dTdt = np.array([1.0, 4.0 / 9, 15.0 / 32])

    def step(self, pb: StateProblem, *args, **kwargs):
        for step, h in enumerate(self.coeff_h):
            self._sub_step(
                pb, h, self.coeff_dTdtm1[step], self.coeff_dTdt[step], *args, **kwargs
            )
        pb.time += pb.dt
        pb.iter += 1

    def _sub_step(
        self,
        pb,
        h,
        coeff_dTdtm1,
        coeff_dTdt,
        *args,
        debug=None,
        bool_debug=False,
        **kwargs
    ):
        dTdt = pb.compute_time_derivative()
        self.K = self.K * coeff_dTdtm1 + dTdt
        if bool_debug and (debug is not None):
            print("step : ", h)
            print("dTdt : ", dTdt)
            print("K    : ", self.K)
        pb.T += h * pb.dt * self.K / coeff_dTdt  # coeff_dTdt est calculé de
        pb.update_markers(h)


class RK4Timestep(TimestepBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pas_de_tempss = np.array([0.0, 0.5, 0.5, 1.0])
        self.Ks = []

    def step(self, pb: StateProblem, *args, **kwargs):
        return NotImplementedError
        # K = [0.0]
        # T_u_l = []
        # lda_gradT_l = []
        # pas_de_temps = np.array([0.0, 0.5, 0.5, 1.0])
        # dx = pb.num_prop.dx
        # markers_int = pb.bulles.copy()
        # for h in pas_de_temps:
        #     I_k = markers_int.indicatrice_liquide(pb.num_prop.x)
        #     rho_cp_inv_h = 1.0 / pb.rho_cp.h(I_k)
        #     markers_int.shift(pb.phy_prop.v * h * pb.dt)
        #
        #     T = pb.T + h * pb.dt * K[-1]
        #
        #     # TODO: bouger ça dans la classe StateProblem
        #     convection = pb._compute_convection_flux(T, markers_int, *args, **kwargs)
        #     conduction = pb._compute_diffusion_flux(T, markers_int, *args, **kwargs)
        #     # TODO: vérifier qu'il ne faudrait pas plutôt utiliser rho_cp^{n,k}
        #     pb._corrige_flux_coeff_interface(T, markers_int, convection, conduction)
        #     convection.perio()
        #     conduction.perio()
        #     dTdt = -integrale_vol_div(
        #         convection, dx
        #     ) + pb.phy_prop.diff * rho_cp_inv_h * integrale_vol_div(conduction, dx)
        #     T_u_l.append(convection)
        #     lda_gradT_l.append(conduction)
        #     K.append(dTdt)
        # coeff = np.array([1.0 / 6, 1 / 3.0, 1 / 3.0, 1.0 / 6])
        # self.flux_conv = np.sum(coeff * Flux(T_u_l).T, axis=-1)
        # self.flux_diff = np.sum(coeff * Flux(lda_gradT_l).T, axis=-1)
        # self.T += np.sum(self.dt * coeff * np.array(K[1:]).T, axis=-1)


class EulerEnergieTimestep(EulerTimestep):
    def _sub_step(self, pb: StateProblem, *args, **kwargs):
        bulles_np1 = pb.bulles.copy()
        bulles_np1.shift(pb.v * pb.dt)
        I_np1 = bulles_np1.indicatrice_liquide(pb.x)
        rho_cp_a_np1 = pb.rho_cp.a(I_np1)
        drhocpTdt = pb.compute_time_derivative(*args, **kwargs)
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        pb.T = (pb.T * pb.rho_cp.a(pb.I) + pb.dt * drhocpTdt) / rho_cp_a_np1
        pb.update_markers()


class RK3EnergieTimestep(RK3Timestep):
    def _sub_step(
        self,
        pb: StateProblem,
        h,
        coeff_dTdtm1,
        coeff_dTdt,
        debug=None,
        bool_debug=False,
        *args,
        **kwargs
    ):
        rho_cp_a = pb.rho_cp.a(pb.I)
        markers_kp1 = pb.bulles.copy()
        markers_kp1.shift(pb.v * h * pb.dt)
        I_kp1 = markers_kp1.indicatrice_liquide(pb.x)
        rho_cp_a_kp1 = pb.rho_cp.a(I_kp1)
        drhocpTdt = pb.compute_time_derivative(debug, bool_debug)
        self.K = self.K * coeff_dTdtm1 + drhocpTdt
        pb.T = (pb.T * rho_cp_a + pb.dt * h * self.K / coeff_dTdt) / rho_cp_a_kp1
        pb.update_markers(h)


class RK3EnergieBisTimestep(RK3Timestep):
    def _sub_step(
        self,
        pb: StateProblem,
        h,
        coeff_dTdtm1,
        coeff_dTdt,
        debug=None,
        bool_debug=False,
        *args,
        **kwargs
    ):
        rho_cp_a = pb.rho_cp.a(pb.I)
        markers_kp1 = pb.bulles.copy()
        markers_kp1.shift(pb.v * h * pb.dt)
        I_kp1 = markers_kp1.indicatrice_liquide(pb.x)
        rho_cp_a_kp1 = pb.rho_cp.a(I_kp1)
        drhocpTdt = pb.compute_time_derivative(debug, bool_debug)
        dTdt = (-(rho_cp_a_kp1 - rho_cp_a) * pb.T + h * pb.dt * drhocpTdt) / (
            rho_cp_a_kp1 * h * pb.dt
        )
        self.K = self.K * coeff_dTdtm1 + dTdt
        pb.T = pb.T + pb.dt * h * self.K / coeff_dTdt
        pb.update_markers(h)


class EulerTempEnerTimestep(EulerTimestep):
    def _sub_step(
        self, pb: StateProblemDiscontinuEnergieTemperatureBase, *args, **kwargs
    ):
        dTdt, dhdt = pb.compute_time_derivative(*args, **kwargs)
        pb.T += pb.dt * dTdt
        pb.h += pb.dt * dhdt
        pb.update_markers()


class RK3TempEnerTimestep(TimestepBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.K_ener = 0.0
        self.K_temp = 0.0
        self.coeff_h = np.array([1.0 / 3, 5.0 / 12, 1.0 / 4])
        self.coeff_dTdtm1 = np.array([0.0, -5.0 / 9, -153.0 / 128])
        self.coeff_dTdt = np.array([1.0, 4.0 / 9, 15.0 / 32])

    def step(self, pb: StateProblemDiscontinuEnergieTemperatureBase, *args, **kwargs):
        for step, h in enumerate(self.coeff_h):
            self._sub_step(
                pb, h, self.coeff_dTdtm1[step], self.coeff_dTdt[step], *args, **kwargs
            )
        pb.time += pb.dt
        pb.iter += 1

    def _sub_step(
        self,
        pb,
        h,
        coeff_dTdtm1,
        coeff_dTdt,
        *args,
        debug=None,
        bool_debug=False,
        **kwargs
    ):
        dTdt, dhdt = pb.compute_time_derivative()
        self.K_temp = self.K_temp * coeff_dTdtm1 + dTdt
        self.K_ener = self.K_ener * coeff_dTdtm1 + dhdt
        pb.T += h * pb.dt * self.K_temp / coeff_dTdt  # coeff_dTdt est calculé de
        pb.h += h * pb.dt * self.K_ener / coeff_dTdt  # coeff_dTdt est calculé de
        pb.update_markers(h)
