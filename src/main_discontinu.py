import numpy as np
from src.main import *


class ProblemDiscontinu(Problem):
    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        self.T1 = self.T.copy()
        self.T2 = self.T.copy()

    def euler_timestep(self, debug=None):
        int_div_T_u = 1 / (self.phy_prop.dS * self.num_prop.dx) * \
                      integrale_volume_div(self.T, self.phy_prop.v * np.ones((self.T.shape[0] + 1,)), I=self.I,
                                           dS=self.phy_prop.dS, schema=self.num_prop.schema)
        int_div_lda_grad_T = 1. / (self.phy_prop.dS * self.num_prop.dx) * \
                             integrale_volume_div(self.Lda_h, grad(self.T, dx=self.num_prop.dx), I=self.I,
                                                  dS=self.phy_prop.dS, schema=self.num_prop.schema)
        rho_cp_inv_h = 1./self.rho_cp_h
        self.T += self.dt * (-int_div_T_u + self.phy_prop.diff * rho_cp_inv_h * int_div_lda_grad_T)
        self.update_markers()
        pass
