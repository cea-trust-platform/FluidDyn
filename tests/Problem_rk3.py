import sys

sys.path = [r'/home/as259691/PycharmProjects/FluidDyn1D'] + sys.path

from src.main import Problem, PhysicalProperties, NumericalProperties, Bulles, get_T_creneau
from src.plot_fields import Plotter

if __name__ == '__main__':
    n_lim = 10
    t_fin_lim = 1.
    phy_prop = PhysicalProperties(Delta=0.02, v=0.2, dS=0.005**2, lda1=5.5*10**-2, lda2=15.5, rho_cp1=70278.,
                                  rho_cp2=702780., diff=1., alpha=0.06, a_i=357)
    num_prop = NumericalProperties(dx=5.*10**-5, schema='upwind', time_scheme='rk3', phy_prop=phy_prop, cfl=0.5, fo=0.5)
    markers = Bulles(phy_prop=phy_prop, n_bulle=1)
    markers.shift(0.0000001)
    plot = Plotter('decale')
    prob = Problem(get_T_creneau, markers=markers, num_prop=num_prop, phy_prop=phy_prop)
    t_fin = 0.2
    prob.timestep(t_fin=min(t_fin, t_fin_lim), n=n_lim, number_of_plots=1, plotter=plot)


