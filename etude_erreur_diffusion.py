"""
Cette fiche sert à montrer la forme et les paramètres qui influencent l'erreur de diffusion pure dans la formulation
classique.
"""


from src.main import *
from src.plot_fields import *


if __name__ == '__main__':
    Delta = 10.
    lda_1 = 1.
    lda_2 = 10.
    rho_cp_1 = 1.
    rho_cp_2 = 10.
    markers = np.array([0.4 * Delta, 0.55 * Delta])
    v = 0.
    dt = 1.
    cfl = 1.

    t_fin = 1.
    Dx = 10. ** np.linspace(-1, -1, 1)
    Fo = 10. ** np.linspace(-1., 0, 2)
    Schema = ['weno']

    Cas_test = itertools.product(Dx, Fo, Schema)

    for dx, fo, schema in Cas_test:
        e_m = None
        t_m = None
        n_moy = 3
        for decal in np.linspace(0., dx, n_moy+2)[1:-1]:
            markers_decal = np.array([markers[0] + decal, markers[1]])
            # x, T = get_T(dx=dx, Delta=Delta, lda_1=lda_1, lda_2=lda_2, markers=markers_decal)
            # T += 1.
            # x, T = get_T_creneau(dx=dx, Delta=Delta, markers=markers_decal)
            # T = 9.*T + 1

            prob = Problem(Delta, dx, lda_1, lda_2, rho_cp_1, rho_cp_2, markers_decal, get_T, v, dt, cfl, fo,
                           diff=1., schema=schema, time_scheme='euler')
            t, e = prob.timestep(t_fin=t_fin, number_of_plots=5, debug=False, plotter=Plotter('classic'))
        t_m = t
        if e_m is None:
            e_m = e
        else:
            e_m += e
        e_m /= n_moy
        plt.figure('energie')
        plt.plot(t_m, e_m, label=prob.name)
        plt.legend()
        n = len(e_m)
        i0 = int(n/5)
        dedt_adim = (e_m[-1] - e_m[i0]) / (t_m[-1] - t_m[i0]) * prob.dt / (rho_cp_1*Delta*1.)  # on a mult
        # par Dt / rho_cp_l T_l V
        print(prob.name, 'dE*/dt* = %f' % dedt_adim)
    plt.show()
