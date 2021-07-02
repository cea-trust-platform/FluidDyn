from main import *
from plot_fields import *


if __name__ == '__main__':
    Delta = 10.
    lda_1 = 1.
    lda_2 = 10.
    rho_cp_1 = 1.
    rho_cp_2 = 10.
    markers = np.array([0.4 * Delta, 0.55 * Delta])
    v = 1.
    dt = 1.
    fo = 0.5

    t_fin = 1.
    Dx = 10. ** np.linspace(-1, -1, 1)
    Cfl = 10. ** np.linspace(-1., -0.5, 1)
    Schema = ['upwind', 'center', 'weno']

    Cas_test = itertools.product(Dx, Cfl, Schema)

    for dx, cfl, schema in Cas_test:
        e_m = None
        t_m = None
        n_moy = 1
        for decal in np.linspace(0., dx, n_moy+2)[1:-1]:
            # x, T = get_T(dx=dx, Delta=Delta, lda_1=lda_1, lda_2=lda_2, markers=markers)
            markers_decal = np.array([markers[0] + decal, markers[1]])
            x, T = get_T(dx=dx, Delta=Delta, markers=markers_decal, lda_1=lda_1, lda_2=lda_2)
            T = T + 1.

            prob = ProblemConserv(Delta, dx, lda_1, lda_2, rho_cp_1, rho_cp_2, markers_decal, T, v, dt, cfl, fo,
                                  diff=0., schema=schema, time_scheme='rk4')
            t, e = prob.timestep(n=5000, number_of_plots=5, debug=False, plotter=Plotter('decale'))
            plt.figure('energie')
            plt.plot(t, e, label='dx = %.3f, cfl = %.3f, schema : %s, sous_pas : %.3f' % (dx, cfl, schema, decal))
            plt.legend()
            t_m = t
            if e_m is None:
                e_m = e
            else:
                e_m += e
        e_m /= n_moy
        plt.figure('energie')
        plt.plot(t_m, e_m, label='dx = %f, cfl = %f, schema : %s' % (dx, cfl, schema))
        plt.legend()
        n = len(e_m)
        i0 = int(n/5)
        dedt_adim = (e_m[-1] - e_m[i0]) / (t_m[-1] - t_m[i0]) * prob.dt / (rho_cp_1*Delta*1.)  # on a mult
        # par Dt / rho_cp_l T_l V
        print('dx = %f, cfl = %f, schema : %s, dE*/dt* = %f' % (dx, cfl, schema, dedt_adim))
    plt.show()
