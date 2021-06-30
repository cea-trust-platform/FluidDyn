from main import *
from plot_fields import *


if __name__ == '__main__':
    Delta = 10.
    lda_1 = 1.
    lda_2 = 1.
    rho_cp_1 = 1.
    rho_cp_2 = 10.
    markers = np.array([0.4 * Delta, 0.55 * Delta])
    v = 1.
    dt = 1.
    fo = 0.5

    t_fin = 1.
    tl = []
    el = []
    Dx = 10. ** np.linspace(-1.5, -1, 1)
    Cfl = 10. ** np.linspace(-1, -0.5, 1)
    Schema = ['upwind', 'center', 'weno']

    Cas_test = itertools.product(Dx, Cfl, Schema)

    for dx, cfl, schema in Cas_test:
        # x, T = get_T(dx=dx, Delta=Delta, lda_1=lda_1, lda_2=lda_2, markers=markers)
        x, T = get_T_creneau(dx=dx, Delta=Delta, markers=markers)

        prob = Problem(Delta, dx, lda_1, lda_2, rho_cp_1, rho_cp_2, markers, T, v, dt, cfl, fo,
                       diff=0., schema=schema, time_scheme='rk4')
        t, e = prob.timestep(n=10000, number_of_plots=3, debug=False, plotter=Plotter('decale'))
        tl.append(t)
        el.append(e)
    i = 0
    Cas_test = itertools.product(Dx, Cfl, Schema)
    for dx, cfl, schema in Cas_test:
        plt.figure('energie')
        plt.plot(tl[i], el[i], label='dx = %f, cfl = %f, schema : %s' % (dx, cfl, schema))
        plt.legend()
        n = len(el[i])
        i0 = int(n/5)
        dedt = (el[i][-1] - el[i][i0]) / (tl[i][-1] - tl[i][i0])
        print('dx = %f, cfl = %f, schema : %s, dE/dt = %f' % (dx, cfl, schema, dedt))
        i += 1
    plt.show()
