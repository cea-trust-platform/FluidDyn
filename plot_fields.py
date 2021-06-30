from main import *


class Plotter:
    def __init__(self, cas='classic'):
        self.cas = cas

    def plot(self, problem):
        if self.cas is 'classic':
            plot_classic(problem)
        elif self.cas is 'decale':
            plot_decale(problem)
        else:
            raise NotImplementedError


def plot_decale(problem):
    plt.figure(get_name(problem))
    x0 = problem.time*problem.v
    x_dec, T_dec = decale_perio(problem.x, problem.T, x0, problem.markers)
    c = plt.plot(x_dec, T_dec, label='time %f' % problem.time)
    col = c[-1].get_color()
    maxi = max(np.max(problem.T), np.max(problem.I))
    mini = min(np.min(problem.T), np.min(problem.I))
    while x0 > problem.Delta:
        x0 -= problem.Delta
    plt.plot([decale_positif(problem.markers[0] - x0, problem.Delta)]*2, [mini, maxi], '--', c=col)
    plt.plot([decale_positif(problem.markers[1] - x0, problem.Delta)]*2, [mini, maxi], '--', c=col)
    plt.xticks(problem.x_f)
    plt.grid(b=True, which='major')
    plt.legend()


def plot_classic(problem):
    plt.figure(get_name(problem))
    c = plt.plot(problem.x, problem.I, '+')
    col = c[-1].get_color()
    plt.plot(problem.x, problem.T, c=col, label='time %f' % problem.time)
    # x, T = get_T(problem.dx, problem.Delta, problem.lda1, problem.lda2, problem.markers)
    # plt.plot(x, T, '--', c=col, label='solution ana, time %f' % problem.time)
    maxi = max(np.max(problem.T), np.max(problem.I))
    mini = min(np.min(problem.T), np.min(problem.I))
    plt.plot([problem.markers[0]]*2, [mini, maxi], '--', c=col)
    plt.plot([problem.markers[1]]*2, [mini, maxi], '--', c=col)
    plt.xticks(problem.x_f)
    plt.grid(which='major')
    plt.legend()


def get_name(problem):
    return 'Cas : %s, %s, %s, dx = %.3f, cfl = %.3f' % (problem.cas, problem.time_scheme, problem.schema, problem.dx, problem.cfl)


def decale_perio(x, T, x0=0., markers=None, plot=False):
    """
    décale de x0 vers la gauche la courbe T en interpolant entre les décalages direct de n*dx < x0 < (n+1)*dx
    avec la formule suivante : x_interp += (x0-n*dx)
    :param x:
    :param T:
    :param x0:
    :return: x et T decalé
    """
    dx = x[1] - x[0]
    Delta = x[-1] + dx/2.
    while x0 > Delta:
        x0 -= Delta
    n = int(x0/dx)
    T_decale = np.r_[T[n:], T[:n]]
    x_decale = x - (x0 - n*dx)
    if plot:
        plt.figure()
        plt.plot(x, T_decale, label='Tn')
        T_np1 = np.r_[T[n + 1:], T[:n + 1]]
        plt.plot(x, T_np1, label='Tnp1')
        plt.plot(x_decale, T_decale, label='T_decale')

        if markers is not None:
            mini = np.min(T_decale)
            maxi = np.max(T_decale)
            plt.plot([decale_positif(markers[0] - n*dx, Delta)] * 2, [mini, maxi], '--')
            plt.plot([decale_positif(markers[1] - n*dx, Delta)] * 2, [mini, maxi], '--')
            plt.plot([decale_positif(markers[0] - (n+1)*dx, Delta)] * 2, [mini, maxi], '--')
            plt.plot([decale_positif(markers[1] - (n+1)*dx, Delta)] * 2, [mini, maxi], '--')
            plt.plot([decale_positif(markers[0] - x0, Delta)]*2, [mini, maxi], '--')
            plt.plot([decale_positif(markers[1] - x0, Delta)] * 2, [mini, maxi], '--')
    return x_decale, T_decale


def decale_positif(mark, Delta):
    while mark < 0.:
        mark += Delta
    return mark
