from src.main_discontinu import *
from src.main import *

from matplotlib import rc
rc('text', usetex=True)
rc('font', size=14)
rc('font', family='serif')
rc('legend', fontsize=13)
rc('figure', max_open_warning=50)
rc('figure', figsize=(19, 8))
rc('figure', dpi=200)
rc('savefig', dpi=300)


import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, cas='classic', lda_gradT=False):
        self._cas = cas
        self.fig = None
        self.ax = None
        self.ax2 = None
        self.lda_gradT = lda_gradT

    @property
    def cas(self):
        return self._cas

    @cas.setter
    def cas(self, value):
        self._cas = value
        print("plotter mode changed to %s" % value)

    def plot(self, problem):
        if self.cas is 'classic':
            self.fig, self.ax, self.ax2 = plot_classic(problem, self.fig, self.ax, ax2=self.ax2,
                                                       lda_gradT=self.lda_gradT)
        elif self.cas is 'decale':
            self.fig, self.ax, self.ax2 = plot_decale(problem, self.fig, self.ax, ax2=self.ax2,
                                                      lda_gradT=self.lda_gradT)
        else:
            raise NotImplementedError


def plot_decale(problem, fig=None, ax=None, ax2=None, lda_gradT=False):
    if (fig is None) or (ax is None):
        if isinstance(problem.bulles, BulleTemperature) and lda_gradT:
            fig, (ax, ax2) = plt.subplots(2, sharex='all')
        else:
            fig, ax = plt.subplots(1)
    fig.suptitle(problem.name)
    x0 = problem.time*problem.phy_prop.v
    x_dec, T_dec = decale_perio(problem.num_prop.x, problem.T, x0, problem.bulles)
    c = ax.plot(x_dec, T_dec, label='%s, time %g' % (problem.name, problem.time))
    col = c[-1].get_color()
    maxi = max(np.max(problem.T), np.max(problem.I))
    mini = min(np.min(problem.T), np.min(problem.I))
    while x0 > problem.phy_prop.Delta:
        x0 -= problem.phy_prop.Delta
    for markers in problem.bulles():
        ax.plot([decale_positif(markers[0] - x0, problem.phy_prop.Delta)]*2, [mini, maxi], '--', c=col)
        ax.plot([decale_positif(markers[1] - x0, problem.phy_prop.Delta)]*2, [mini, maxi], '--', c=col)
    if isinstance(problem.bulles, BulleTemperature):
        plot_temperature_bulles(problem, x0=x0, ax=ax, col=col, ax2=ax2, lda_gradT=lda_gradT)
    ax.set_xticks(problem.num_prop.x_f)
    ax.set_xticklabels([])
    ax.grid(b=True, which='major')
    ax.legend()
    return fig, ax, ax2


def plot_classic(problem, fig=None, ax=None, ax2=None, lda_gradT=False):
    if (fig is None) or (ax is None):
        if isinstance(problem.bulles, BulleTemperature) and lda_gradT:
            fig, (ax, ax2) = plt.subplots(2, sharex='all')
        else:
            fig, ax = plt.subplots(1)
    fig.suptitle(problem.name)
    # c = ax.plot(problem.num_prop.x, problem.I, '+')
    c = ax.plot(problem.num_prop.x, problem.T, label='%s, time %g' % (problem.name, problem.time))
    col = c[-1].get_color()
    maxi = max(np.max(problem.T), np.max(problem.I))
    mini = min(np.min(problem.T), np.min(problem.I))
    for markers in problem.bulles():
        ax.plot([markers[0]]*2, [mini, maxi], '--', c=col)
        ax.plot([markers[1]]*2, [mini, maxi], '--', c=col)
    if isinstance(problem.bulles, BulleTemperature):
        plot_temperature_bulles(problem, ax=ax, col=col, ax2=ax2, lda_gradT=lda_gradT)
    ax.set_xticks(problem.num_prop.x_f)
    ax.set_xticklabels([])
    ax.grid(b=True, which='major')
    ax.legend()
    return fig, ax, ax2


def plot_temperature_bulles(problem, x0=None, ax=None, col=None, ax2=None, quiver=False, lda_gradT=False):
    if x0 is None:
        x0 = 0.
        decale = False
    else:
        decale = True
    n = len(problem.num_prop.x)
    Delta = problem.phy_prop.Delta
    while x0 - Delta > -problem.num_prop.dx:
        x0 -= Delta
    # fig1, ax1 = plt.subplots(1)
    # lda_grad_T = interpolate_from_center_to_face_center(problem.Lda_h) * grad(problem.T, dx=problem.num_prop.dx)
    xil = []
    x0l = []
    Ti = []
    Tig = []
    Tid = []
    # print('x0 : %f' % x0)
    for i_int, x in enumerate(problem.bulles()):
        for j, xi in enumerate(x):
            i = problem.bulles.ind[i_int, j]
            # print(problem.time)
            # print('i : %d' % i)
            xil.append(xi - x0)
            # print('xi dec : %f' % xil[-1])
            x0l.append(problem.num_prop.x[i] - x0)
            # print('num_prop.x[i] : %f' % x0l[-1])
            Ti.append(problem.bulles.T[i_int, j])
            Tig.append(problem.bulles.Tg[i_int, j])
            Tid.append(problem.bulles.Td[i_int, j])
            ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(problem, i, liqu_a_gauche=(not j))
            if quiver:
                if i > 1:
                    ax.quiver(problem.num_prop.x_f[i-1]-x0, (problem.T[i-2] + problem.T[i-1])/2., 1.,
                              (problem.T[i-1] - problem.T[i-2])/problem.num_prop.dx, angles='xy')
                if problem.time > 0.:
                    ax.quiver(xi - x0, problem.bulles.T[i_int, j], 1., problem.bulles.lda_grad_T[i_int, j]/ldag,
                              0., angles='xy')
                    ax.quiver(xi - x0, problem.bulles.T[i_int, j], 1., problem.bulles.lda_grad_T[i_int, j]/ldad,
                              1., angles='xy')
                if i < n-1:
                    ax.quiver(problem.num_prop.x_f[i+2]-x0, (problem.T[i+2] + problem.T[i+1])/2., 1.,
                              (problem.T[i+2] - problem.T[i+1])/problem.num_prop.dx, angles='xy')
            cells_suivi = problem.bulles.cells[i_int, j]
            if isinstance(cells_suivi, CellsSuiviInterface):
                ax.plot(cells_suivi.xj + problem.num_prop.x[i], cells_suivi.Tj,
                        '--', label='Tj interp', c=col)
    if problem.time > 0. and quiver:
        ax.plot(xil, Ti, 'k+')
        ax.plot(x0l, Tig, '+', label=r'$T_g$')
        ax.plot(x0l, Tid, '+', label=r'$T_d$')
    if decale:
        xf_dec, lda_grad_T_dec = decale_perio(problem.num_prop.x_f, problem.flux_diff, x0=x0)
        if lda_gradT:
            ax2.plot(xf_dec, lda_grad_T_dec, label=r'$\lambda \nabla T$')
    else:
        if lda_gradT:
            ax2.plot(problem.num_prop.x_f, problem.flux_diff, label=r'$\lambda \nabla T$')
            ax2.plot(problem.bulles.markers.flatten() - x0, problem.bulles.lda_grad_T.flatten(),
                     '+', label=r'$\lambda \nabla T_I$')
    if lda_gradT:
        ax2.legend()
        ax2.set_xticks(problem.num_prop.x_f)
        ax2.set_xticklabels([])
        ax2.grid(b=True, which='both')


def decale_perio(x, T, x0=0., markers=None, plot=False):
    """
    décale de x0 vers la gauche la courbe T en interpolant entre les décalages direct de n*dx < x0 < (n+1)*dx
    avec la formule suivante : x_interp += (x0-n*dx)

    Args:
        markers:
        plot:
        x:
        T:
        x0:

    Returns:
        x et T decalé
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
            for couple_marker in markers:
                plt.plot([decale_positif(couple_marker[0] - n*dx, Delta)] * 2, [mini, maxi], '--')
                plt.plot([decale_positif(couple_marker[1] - n*dx, Delta)] * 2, [mini, maxi], '--')
                plt.plot([decale_positif(couple_marker[0] - (n+1)*dx, Delta)] * 2, [mini, maxi], '--')
                plt.plot([decale_positif(couple_marker[1] - (n+1)*dx, Delta)] * 2, [mini, maxi], '--')
                plt.plot([decale_positif(couple_marker[0] - x0, Delta)]*2, [mini, maxi], '--')
                plt.plot([decale_positif(couple_marker[1] - x0, Delta)] * 2, [mini, maxi], '--')
    return x_decale, T_decale


def decale_positif(mark, Delta):
    while mark < 0.:
        mark += Delta
    return mark
