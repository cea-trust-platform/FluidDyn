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

import math
import re
from flu1ddyn.main import *
from flu1ddyn.main_discontinu import get_prop, BulleTemperature, StateProblemDiscontinu

from matplotlib import rc
import matplotlib.pyplot as plt

rc("text", usetex=True)
rc("font", size=18)
rc("font", family="serif")
rc("legend", fontsize=16)
rc("figure", max_open_warning=50)
# rc("figure", figsize=(19, 8))
# rc("figure", dpi=200)
rc("savefig", dpi=300)
rc("legend", loc="upper right")


class Plotter:
    """
    Cette classe permet de tracer facilement les résultats d'une simulation.
    Elle trace le domaine qui va de 0. à Delta (le premier point de température
    est en dx/2. et le premier flux en 0.).
    """

    def __init__(
        self,
        cas="classic",
        lda_gradT=False,
        flux_conv=False,
        time=True,
        markers=True,
        zoom=None,
        dx=False,
        dt=False,
        ispretty=True,
        **kwargs
    ):
        self._cas = cas
        self.fig = None
        self.ax = None
        self.ax2 = None
        self.ax3 = None
        self.bulles_poly = None
        self.ymini = None
        self.ymaxi = None
        self.bulles_poly2 = None
        self.ymini2 = None
        self.ymaxi2 = None
        self.lda_gradT = lda_gradT
        self.flux_conv = flux_conv
        self.time = time
        self.dx = dx
        self.dt = dt
        self.markers = markers
        self.zoom = zoom
        self.ispretty = ispretty
        self.kwargs = kwargs

    @property
    def cas(self):
        return self._cas

    @cas.setter
    def cas(self, value):
        self._cas = value
        print("plotter mode changed to %s" % value)

    def plot(self, problem, plot_Ti=False, dlabel=None, **kwargs):
        first_plot = False
        if "label" not in kwargs.keys():
            kwargs["label"] = dlabel
        # Set up of fig and ax
        if (self.fig is None) or (self.ax is None):
            if self.lda_gradT or self.flux_conv:
                self.fig, (self.ax, self.ax2) = plt.subplots(
                    2, sharex="all", **self.kwargs
                )
                self.fig.set_size_inches(9.5, 7)
                self.ax2.minorticks_on()
            else:
                self.fig, self.ax = plt.subplots(1)
                self.fig.set_size_inches(9.5, 5)
            if self.lda_gradT and self.flux_conv:
                self.ax3 = self.ax2.twinx()
            elif not self.lda_gradT and self.flux_conv:
                self.ax3 = self.ax2
                self.ax3.minorticks_on()
                self.ax2 = None
            self.ax.minorticks_on()
            first_plot = True

        # Set up label
        if kwargs["label"] is None:
            lab = "%s" % (problem.name.replace("_", " "))
            if self.time:
                lab += ", time %.2g" % problem.time
            if self.dx:
                leg = to_scientific("%.2e" % problem.num_prop.dx)
                lab += r", $\Delta x = %s$" % leg
            if self.dt:
                leg = to_scientific("%.2e" % problem.dt)
                lab += r", $\Delta t = %s$" % leg
            kwargs["label"] = lab

        # Set up decalage
        x0 = problem.time * problem.phy_prop.v
        while x0 > problem.phy_prop.Delta:
            x0 -= problem.phy_prop.Delta
        if self.cas == "classic":
            x0 = 0.0

        # Plot T
        self.fig, self.ax, c = plot_temp(
            problem, x0=x0, fig=self.fig, ax=self.ax, **kwargs
        )

        # Plot lda gradT
        if self.lda_gradT and (self.ax2 is not None):
            xf_dec, lda_grad_T_dec = decale_perio(
                problem.num_prop.x_f,
                problem.flux_diff,
                Delta=problem.phy_prop.Delta,
                x0=x0,
            )
            if "c" not in kwargs.keys():
                kwargs["c"] = c
            self.ax2.plot(xf_dec, lda_grad_T_dec, **kwargs)

        # Plot flux conv
        if self.flux_conv and (self.ax3 is not None):
            xf_dec, flux_conv_dec = decale_perio(
                problem.num_prop.x_f,
                problem.flux_conv,
                Delta=problem.phy_prop.Delta,
                x0=x0,
            )
            if "c" not in kwargs.keys():
                kwargs["c"] = c
            kwargs["label"] = self.flux_conv
            self.ax3.plot(xf_dec, flux_conv_dec, "--", **kwargs)

        ticks_major, ticks_minor, M1, Dx = get_ticks(problem, x0=x0)

        # Plot markers
        if (
            isinstance(problem.bulles, BulleTemperature)
            and self.markers
            and not self.ispretty
        ):
            plot_temperature_bulles(
                problem,
                x0=x0,
                ax=self.ax,
                ax2=self.ax2,
                lda_gradT=self.lda_gradT,
                plot_Ti=plot_Ti,
                color=c,
            )

        self.ax.legend(loc="upper right")
        if first_plot:
            self.ymini, self.ymaxi = self.ax.get_ylim()
            self.ax.set_ymargin(0.0)
            if not self.ispretty:
                self.ax.set_xticks([], minor=False)
                self.ax.set_xticks(problem.num_prop.x_f, minor=True)
                self.ax.set_xticklabels([], minor=False)
                self.ax.set_xticklabels([], minor=True)
            else:
                self.ax.set_xticks(ticks_major, minor=False)
                self.ax.set_xticks(ticks_minor, minor=True)
                self.ax.set_xticklabels(
                    np.rint((ticks_major - M1) / Dx).astype(int), minor=False
                )
            if self.zoom is not None:
                z0 = M1 + self.zoom[0] * Dx
                z1 = M1 + self.zoom[1] * Dx
                self.ax.set_xlim(z0, z1)
            self.ax.grid(b=True, which="major")
            self.ax.grid(b=True, which="minor", alpha=0.2)
            self.ax.set_ylabel(r"$T$", size="x-large")
            for markers in problem.bulles():
                bulle0 = decale_positif(markers[0] - x0, problem.phy_prop.Delta)
                bulle1 = decale_positif(markers[1] - x0, problem.phy_prop.Delta)
                self.bulles_poly = self.ax.fill_between(
                    [bulle0, bulle1],
                    [self.ymini] * 2,
                    [self.ymaxi] * 2,
                    color="grey",
                    alpha=0.2,
                )
            if self.ax2 is not None:
                self.ymini2, self.ymaxi2 = self.ax.get_ylim()
                self.ax2.set_ymargin(0.0)
                # if not self.ispretty:
                #     self.ax2.set_xticks(problem.num_prop.x_f, minor=True)
                #     self.ax2.set_xticklabels([], minor=True)
                # else:
                #     self.ax2.set_xticks(ticks_major, minor=False)
                #     self.ax2.set_xticks(ticks_minor, minor=True)
                #     self.ax2.set_xticklabels(
                #         np.rint((ticks_major - M1) / Dx).astype(int), minor=False
                #     )
                self.ax2.set_xlabel(r"$x / D_b$", size="x-large")
                self.ax2.set_ylabel(r"$\lambda \nabla T$", size="x-large")
                self.ax2.grid(b=True, which="major")
                self.ax2.grid(b=True, which="minor", alpha=0.2)
                # if self.zoom is not None:
                #     self.ax2.set_xlim(z0, z1)
                for markers in problem.bulles():
                    bulle0 = decale_positif(markers[0] - x0, problem.phy_prop.Delta)
                    bulle1 = decale_positif(markers[1] - x0, problem.phy_prop.Delta)
                    self.bulles_poly2 = self.ax2.fill_between(
                        [bulle0, bulle1],
                        [self.ymini2] * 2,
                        [self.ymaxi2] * 2,
                        color="grey",
                        alpha=0.2,
                    )
            elif self.ax3 is not None:
                self.ymini2, self.ymaxi2 = self.ax.get_ylim()
                self.ax3.set_ymargin(0.0)
                self.ax3.set_xlabel(r"$x / D_b$", size="x-large")
                self.ax3.set_ylabel(r"$\rho C_p T$", size="x-large")
                self.ax3.grid(b=True, which="major")
                self.ax3.grid(b=True, which="minor", alpha=0.2)
                for markers in problem.bulles():
                    bulle0 = decale_positif(markers[0] - x0, problem.phy_prop.Delta)
                    bulle1 = decale_positif(markers[1] - x0, problem.phy_prop.Delta)
                    self.bulles_poly2 = self.ax3.fill_between(
                        [bulle0, bulle1],
                        [self.ymini2] * 2,
                        [self.ymaxi2] * 2,
                        color="grey",
                        alpha=0.2,
                    )
            else:
                self.ax.set_xlabel(r"$x / D_b$", size="x-large")
        # if self.ax2 is not None:
        #     self.ax2.legend(loc='upper right')
        # if self.ax3 is not None:
        #     self.ax3.legend(loc='lower right')
        self.fig.tight_layout()

        # Zone de bulles mise à jour à chaque nouvelle courbe tracée
        mini, maxi = self.ax.get_ylim()
        self.ymini = min(self.ymini, mini)
        self.ymaxi = max(self.ymaxi, maxi)
        delta = self.ymaxi - self.ymini
        margin = 0.05
        if self.ax2 is not None:
            mini, maxi = self.ax2.get_ylim()
            self.ymini2 = min(self.ymini2, mini)
            self.ymaxi2 = max(self.ymaxi2, maxi)
            delta2 = self.ymaxi2 - self.ymini2
        elif self.ax3 is not None:
            mini, maxi = self.ax3.get_ylim()
            self.ymini2 = min(self.ymini2, mini)
            self.ymaxi2 = max(self.ymaxi2, maxi)
            delta2 = self.ymaxi2 - self.ymini2
        else:
            delta2 = 1.0  # arbitraire, delta2 ne sert pas

        for markers in problem.bulles():
            bulle0 = decale_positif(markers[0] - x0, problem.phy_prop.Delta)
            bulle1 = decale_positif(markers[1] - x0, problem.phy_prop.Delta)
            # self.ax.plot([bulle0] * 2, [self.ymini, self.ymaxi], c='black', lw=0.2)
            # self.ax.plot([bulle1] * 2, [self.ymini, self.ymaxi], c='black', lw=0.2)
            self.bulles_poly.remove()
            self.bulles_poly = self.ax.fill_between(
                [bulle0, bulle1],
                [self.ymini - margin * delta] * 2,
                [self.ymaxi + margin * delta] * 2,
                color="grey",
                alpha=0.2,
            )
            if self.ax2 is not None:
                # self.ax2.plot([bulle0] * 2, [self.ymini2, self.ymaxi2], c='black', lw=0.2)
                # self.ax2.plot([bulle1] * 2, [self.ymini2, self.ymaxi2], c='black', lw=0.2)
                self.bulles_poly2.remove()
                self.bulles_poly2 = self.ax2.fill_between(
                    [bulle0, bulle1],
                    [self.ymini2 - margin * delta2] * 2,
                    [self.ymaxi2 + delta2 * margin] * 2,
                    color="grey",
                    alpha=0.2,
                )
            elif self.ax3 is not None:
                self.bulles_poly2.remove()
                self.bulles_poly2 = self.ax3.fill_between(
                    [bulle0, bulle1],
                    [self.ymini2 - margin * delta2] * 2,
                    [self.ymaxi2 + delta2 * margin] * 2,
                    color="grey",
                    alpha=0.2,
                )

        self.ax.set_ymargin(0.0)
        if self.ax2 is not None:
            self.ax2.set_ymargin(0.0)
        elif self.ax3 is not None:
            self.ax3.set_ymargin(0.0)
        self.fig.tight_layout()
        return c


def plot_temp(problem: StateProblem, fig=None, x0=0.0, ax=None, label=None, **kwargs):
    # fig.suptitle(problem.name.replace('_', ' '))
    x_dec, T_dec = decale_perio(
        problem.num_prop.x, problem.T, x0=x0, markers=problem.bulles
    )
    c = ax.plot(x_dec, T_dec, label=label, **kwargs)
    col = c[-1].get_color()
    # maxi = max(np.max(problem.T), np.max(problem.I))
    # mini = min(np.min(problem.T), np.min(problem.I))
    return fig, ax, col


# def plot_classic(problem, fig=None, ax=None, label=None):
#     # fig.suptitle(problem.name.replace('_', '-'))
#     # c = ax.plot(problem.num_prop.x, problem.I, '+')
#     if label is None:
#         label = '%s' % problem.name.replace('_', ' ')
#     c = ax.plot(problem.num_prop.x, problem.T, label=label)
#     col = c[-1].get_color()
#     maxi = max(np.max(problem.T), np.max(problem.I))
#     mini = min(np.min(problem.T), np.min(problem.I))
#     for markers in problem.bulles():
#         ax.plot([markers[0]]*2, [mini, maxi], '--', c=col)
#         ax.plot([markers[1]]*2, [mini, maxi], '--', c=col)
#     return fig, ax


def plot_temperature_bulles(
    problem: StateProblemDiscontinu,
    x0=0.0,
    ax=None,
    ax2=None,
    quiver=False,
    lda_gradT=False,
    plot_Ti=False,
    color=None,
):
    n = len(problem.num_prop.x)
    Delta = problem.phy_prop.Delta
    while x0 - Delta > -problem.num_prop.dx:
        x0 -= Delta
    xil = []
    x0l = []
    Ti = []
    Tig = []
    Tid = []
    lda_grad_Ti = []
    for i_int, x in enumerate(problem.bulles()):
        for j, xi in enumerate(x):
            i = problem.bulles.ind[i_int, j]
            xil.append(xi - x0)
            x0l.append(problem.num_prop.x[i] - x0)
            Ti.append(problem.bulles.T[i_int, j])
            Tig.append(problem.bulles.Tg[i_int, j])
            Tid.append(problem.bulles.Td[i_int, j])
            lda_grad_Ti.append(problem.bulles.lda_grad_T[i_int, j])
            ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(
                problem, i, liqu_a_gauche=(not j)
            )
            if quiver and (ax is not None):
                if i > 1:
                    ax.quiver(
                        problem.num_prop.x_f[i - 1] - x0,
                        (problem.T[i - 2] + problem.T[i - 1]) / 2.0,
                        1.0,
                        (problem.T[i - 1] - problem.T[i - 2]) / problem.num_prop.dx,
                        angles="xy",
                    )
                if problem.time > 0.0:
                    ax.quiver(
                        xi - x0,
                        problem.bulles.T[i_int, j],
                        1.0,
                        problem.bulles.lda_grad_T[i_int, j] / ldag,
                        0.0,
                        angles="xy",
                    )
                    ax.quiver(
                        xi - x0,
                        problem.bulles.T[i_int, j],
                        1.0,
                        problem.bulles.lda_grad_T[i_int, j] / ldad,
                        1.0,
                        angles="xy",
                    )
                if i < n - 1:
                    ax.quiver(
                        problem.num_prop.x_f[i + 2] - x0,
                        (problem.T[i + 2] + problem.T[i + 1]) / 2.0,
                        1.0,
                        (problem.T[i + 2] - problem.T[i + 1]) / problem.num_prop.dx,
                        angles="xy",
                    )
            # cells_suivi = problem.bulles.cells[2 * i_int + j]
            # if isinstance(cells_suivi, CellsSuiviInterface) and (ax is not None):
            #     ax.plot(cells_suivi.xj + problem.num_prop.x[i], cells_suivi.Tj,
            #             '--', label='Tj interp', c=col)
            # if ax is not None:
            #     ax.plot(
            #         problem.bulles.markers.flatten() - x0,
            #         problem.bulles.Ti.flatten(),
            #         "+",
            #     )  # , label=r'$T_I$')
    if problem.time > 0.0 and plot_Ti and (ax is not None):
        ax.plot(xil, Ti, c=color, ls="", marker="+", ms=7.0)  # , label=r"$T_I$")
        ax.plot(x0l, Tig, c=color, ls="", marker="^", ms=6.0)  # , label=r"$T_g$")
        ax.plot(x0l, Tid, c=color, ls="", marker="s", ms=6.0)  # , label=r"$T_d$")
    if problem.time > 0.0 and lda_gradT and plot_Ti and (ax2 is not None):
        ax2.plot(
            xil, lda_grad_Ti, marker="+", ls="", c=color, ms=7.0
        )  # , label=r"$\lambda \nabla T_I$")
        # ax2.set_xticks(problem.num_prop.x_f)
        # ax2.set_xticklabels([])
        # ax2.grid(b=True, which='major')
        # ax2.grid(b=True, which='minor', alpha=0.2)


def align_y_axis(ax1, ax2):
    ylim1 = ax1.axes.get_ylim()
    ylim2 = ax2.axes.get_ylim()
    coeff1 = -ylim1[0] / (ylim1[1] - ylim1[0])
    coeff2 = -ylim2[0] / (ylim2[1] - ylim2[0])
    coeff = (coeff1 + coeff2) / 2.0
    print("coeff : ", coeff)
    if np.abs(coeff - 0.5) > 0.5:
        print("La situation n est pas adaptée à l alignement des zéros")
        return
    # coeff prend ses valeurs entre 0 et 1
    if coeff > coeff1:
        # On veut monter le zéro relatif en diminuant la valeur de y1[0]
        new_y0 = coeff / (coeff - 1.0) * ylim1[1]
        ax1.set_ylim(new_y0, ylim1[1])
    else:
        # On veut descendre le zéro relatif en augmentant la valeur de y1[1]
        new_y1 = (coeff - 1.0) / coeff * ylim1[0]
        ax1.set_ylim(ylim1[0], new_y1)
    if coeff > coeff2:
        # On veut monter le zéro relatif en diminuant la valeur de y2[0]
        new_y0 = coeff / (coeff - 1.0) * ylim2[1]
        ax2.set_ylim(new_y0, ylim2[1])
    else:
        # On veut descendre le zéro relatif en augmentant la valeur de y2[1]
        new_y1 = (coeff - 1.0) / coeff * ylim2[0]
        ax2.set_ylim(ylim2[0], new_y1)


def get_ticks(problem: StateProblem, x0=0.0):
    Delta = problem.phy_prop.Delta
    M1, M2 = problem.bulles.markers[0] - x0
    if M2 < M1:
        M2 += Delta
    assert M2 > M1
    Dx_major = M2 - M1
    while M1 < 0.0:
        M1 += Delta
        M2 += Delta
    if M2 > Delta:
        M2 -= Delta
    assert Dx_major > 0.0
    Dx_minor = Dx_major / 4.0
    ticks_major = []
    ticks_minor = []
    mark = M1
    while mark > 0.0:
        ticks_minor = [mark] + ticks_minor
        mark -= Dx_minor
    mark = M1 + Dx_minor
    while mark < Delta:
        ticks_minor = ticks_minor + [mark]
        mark += Dx_minor
    mark = M1
    while mark > 0.0:
        ticks_major = [mark] + ticks_major
        mark -= Dx_major
    mark = M1 + Dx_major
    while mark < problem.phy_prop.Delta:
        ticks_major = ticks_major + [mark]
        mark += Dx_major
    ticks_minor = np.array(ticks_minor)
    ticks_major = np.array(ticks_major)
    # print('ticks ', ticks_major)
    return ticks_major, ticks_minor, M1, Dx_major


def decale_perio(x, T, Delta=None, x0=0.0, markers=None, plot=False):
    """
    décale de x0 vers la gauche la courbe T en interpolant entre les décalages direct de n*dx < x0 < (n+1)*dx
    avec la formule suivante : x_interp += (x0-n*dx)

    Args:
        markers:
        plot:
        x:
        T:
        Delta:
        x0:

    Returns:
        x et T decalé
    """
    dx = x[1] - x[0]
    # Dans ce cas là on considère qu'on trace des valeurs au centre des mailles (Température)
    if Delta is None:
        Delta = x[-1] + dx / 2.0
    while x0 > Delta:
        x0 -= Delta
    n = math.ceil(x0 / dx - 10**-9)
    T_decale = np.r_[T[n:], T[:n]]
    x_decale = x - (x0 - n * dx)
    if plot:
        plt.figure()
        plt.plot(x, T_decale, label="Tn")
        T_np1 = np.r_[T[n + 1 :], T[: n + 1]]
        plt.plot(x, T_np1, label="Tnp1")
        plt.plot(x_decale, T_decale, label="T decale")

        if markers is not None:
            mini = np.min(T_decale)
            maxi = np.max(T_decale)
            for couple_marker in markers:
                plt.plot(
                    [decale_positif(couple_marker[0] - n * dx, Delta)] * 2,
                    [mini, maxi],
                    "--",
                )
                plt.plot(
                    [decale_positif(couple_marker[1] - n * dx, Delta)] * 2,
                    [mini, maxi],
                    "--",
                )
                plt.plot(
                    [decale_positif(couple_marker[0] - (n + 1) * dx, Delta)] * 2,
                    [mini, maxi],
                    "--",
                )
                plt.plot(
                    [decale_positif(couple_marker[1] - (n + 1) * dx, Delta)] * 2,
                    [mini, maxi],
                    "--",
                )
                plt.plot(
                    [decale_positif(couple_marker[0] - x0, Delta)] * 2,
                    [mini, maxi],
                    "--",
                )
                plt.plot(
                    [decale_positif(couple_marker[1] - x0, Delta)] * 2,
                    [mini, maxi],
                    "--",
                )
    return x_decale, T_decale


class Animate:
    def __init__(self, tpb, ax: plt.Axes):
        self.tpb = tpb
        self.ax = ax
        self.ax.minorticks_on()
        self.ax.grid(visible=True, which="major")
        self.ax.grid(visible=True, which="minor", alpha=0.2)
        self.ax.set_xlim(self.tpb.problem_state.x[0], self.tpb.problem_state.x[-1])
        self.ax.set_ylim(0.0, 1.2)
        (self.line,) = self.ax.plot([], [])

    def __call__(self, i):
        self.tpb.timestep(n=1)
        self.line.set_data(self.tpb.problem_state.x, self.tpb.problem_state.T)
        return (self.line,)


class Compare:
    def __init__(
        self,
        tpbs,
        ax: plt.Axes,
        ylim=(None, None),
        n_dt_per_frame=None,
        run_time=None,
        n_frames=15,
    ):
        self.tpbs = tpbs
        self.ax = ax
        self.ax.minorticks_on()
        self.ax.grid(visible=True, which="major")
        self.ax.grid(visible=True, which="minor", alpha=0.2)
        self.ax.set_xlim(
            self.tpbs[0].problem_state.x[0], self.tpbs[0].problem_state.x[-1]
        )
        self.ax.set_ylim(*ylim)
        if n_dt_per_frame is None and run_time is None:
            n_dt_per_frame = 1
        if run_time is not None:
            self.dt = run_time / n_frames
            assert self.dt > max([tpb.dt for tpb in self.tpbs])
        else:
            self.dt = n_dt_per_frame * max([tpb.dt for tpb in self.tpbs])

        self.lines = []
        for _ in self.tpbs:
            (line,) = self.ax.plot([], [])
            self.lines.append(line)

    def __call__(self, i):
        for j, tpb in enumerate(self.tpbs):
            time_fin = i * self.dt
            dt = time_fin - tpb.problem_state.time
            tpb.timestep(t_fin=dt)
            spb = tpb.problem_state
            self.lines[j].set_data(spb.x, spb.T)
        return (self.lines,)


def decale_positif(mark, Delta):
    while mark < 0.0:
        mark += Delta
    return mark


def to_scientific(leg):
    leg = leg.split("e")
    if len(leg) == 2:
        leg[1] = re.sub("^-0*", "-", leg[1])
        leg[1] = re.sub("^0*", "", leg[1])
        leg = leg[0] + r"\times 10^{" + leg[1] + "}"
    else:
        leg = leg[0]
    return leg
