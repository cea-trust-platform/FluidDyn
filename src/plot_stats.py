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

from src.main import *
from src.statistics import Statistics
from src.time_problem import TimeProblem

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


class TimePlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1)
        self.fig.set_size_inches(9.5, 5)
        self.ax.minorticks_on()
        self.ax.grid(b=True, which="major")
        self.ax.grid(b=True, which="minor", alpha=0.2)
        self.ax.set_xlabel(r"$t [s]$")

    def plot(self, t, e, label=None, **kwargs):
        c = self.ax.plot(t, e, label=label, **kwargs)
        if label is not None:
            self.ax.legend(loc="upper right")
        self.fig.tight_layout()

        print()
        if label is not None:
            print(label)
            print("=" * len(label))
        else:
            print("Calcul sans label")
            print("=================")

        return c


class EnergiePlot(TimePlot):
    def __init__(self, e0=None):
        super().__init__()
        self.ax.set_ylabel(r"$E_{tot} [J/m^3]$")
        self.e0 = e0

    def plot(self, t, e, label=None, **kargs):
        if self.e0 is None:
            self.e0 = e[0]

        super().plot(t, e, label, **kargs)

        n = len(e)
        i0 = int(n / 5)
        dedt_adim = (
            (e[-1] - e[i0]) / (t[-1] - t[i0]) * (t[1] - t[0]) / self.e0
        )  # on a mult
        print("dE*/dt* = %g" % dedt_adim)

    def plot_pb(self, pb: Problem, fac=None, label=None):
        if fac is None:
            fac = pb.phy_prop.Delta * pb.phy_prop.dS
        if label is None:
            label = pb.name
        self.plot(pb.t, pb.E / fac, label=label)

    def plot_stat(self, stat, label=None):
        self.plot(stat.t, stat.E, label=label)

    def plot_tpb(self, tpb: TimeProblem, label=None):
        if label is None:
            label = tpb.problem_state.name
        self.plot(tpb.stat.t, tpb.stat.E, label=label)

    def add_E0(self):
        self.fig.canvas.draw_idle()
        labels = [item.get_text() for item in self.ax.get_yticklabels()]
        ticks = list(self.ax.get_yticks())
        ticks.append(self.e0)
        labels.append(r"$E_0$")
        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels(labels)
        self.fig.tight_layout()


class TemperaturePlot(TimePlot):
    def __init__(self):
        super().__init__()
        self.ax.set_ylabel(r"$T_{k} [K]$")
        self.T_final = None
        self.T_final_prevu = None

    def plot(self, t, T, label=None, **kargs):
        c = super().plot(t, T, label, **kargs)

        n = len(T)
        i0 = int(n / 5)
        dedt_adim = (T[-1] - T[i0]) / (t[-1] - t[i0])  # on a mult
        print("dT/dt = %g" % dedt_adim)
        return c

    def plot_stat(self, stat: Statistics, state_pb: StateProblem, label=None):
        if self.T_final is None:
            self.T_final = state_pb.T_final
            self.T_final_prevu = state_pb.T_final_prevu
        if label is None:
            label = ", " + state_pb.name
        c = self.plot(stat.t, stat.Tl, label=r"$T_l$" + label)
        self.plot(stat.t, stat.Tv, label=r"$T_v$" + label, c=c[-1].get_color())

    def plot_tpb(self, timeproblem: TimeProblem, label=None):
        self.plot_stat(timeproblem.stat, timeproblem.problem_state, label=label)

    def add_T_final(self):
        self.fig.canvas.draw_idle()
        labels = [item.get_text() for item in self.ax.get_yticklabels()]
        ticks = list(self.ax.get_yticks())
        ticks.append(self.T_final)
        labels.append(r"$T_f\quad\cdot$")
        ticks.append(self.T_final_prevu)
        labels.append(r"$T_f^\textrm{expected}\quad\cdot$")
        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels(labels)
        self.fig.tight_layout()
