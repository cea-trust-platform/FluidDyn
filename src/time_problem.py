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

import pickle
from copy import deepcopy
from glob import glob

from src.main import StateProblem
from src.plot_fields import Plotter
from src.statistics import Statistics
from src.time_scheme import EulerTimestep, RK3Timestep, TimestepBase


class TimeProblem:
    timestep_scheme: TimestepBase
    problem_state: StateProblem
    stat: Statistics
    plotter: Plotter

    def __init__(self, problem_state=None, stat=None, plotter=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if problem_state is None:
            problem_state = StateProblem(*args, **kwargs)
        self.problem_state = problem_state
        if stat is None:
            stat = Statistics()
        self.timestep_scheme = self._init_timestep(
            self.problem_state.num_prop.time_scheme
        )
        self.stat = stat
        if plotter is None:
            plotter = Plotter()
        self.plotter = plotter

    @staticmethod
    def _init_timestep(time_scheme_name: str):
        if time_scheme_name == "euler":
            timestep_method = EulerTimestep()
        elif time_scheme_name == "rk3":
            timestep_method = RK3Timestep()
        else:
            raise NotImplementedError
        return timestep_method

    def copy(self, pb):
        self.problem_state.copy(pb.problem_state)
        self.stat = deepcopy(pb.stat)

    def timestep(
        self,
        n=None,
        t_fin=None,
        plot_for_each=1,
        number_of_plots=None,
        plotter=None,
        debug=None,
        **kwargs
    ):
        n = self._get_iteration_number(n, t_fin)

        if plotter is None:
            plotter = [self.plotter]
        if not isinstance(plotter, list):
            plotter = [plotter]

        plot_for_each = self._get_plot_for_each(n, plot_for_each, number_of_plots)

        self.stat.extend(n)
        self.stat.collect(self.problem_state)

        for i in range(n):
            self.timestep_scheme.step(
                self.problem_state, debug=debug, bool_debug=(i % plot_for_each == 0)
            )
            self.stat.collect(self.problem_state)
            # intermediary plots
            if (i % plot_for_each == 0) and (i != 0) and (i != n - 1):
                for plott in plotter:
                    plott.plot(self.problem_state, **kwargs)

        # final plot
        for plott in plotter:
            plott.plot(self.problem_state, **kwargs)

        return self.stat.t, self.stat.E

    def _get_iteration_number(self, n=None, t_fin=None):
        if (n is None) and (t_fin is None):
            raise NotImplementedError
        elif (n is not None) and (t_fin is not None):
            n = min(n, int(t_fin / self.problem_state.dt))
        elif t_fin is not None:
            n = int(t_fin / self.problem_state.dt)
        return n

    @staticmethod
    def _get_plot_for_each(n, plot_for_each=1, number_of_plots=None):
        if number_of_plots is not None:
            plot_for_each = int((n - 1) / number_of_plots)
        if plot_for_each <= 0:
            plot_for_each = 1
        return plot_for_each

    def load_or_compute(self, pb_name=None, t_fin=0.0, **kwargs):
        if pb_name is None:
            pb_name = self.problem_state.full_name

        simu_name = SimuName(pb_name)
        closer_simu = simu_name.get_closer_simu(self.problem_state.time + t_fin)

        if closer_simu is not None:
            with open(closer_simu, "rb") as f:
                saved = pickle.load(f)
            self.problem_state.copy(saved)
            launch_time = t_fin - self.problem_state.time
            print(
                "Loading ======> %s\nremaining time to compute : %f"
                % (closer_simu, launch_time)
            )
        else:
            launch_time = t_fin - self.problem_state.time

        t, E = self.timestep(t_fin=launch_time, **kwargs)

        save_name = simu_name.get_save_path(self.problem_state.time)
        with open(save_name, "wb") as f:
            pickle.dump(self.problem_state, f)
        with open("statistics_" + save_name, "wb") as f:
            pickle.dump(self.stat, f)

        return t, E


class SimuName:
    def __init__(self, name: str, directory=None):
        self._name = name
        if directory is None:
            self.directory = "../References"
        else:
            self.directory = directory

    @property
    def name(self):
        return self._name

    def get_closer_simu(self, t: float):
        simu_list = glob(self.directory + "/" + self.name + "_t_" + "*" + ".pkl")
        print("Liste des simus similaires : ")
        print(simu_list)
        closer_time = 0.0
        closer_simu = None
        for simu in simu_list:
            time = self._get_time(simu)
            if (time <= t) & (time > closer_time):
                closer_time = time
                closer_simu = simu
        remaining_running_time = t - closer_time
        assert remaining_running_time >= 0.0
        return closer_simu  # , remaining_running_time

    @staticmethod
    def _get_time(path_to_save_file: str) -> float:
        time = path_to_save_file.split("_t_")[-1].split(".pkl")[0]
        return round(float(time), 6)

    def get_save_path(self, t) -> str:
        return self.directory + "/" + self.name + "_t_%f" % round(t, 6) + ".pkl"
