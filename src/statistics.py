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

import numpy as np
from src.main import StateProblem


class Statistics:
    def __init__(self):
        self.time_stats = {
            "t": TimeStatArray(),
            "E": TimeStatArray(),
            "El": TimeStatArray(),
            "Ev": TimeStatArray(),
            "El_pure": TimeStatArray(),
            "Ev_pure": TimeStatArray(),
            "T": TimeStatArray(),
            "Tl": TimeStatArray(),
            "Tv": TimeStatArray(),
            "Tl_pure": TimeStatArray(),
            "Tv_pure": TimeStatArray(),
        }
        self.n = 0
        self.step = 0

    def copy(self, other):
        self.n = other.n
        self.step = other.step
        for key in self.time_stats.keys():
            if key in other.time_stats.keys():
                self.time_stats[key].data = other.time_stats[key].data.copy()
            else:
                print(key, "was not saved in", other, "object")
                self.time_stats[key].alloc(self.n + 1)

    def collect(self, pb: StateProblem):
        self.time_stats["t"][self.step] = pb.time
        self.time_stats["E"][self.step] = pb.energy
        self.time_stats["El"][self.step] = np.sum(
            pb.T * pb.rho_cp.a(pb.I) * pb.I
        ) / np.sum(pb.I)
        self.time_stats["Ev"][self.step] = np.sum(
            pb.T * pb.rho_cp.a(pb.I) * (1.0 - pb.I)
        ) / np.sum(1.0 - pb.I)
        self.time_stats["El_pure"][self.step] = np.sum(
            np.where(pb.I == 1.0, pb.T * pb.rho_cp.a(pb.I), 0.0)
        ) / np.sum(np.where(pb.I == 1.0, 1.0, 0.0))
        self.time_stats["Ev_pure"][self.step] = np.sum(
            np.where(pb.I == 0.0, pb.T * pb.rho_cp.a(pb.I), 0.0)
        ) / np.sum(np.where(pb.I == 0.0, 1.0, 0.0))
        self.time_stats["T"][self.step] = np.sum(pb.T) / pb.T.size
        self.time_stats["Tl"][self.step] = np.sum(pb.T * pb.I) / np.sum(pb.I)
        self.time_stats["Tv"][self.step] = np.sum(pb.T * (1.0 - pb.I)) / np.sum(
            1.0 - pb.I
        )
        self.time_stats["Tl_pure"][self.step] = np.sum(
            np.where(pb.I == 1.0, pb.T, 0.0)
        ) / np.sum(np.where(pb.I == 1.0, 1.0, 0.0))
        self.time_stats["Tv_pure"][self.step] = np.sum(
            np.where(pb.I == 0.0, pb.T, 0.0)
        ) / np.sum(np.where(pb.I == 0.0, 1.0, 0.0))
        self.step += 1

    def extend(self, longer_n):
        self.n = longer_n
        if self.E.size == 0:
            self.step = 0
            for stat in self.time_stats.values():
                stat.alloc(longer_n + 1)
        else:
            self.step = self.E.size - 1
            for stat in self.time_stats.values():
                stat.extend(longer_n)

    def __getattr__(self, item):
        return self.time_stats[item].data


class TimeStatArray:
    def __init__(self):
        self.data = np.array([])  # type: np.ndarray

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def alloc(self, n):
        self.data = np.zeros((n,))

    def extend(self, n):
        self.data = np.r_[self.data, np.zeros((n,))]
