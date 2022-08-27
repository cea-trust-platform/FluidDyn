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
        self.t = None
        self.E = None
        self.Tl = None
        self.Tv = None
        self.n = 0
        self.step = 0

    def collect(self, pb: StateProblem):
        self.t[self.step] = pb.time
        self.E[self.step] = pb.energy
        self.Tl[self.step] = np.sum(pb.T * pb.I) / np.sum(pb.I)
        self.Tv[self.step] = np.sum(pb.T * (1.-pb.I)) / np.sum(1. - pb.I)
        self.step += 1

    def extend(self, longer_n):
        assert longer_n > self.n
        self.n = longer_n
        if self.E is None:
            self.step = 0
            self.t = np.zeros((longer_n + 1,))
            self.E = np.zeros((longer_n + 1,))
            self.Tl = np.zeros((longer_n + 1,))
            self.Tv = np.zeros((longer_n + 1,))
        else:
            self.step = self.E.size - 1
            self.E = np.r_[self.E, np.zeros((self.n,))]
            self.t = np.r_[self.t, np.zeros((self.n,))]
            self.Tl = np.r_[self.Tl, np.zeros((self.n,))]
            self.Tv = np.r_[self.Tv, np.zeros((self.n,))]


