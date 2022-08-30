import numpy as np


EPS = 10**-6


class PhysicalProperties:
    def __init__(
        self,
        Delta=1.0,
        lda1=1.0,
        lda2=0.0,
        rho_cp1=1.0,
        rho_cp2=1.0,
        v=1.0,
        diff=1.0,
        a_i=358.0,
        alpha=0.06,
        dS=1.0,
    ):
        self._Delta = Delta
        self._lda1 = lda1
        self._lda2 = lda2
        self._rho_cp1 = rho_cp1
        self._rho_cp2 = rho_cp2
        self._v = v
        self._diff = diff
        self._a_i = a_i
        self._alpha = alpha
        self._dS = dS
        if self._v == 0.0:
            self._cas = "diffusion"
        elif self._diff == 0.0:
            self._cas = "convection"
        else:
            self._cas = "mixte"

    @property
    def Delta(self):
        return self._Delta

    @property
    def cas(self):
        return self._cas

    @property
    def lda1(self):
        return self._lda1

    @property
    def lda2(self):
        return self._lda2

    @property
    def rho_cp1(self):
        return self._rho_cp1

    @property
    def rho_cp2(self):
        return self._rho_cp2

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, val):
        self._v = val

    @property
    def diff(self):
        return self._diff

    @property
    def alpha(self):
        return self._alpha

    @property
    def a_i(self):
        return self._a_i

    @property
    def dS(self):
        return self._dS

    def isequal(self, prop):
        dic = {key: self.__dict__[key] == prop.__dict__[key] for key in self.__dict__}
        equal = np.all(list(dic.values()))
        if not equal:
            print("Attention, les propriétés ne sont pas égales :")
            print(dic)
        return equal


class NumericalProperties:
    def __init__(
        self,
        dx=0.1,
        dt=1.0,
        cfl=1.0,
        fo=1.0,
        schema="weno",
        time_scheme="euler",
        phy_prop=None,
        Delta=None,
        interp_type=None,
        conv_interf=None,
        time_integral=None,
        formulation=None
    ):
        if phy_prop is None and Delta is None:
            raise Exception("Impossible sans phy_prop ou Delta")
        if phy_prop is not None:
            Delta = phy_prop.Delta
        self.interp_type = interp_type
        self.conv_interf = conv_interf
        self.time_integral = time_integral
        self.formulation = formulation
        self._cfl_lim = cfl
        self._fo_lim = fo
        self._schema = schema
        self._time_scheme = time_scheme
        self._dx_lim = dx
        nx = int(Delta / dx)
        dx = Delta / nx
        self._dx = dx
        self._x = np.linspace(dx / 2.0, Delta - dx / 2.0, nx)
        self._x_f = np.linspace(0.0, Delta, nx + 1)
        self._dt_min = dt

    @property
    def cfl_lim(self):
        return self._cfl_lim

    @property
    def fo_lim(self):
        return self._fo_lim

    @property
    def schema(self):
        return self._schema

    @property
    def time_scheme(self):
        return self._time_scheme

    @property
    def x(self):
        return self._x

    @property
    def x_f(self):
        return self._x_f

    @property
    def dx(self):
        return self._dx

    @property
    def dt_min(self):
        return self._dt_min

    @property
    def dx_lim(self):
        return self._dx_lim

    def isequal(self, prop):
        dic = {key: self.__dict__[key] == prop.__dict__[key] for key in self.__dict__}
        for key in dic.keys():
            if isinstance(dic[key], np.ndarray):
                dic[key] = np.all(dic[key])
        equal = np.all(list(dic.values()))
        if not equal:
            print("Attention, les propriétés numériques ne sont pas égales :")
            print(dic)
        return equal


class Bulles:
    def __init__(
        self, markers=None, phy_prop=None, n_bulle=None, Delta=1.0, alpha=0.06, a_i=None
    ):
        self.diam = 0.0
        if phy_prop is not None:
            self.Delta = phy_prop.Delta
            self.alpha = phy_prop.alpha
            self.a_i = phy_prop.a_i
        else:
            self.Delta = Delta
            self.alpha = alpha
            self.a_i = a_i

        if markers is None:
            self.markers = []
            if n_bulle is None:
                if self.a_i is None:
                    raise Exception(
                        "On ne peut pas déterminer auto la géométrie des bulles sans le rapport surfacique"
                    )
                else:
                    # On détermine le nombre de bulle pour avoir une aire interfaciale donnée.
                    # On considère ici une géométrie 1D comme l'équivalent d'une situation 3D
                    n_bulle = int(self.a_i / 2.0 * self.Delta) + 1
            # Avec le taux de vide, on en déduit le diamètre d'une bulle. On va considérer que le taux de vide
            # s'exprime en 1D, cad : phy_prop.alpha = n*d*dS/(Dx*dS)
            self.diam = self.alpha * self.Delta / n_bulle
            centers = np.linspace(self.diam, self.Delta + self.diam, n_bulle + 1)[:-1]
            for center in centers:
                self.markers.append(
                    (center - self.diam / 2.0, center + self.diam / 2.0)
                )
            self.markers = np.array(self.markers)
            self.shift(self.Delta * 1.0 / 4)
        else:
            self.markers = np.array(markers).copy()  # type: np.ndarray
            n_bulle = self.markers.shape[0]
            for marker_pair in self.markers:
                mark1 = marker_pair[1]
                mark0 = marker_pair[0]
                while mark1 < mark0:
                    mark1 += self.Delta
                self.diam = mark1 - mark0
        self.n_bulles = n_bulle

        depasse = (self.markers > self.Delta) | (self.markers < 0.0)
        if np.any(depasse):
            print("Delta : ", self.Delta)
            print("markers : ", self.markers)
            print("depasse : ", depasse)
            raise Exception("Les marqueurs dépassent du domaine")

        self.init_markers = self.markers.copy()

    def __call__(self, *args, **kwargs):
        return self.markers

    def copy(self):
        cls = self.__class__
        copie = cls(markers=self.markers.copy(), Delta=self.Delta)
        copie.diam = self.diam
        return copie

    def indicatrice_liquide(self, x):
        """
        Calcule l'indicatrice qui correspond au liquide avec les marqueurs selon la grille x

        Args:
            x: les positions des centres des mailles

        Returns:
            l'indicatrice
        """
        i = np.ones_like(x)
        dx = x[1] - x[0]
        for markers in self.markers:
            if markers[0] < markers[1]:
                i[(x > markers[0]) & (x < markers[1])] = 0.0
            else:
                i[(x > markers[0]) | (x < markers[1])] = 0.0
            diph0 = np.abs(x - markers[0]) < dx / 2.0
            i[diph0] = (markers[0] - x[diph0]) / dx + 0.5
            diph1 = np.abs(x - markers[1]) < dx / 2.0
            i[diph1] = -(markers[1] - x[diph1]) / dx + 0.5
        return i

    def shift(self, dx):
        """
        On déplace les marqueurs vers la droite

        Args:
            dx: la distance du déplacement

        """
        self.markers += dx
        depasse = self.markers > self.Delta
        self.markers[depasse] -= self.Delta
