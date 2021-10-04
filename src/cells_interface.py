import numpy as np
from src.main import integrale_vol_div
from numba.experimental import jitclass
from numba import float64    # import the types
# from copy import copy
# from functools import wraps


# def wjitclass(ags):
#     def wrapper(cls):
#         print(cls.__doc__)
#
#         class Shadow:
#             pass
#
#         Shadow.__name__ = 'Test'
#         Shadow.__doc__ = cls.__doc__
#
#         jitclass(ags)(cls)
#         return Shadow
#     return wrapper


@jitclass([('Td', float64[:]), ('Tg', float64[:]), ('_rhocp_f', float64[:]), ('lda_f', float64[:]), ('_Ti', float64),
           ('_lda_gradTi', float64), ('ldag', float64), ('ldad', float64), ('rhocpg', float64), ('rhocpd', float64),
           ('ag', float64), ('ad', float64), ('dx', float64), ('vdt', float64), ('Tgc', float64), ('Tdc', float64),
           ('_dTdxg', float64), ('_dTdxd', float64), ('_d2Tdx2g', float64), ('_d2Tdx2d', float64),
           ('_d3Tdx3g', float64), ('_d3Tdx3d', float64), ('_T_f', float64[:]), ('_gradT_f', float64[:])])
class CellsInterface:
    schema_conv: str
    schema_diff: str
    interp_type: str

    """
    Cellule type ::

            Tg, gradTg                          Tghost
           +---------+----------+---------+---------+
           |         |          |         |   |     |
           |    +   -|>   +    -|>   +   -|>  |+    |
           |    0    |    1     |    2    |   |3    |
           +---------+----------+---------+---------+---------+---------+---------+
                                          |   |     |         |         |         |
                             Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                          |   |0    |    1    |    2    |    3    |
                                          +---------+---------+---------+---------+
                                              Ti,                              |--|
                                              lda_gradTi                         vdt

    Dans cette représentation de donnée de base, on peut stocker les valeurs de la température au centre, aux faces,
    la valeur des gradients aux faces, la valeur des lda et rhocp aux faces. On stocke aussi les valeurs à
    l'interface.

    Args:
        ldag:
        ldad:
        ag:
        dx:
        T:
        rhocpg:
        rhocpd:
    """

    def __init__(self, ldag=1., ldad=1., ag=1., dx=1., T=np.zeros((7,)), rhocpg=1., rhocpd=1., vdt=0.,
                 schema_conv='upwind', interp_type='Ti', schema_diff='centre'):
        self.ldag = ldag
        self.ldad = ldad
        self.rhocpg = rhocpg
        self.rhocpd = rhocpd
        self.ag = ag
        self.ad = 1. - ag
        self.dx = dx
        if len(T) < 7:
            raise(Exception('T n est pas de taille 7'))
        self.Tg = T[:4].copy()
        self.Tg[-1] = np.nan
        self.Td = T[3:].copy()
        self.Td[0] = np.nan
        self._rhocp_f = np.array([rhocpg, rhocpg, rhocpg, np.nan, rhocpd, rhocpd])
        self.lda_f = np.array([ldag, ldag, ldag, ldad, ldad, ldad])
        self._Ti = -1.
        self._lda_gradTi = 0.
        self._dTdxg = 0.
        self._dTdxd = 0.
        self._d2Tdx2g = 0.
        self._d2Tdx2d = 0.
        self._d3Tdx3g = 0.
        self._d3Tdx3d = 0.
        self.schema_conv = schema_conv
        self.schema_diff = schema_diff
        self.vdt = vdt
        self.interp_type = interp_type
        self.Tgc = -1.
        self.Tdc = -1.
        self._T_f = np.empty((6,), dtype=np.float_)
        self._gradT_f = np.empty((6,), dtype=np.float_)
        # On fait tout de suite le calcul qui nous intéresse, il est nécessaire pour la suite
        if self.interp_type == 'Ti' or self.interp_type == 'Ti_vol':
            self.compute_from_Ti()
        elif self.interp_type == 'Ti2':
            self.compute_from_Ti2()
        elif self.interp_type == 'Ti3':
            self.compute_from_Ti3()
        elif self.interp_type == 'Ti2_vol':
            self.compute_from_Ti2_vol()
        elif self.interp_type == 'Ti3_vol':
            self.compute_from_Ti3_vol()
        elif self.interp_type == 'Ti3_1_vol':
            self.compute_from_Ti3_1_vol()
        elif self.interp_type == 'gradTi':
            self.compute_from_ldagradTi()
        elif self.interp_type == 'gradTi2':
            self.compute_from_ldagradTi_ordre2()

        if self.interp_type.endswith('_vol'):
            self.compute_T_f_gradT_f_quick_vol()
        else:
            if self.schema_conv == 'weno':
                self.compute_T_f_gradT_f_quick()
            if self.schema_conv == 'quick':
                self.compute_T_f_gradT_f_quick()

    @staticmethod
    def pid_interp(T: float64[:], d: float64[:]) -> float:
        TH = 10**-15
        inf_lim = d < TH
        Tm = np.sum(T / d) / np.sum(1. / d)
        if np.any(inf_lim):
            Tm = T[inf_lim][0]
        return Tm

    @property
    def T_f(self):
        if self.schema_conv == 'upwind':
            return np.concatenate((self.Tg[:-1], self.Td[:-1]))
        if self.schema_conv == 'weno':
            return self._T_f
        if self.schema_conv == 'quick':
            return self._T_f
        else:
            # schema centre
            return np.concatenate(((self.Tg[1:] + self.Tg[:-1])/2., (self.Td[1:] + self.Td[:-1])/2.))

    @property
    def gradTg(self) -> np.ndarray((3,), dtype=float):
        return (self.Tg[1:] - self.Tg[:-1])/self.dx

    @property
    def gradTd(self) -> np.ndarray((3,), dtype=float):
        return (self.Td[1:] - self.Td[:-1])/self.dx

    @property
    def grad_lda_gradT_n_g(self) -> float:
        # remarque : cette valeur n'est pas calculée exactement à l'interface
        # mais plutôt entre l'interface et la face 32, mais je pense pas que ce soit très grave
        # et j'ai pas le courage de faire autre chose
        d = self.dx * (1. + self.ag)
        lda_gradT_32 = self.ldag * (self.Tg[-2] - self.Tg[-3]) / self.dx
        return (self.lda_gradTi - lda_gradT_32)/d

    @property
    def grad_lda_gradT_n_d(self) -> float:
        # remarque idem que au dessus
        d = self.dx * (1. + self.ad)
        lda_gradT_32 = self.ldad * (self.Td[2] - self.Td[1]) / self.dx
        return (lda_gradT_32 - self.lda_gradTi)/d

    @property
    def gradT(self) -> np.ndarray((6,), dtype=float):
        if self.schema_diff == 'DL' or self.schema_conv == 'weno' or self.schema_conv == 'quick':
            return self._gradT_f
        else:
            return np.concatenate((self.gradTg, self.gradTd))

    @property
    def rhocp_f(self) -> np.ndarray((6,), dtype=float):
        if self.vdt > 0.:
            coeff_d = min(self.vdt, self.ad*self.dx)/self.vdt
            self._rhocp_f[3] = coeff_d * self.rhocpd + (1. - coeff_d) * self.rhocpg
            return self._rhocp_f
        else:
            self._rhocp_f[3] = self.rhocpd
            return self._rhocp_f

    @property
    def rhocpT_f(self) -> np.ndarray((6,), dtype=float):
        if self.vdt > 0.:
            # TODO: implémenter une méthode qui renvoie rho * cp * T intégré sur le volume qui va passer d'une cellule à
            #  l'autre. Cette précision n'est peut-être pas nécessaire
            rhocpTf = self.rhocp_f * self.T_f
            return rhocpTf
        else:
            return self.rhocp_f * self.T_f

    @property
    def Ti(self) -> float:
        return self._Ti

    @property
    def lda_gradTi(self) -> float:
        return self._lda_gradTi

    def compute_T_f_gradT_f_weno(self):
        # TODO: il faut changer de manière à toujours faire une interpolation amont de la température et centrée des
        #  gradients
        # Tim52, dTdxim52, Tim32, dTdxim32 = self._interp_lagrange3(self.Tg[0], self.Tg[1], self.Tg[2])
        # Tip32, dTdxip32, Tip52, dTdxip52 = self._interp_lagrange3(self.Td[1], self.Td[2], self.Td[3])
        # self._T_f[0] = Tim52
        # self._T_f[1] = Tim32
        # self._T_f[4] = Tip32
        # self._T_f[5] = Tip52
        # self._T_f[2] = self._T_dlg(0.)
        # self._T_f[3] = self._T_dld(self.dx)
        # self._gradT_f[0] = dTdxim52
        # self._gradT_f[1] = dTdxim32
        # self._gradT_f[4] = dTdxip32
        # self._gradT_f[5] = dTdxip52
        # self._gradT_f[2] = self._gradT_dlg(0.)
        # self._gradT_f[3] = self._gradT_dld(self.dx)
        raise NotImplemented

    def compute_T_f_gradT_f_quick(self):
        """
        Cellule type ::

                Tg, gradTg                          Tghost
                         0          1         2         3
               +---------+----------+---------+---------+
               |         |          |         |   |     |
               |    +   -|>   +    -|>   +   -|>  |+    |
               |    0    |    1     |    2    |   |3    |         4         5
               +---------+----------+---------+---------+---------+---------+---------+
                                              |   |     |         |         |         |
                                 Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                              |   |0    |    1    |    2    |    3    |
                                              +---------+---------+---------+---------+
                                                  Ti,                              |--|
                                                  lda_gradTi                         vdt

        Returns:

        """
        Tim32, dTdxim32, _ = self._interp_lagrange_amont(self.Tg[0], self.Tg[1], self.Tg[2],
                                                         -2 * self.dx, -1. * self.dx, 0. * self.dx,
                                                         -0.5 * self.dx)
        Tim12, dTdxim12, _ = self._interp_lagrange_amont(self.Tg[1], self.Tg[2], self._Ti,
                                                         -2. * self.dx, -1. * self.dx, (self.ag - 0.5) * self.dx,
                                                         -0.5 * self.dx)
        Tip12, dTdxip12, _ = self._interp_lagrange_amont_grad(self._Ti, self.Td[1], self._dTdxd,
                                                              (0.5 - self.ad) * self.dx, 1. * self.dx,
                                                              0.5 * self.dx)
        Tip32, dTdxip32, _ = self._interp_lagrange_amont(self._Ti, self.Td[1], self.Td[2],
                                                         (0.5 - self.ad) * self.dx, 1. * self.dx, 2. * self.dx,
                                                         1.5 * self.dx)
        Tip52, dTdxip52, _ = self._interp_lagrange_amont(self.Td[1], self.Td[2], self.Td[3],
                                                         1. * self.dx, 2. * self.dx, 3. * self.dx,
                                                         2.5 * self.dx)
        self._T_f[0] = np.nan  # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[1] = Tim32
        self._T_f[2] = Tim12  # self._T_dlg(0.)
        self._T_f[3] = Tip12  # self._T_dld(self.dx)
        self._T_f[4] = Tip32
        self._T_f[5] = Tip52
        self._gradT_f[0] = np.nan
        self._gradT_f[1] = dTdxim32
        self._gradT_f[2] = dTdxim12
        self._gradT_f[3] = dTdxip12
        self._gradT_f[4] = dTdxip32
        self._gradT_f[5] = dTdxip52

    def compute_T_f_gradT_f_upwind(self):
        """
        Cellule type ::

                Tg, gradTg                          Tghost
                         0          1         2         3
               +---------+----------+---------+---------+
               |         |          |         |   |     |
               |    +   -|>   +    -|>   +   -|>  |+    |
               |    0    |    1     |    2    |   |3    |         4         5
               +---------+----------+---------+---------+---------+---------+---------+
                                              |   |     |         |         |         |
                                 Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                              |   |0    |    1    |    2    |    3    |
                                              +---------+---------+---------+---------+
                                                  Ti,                              |--|
                                                  lda_gradTi                         vdt

        Returns:

        """
        Tim32, dTdxim32, _ = self._interp_lagrange_upwind(self.Tg[1], self.Tg[2],
                                                             -1. * self.dx, 0. * self.dx,
                                                             -0.5 * self.dx)
        Tim12, dTdxim12, _ = self._interp_lagrange_upwind(self.Tg[2], self._Ti,
                                                             -1. * self.dx, (self.ag - 0.5) * self.dx,
                                                             -0.5 * self.dx)
        Tip12, dTdxip12, _ = self._interp_lagrange_upwind_grad(self._Ti, self.Td[1], self._dTdxd,
                                                                  (0.5 - self.ad) * self.dx, 1. * self.dx,
                                                                  0.5 * self.dx)
        Tip32, dTdxip32, _ = self._interp_lagrange_upwind(self.Td[1], self.Td[2],
                                                             1. * self.dx, 2. * self.dx,
                                                             1.5 * self.dx)
        Tip52, dTdxip52, _ = self._interp_lagrange_upwind(self.Td[2], self.Td[3],
                                                             2. * self.dx, 3. * self.dx,
                                                             2.5 * self.dx)
        self._T_f[0] = np.nan  # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[1] = Tim32
        self._T_f[2] = Tim12  # self._T_dlg(0.)
        self._T_f[3] = Tip12  # self._T_dld(self.dx)
        self._T_f[4] = Tip32
        self._T_f[5] = Tip52
        self._gradT_f[0] = np.nan
        self._gradT_f[1] = dTdxim32
        self._gradT_f[2] = dTdxim12
        self._gradT_f[3] = dTdxip12
        self._gradT_f[4] = dTdxip32
        self._gradT_f[5] = dTdxip52

    def compute_T_f_gradT_f_quick_vol(self):
        """
        Cellule type ::

                Tg, gradTg                          Tghost
                         0          1         2         3
               +---------+----------+---------+---------+
               |         |          |         |   |     |
               |    +   -|>   +    -|>   +   -|>  |+    |
               |    0    |    1     |    2    |   |3    |         4         5
               +---------+----------+---------+---------+---------+---------+---------+
                                              |   |     |         |         |         |
                                 Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                              |   |0    |    1    |    2    |    3    |
                                              +---------+---------+---------+---------+
                                                  Ti,                              |--|
                                                  lda_gradTi                         vdt

        Returns:

        """
        Tim32, dTdxim32, _ = self._interp_lagrange_amont_vol(self.Tg[0], self.Tg[1], self.Tg[2],
                                                             -2 * self.dx, -1. * self.dx, 0. * self.dx,
                                                             -0.5 * self.dx)
        Tim12, dTdxim12, _ = self._interp_lagrange_amont_vol(self.Tg[1], self.Tg[2], self._Ti,
                                                             -2. * self.dx, -1. * self.dx, (self.ag - 0.5) * self.dx,
                                                             -0.5 * self.dx)
        Tip12, dTdxip12, _ = self._interp_lagrange_amont_grad_vol(self._Ti, self.Td[1], self._dTdxd,
                                                                  (0.5 - self.ad) * self.dx, 1. * self.dx,
                                                                  0.5 * self.dx)
        Tip32, dTdxip32, _ = self._interp_lagrange_amont_vol(self._Ti, self.Td[1], self.Td[2],
                                                             (0.5 - self.ad) * self.dx, 1. * self.dx, 2. * self.dx,
                                                             1.5 * self.dx)
        Tip52, dTdxip52, _ = self._interp_lagrange_amont_vol(self.Td[1], self.Td[2], self.Td[3],
                                                             1. * self.dx, 2. * self.dx, 3. * self.dx,
                                                             2.5 * self.dx)
        self._T_f[0] = np.nan  # on ne veut pas se servir de cette valeur, on veut utiliser la version weno / quick
        self._T_f[1] = Tim32
        self._T_f[2] = Tim12  # self._T_dlg(0.)
        self._T_f[3] = Tip12  # self._T_dld(self.dx)
        self._T_f[4] = Tip32
        self._T_f[5] = Tip52
        self._gradT_f[0] = np.nan
        self._gradT_f[1] = dTdxim32
        self._gradT_f[2] = dTdxim12
        self._gradT_f[3] = dTdxip12
        self._gradT_f[4] = dTdxip32
        self._gradT_f[5] = dTdxip52

    def compute_from_ldagradTi(self):
        """
        On commence par récupérer lda_grad_Ti, gradTg, gradTd par continuité à partir de gradTim32 et gradTip32.
        On en déduit Ti par continuité à gauche et à droite.

        Returns:
            Calcule les gradients g, I, d, et Ti. gradTig et gradTid sont les gradients centrés entre TI et Tim1 et TI
            et Tip1

        Warnings:
            Cette méthode est dépréciée, elle a recours à des extrapolation d'ordre 1
        """
        lda_gradTg, lda_gradTig, self._lda_gradTi, lda_gradTid, lda_gradTd = \
            self._get_lda_grad_T_i_from_ldagradT_continuity(self.Tg[1], self.Tg[2], self.Td[1], self.Td[2],
                                                            (1. + self.ag) * self.dx, (1. + self.ad) * self.dx)

        self._Ti = self._get_Ti_from_lda_grad_Ti(self.Tg[-2], self.Td[1],
                                                 (0.5+self.ag) * self.dx, (0.5+self.ad) * self.dx,
                                                 lda_gradTig, lda_gradTid)
        self.Tg[-1] = self.Tg[-2] + lda_gradTg/self.ldag * self.dx
        self.Td[0] = self.Td[1] - lda_gradTd/self.ldad * self.dx

    def compute_from_ldagradTi_ordre2(self):
        """
        On commence par récupérer lda_grad_Ti par continuité à partir de gradTim52 gradTim32 gradTip32
        et gradTip52.
        On en déduit Ti par continuité à gauche et à droite.

        Returns:
            Calcule les gradients g, I, d, et Ti

        Warnings:
            Cette méthode est dépréciée, elle a recours à des extrapolation d'ordre 1
        """
        lda_gradTg, lda_gradTig, self._lda_gradTi, lda_gradTid, lda_gradTd = \
            self._get_lda_grad_T_i_from_ldagradT_interp(self.Tg[0], self.Tg[1], self.Tg[2],
                                                        self.Td[1], self.Td[2], self.Td[3],
                                                        self.ag, self.ad)

        self._Ti = self._get_Ti_from_lda_grad_Ti(self.Tg[-2], self.Td[1],
                                                 (0.5+self.ag) * self.dx, (0.5+self.ad) * self.dx,
                                                 lda_gradTig, lda_gradTid)
        self.Tg[-1] = self.Tg[-2] + lda_gradTig/self.ldag * self.dx
        self.Td[0] = self.Td[1] - lda_gradTid/self.ldad * self.dx

    def compute_from_Ti(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1.
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        C'est la méthode qui donne les résultats les plus stables. Probablement parce qu'elle donne un poids plus
        important aux valeurs des mailles monophasiques

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(self.Tg[-2], self.Td[1],
                                                                    (1./2 + self.ag) * self.dx,
                                                                    (1./2 + self.ad) * self.dx)
        # self._Ti, self._lda_gradTi, _, _, _, _ = \
        #     self._get_T_i_and_lda_grad_T_i3(self.Tg[-2], self.Td[1], self.Tg[-3], self.Td[2], self.Tg[-4], self.Td[3],
        #                                     (1./2 + self.ag) * self.dx, (1./2 + self.ad) * self.dx)
        self._dTdxg = self._lda_gradTi/self.ldag
        self._dTdxd = self._lda_gradTi/self.ldad
        # grad_Tg = self.pid_interp(np.array([self.gradTg[1], self._lda_gradTi/self.ldag]),
        #                           np.array([1., self.ag])*self.dx)
        # grad_Td = self.pid_interp(np.array([self._lda_gradTi/self.ldad, self.gradTd[1]]),
        #                           np.array([self.ad, 1.])*self.dx)
        # self.Tg[-1] = self.Tg[-2] + grad_Tg * self.dx
        # if np.isnan(self.Tg[-1]):
        #     print('Tg2 : ', self.Tg[-2])
        #     print('gradTg : ', grad_Tg)
        #     print('array : ', np.array([self.gradTg[1], self._lda_gradTi/self.ldag]))
        #     print('d : ', np.array([1., self.ag])*self.dx)
        # self.Td[0] = self.Td[1] - grad_Td * self.dx

    def compute_from_Ti2(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1, Tim2 et Tip2
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        self._Ti, self._lda_gradTi, self._d2Tdx2g, self._d2Tdx2d = \
            self._get_T_i_and_lda_grad_T_i2(self.Tg[-2], self.Td[1], self.Tg[-3], self.Td[2],
                                            (1./2 + self.ag) * self.dx, (1./2 + self.ad) * self.dx)
        self._dTdxg = self._lda_gradTi/self.ldag
        self._dTdxd = self._lda_gradTi/self.ldad
        # self.Tg[-1] = self._T_dlg(0.5 * self.dx)
        # self.Td[0] = self._T_dld(0.5 * self.dx)

    def compute_from_Ti2_vol(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1, Tim2 et Tip2
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        self._Ti, self._lda_gradTi, self._d2Tdx2g, self._d2Tdx2d = \
            self._get_T_i_and_lda_grad_T_i2_vol(self.Tg[-2], self.Td[1], self.Tg[-3], self.Td[2],
                                                (1./2 + self.ag) * self.dx, (1./2 + self.ad) * self.dx)
        self._dTdxg = self._lda_gradTi/self.ldag
        self._dTdxd = self._lda_gradTi/self.ldad

    def compute_from_Ti3(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1, Tim2 et Tip2
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        self._Ti, self._lda_gradTi, self._d2Tdx2g, self._d2Tdx2d, self._d3Tdx3g, self._d3Tdx3d = \
            self._get_T_i_and_lda_grad_T_i3(self.Tg[-2], self.Td[1], self.Tg[-3], self.Td[2], self.Tg[-4], self.Td[3],
                                            (1./2 + self.ag) * self.dx, (1./2 + self.ad) * self.dx)
        self._dTdxg = self._lda_gradTi/self.ldag
        self._dTdxd = self._lda_gradTi/self.ldad
        # self.Tg[-1] = self._T_dlg(0.5 * self.dx)
        # self.Td[0] = self._T_dld(0.5 * self.dx)

    def compute_from_Ti3_vol(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1, Tim2 et Tip2
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        self._Ti, self._lda_gradTi, self._d2Tdx2g, self._d2Tdx2d, self._d3Tdx3g, self._d3Tdx3d = \
            self._get_T_i_and_lda_grad_T_i3_vol(self.Tg[-2], self.Td[1], self.Tg[-3], self.Td[2], self.Tg[-4], self.Td[3],
                                                (1./2 + self.ag) * self.dx, (1./2 + self.ad) * self.dx)
        self._dTdxg = self._lda_gradTi/self.ldag
        self._dTdxd = self._lda_gradTi/self.ldad

    def compute_from_Ti3_1_vol(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1, Tim2 et Tip2
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        self._Ti, self._lda_gradTi, self._d2Tdx2g, self._d2Tdx2d = \
            self._get_T_i_and_lda_grad_T_i3_1_vol(self.Tg[-4], self.Tg[-3], self.Tg[-2], self.Td[1],
                                                  (1./2 + self.ag) * self.dx, (1./2 + self.ad) * self.dx)
        self._dTdxg = self._lda_gradTi/self.ldag
        self._dTdxd = self._lda_gradTi/self.ldad

    def _compute_Tgc_Tdc(self, h: float, T_mean: float):
        """
        Résout le système d'équation entre la température moyenne et l'énergie de la cellule pour trouver les valeurs de
        Tgc et Tdc.
        Le système peut toujours être résolu car rhocpg != rhocpd

        Returns:
            Rien mais mets à jour Tgc et Tdc
        """
        system = np.array([[self.ag*self.rhocpg, self.ad*self.rhocpd],
                           [self.ag, self.ad]])
        self.Tgc, self.Tdc = np.dot(np.linalg.inv(system), np.array([h, T_mean]))

    def compute_from_h_T(self, h: float, T_mean: float):
        """
    Cellule type ::

                                         Ti,
                                         lda_gradTi
                                         Ti0g
                                         Ti0d
                                    Tgf Tgc Tdc Tdf
                +----------+---------+---------+---------+---------+
                |          |         | gc|  dc |         |         |
                |    +    -|>   +   -|>* | +* -|>   +   -|>   +    |
                |          |         |   |     |         |         |
                +----------+---------+---------+---------+---------+
                        gradTi-3/2  gradTg    gradTd    gradTi+3/2

    Dans ce modèle on connait initialement les températures moyenne aux centes de toutes les cellules.
    On reconstruit les valeurs de Tgc et Tdc avec le système sur la valeur moyenne de température dans la maille et
    la valeur moyenne de l'énergie.
    Ensuite évidemment on interpole là ou on en aura besoin.
    Il faudra faire 2 équations d'évolution dans la cellule i, une sur h et une sur T.

    Selon la method calcule les flux et les températures aux interfaces.
    Si la méthode est classique, on calcule tout en utilisant Tim1, Tgc et T_I (et de l'autre côté T_I, Tdc et Tip1)

    Args:
        h:
        T_mean:

    Returns:
        Rien mais met à jour self.grad_T et self.T_f
    """
        self._compute_Tgc_Tdc(h, T_mean)
        # On commence par calculer T_I et lda_grad_Ti en fonction de Tgc et Tdc :
        self._Ti, self._lda_gradTi = self._get_T_i_and_lda_grad_T_i(self.Tgc, self.Tdc, self.ag / 2. * self.dx,
                                                                    self.ad / 2. * self.dx)

        # Calcul des gradient aux faces

        # À gauche :
        aim1_gc = 0.5 + self.ag/2.
        gradTim1_gc = (self.Tgc - self.Tg[-2])/(aim1_gc*self.dx)
        gradTg_v = np.array([self.gradTg[-2], gradTim1_gc, self.lda_gradTi/self.ldag])
        dist = np.array([1., np.abs(0.5 - aim1_gc/2.), self.ag])*self.dx
        gradTg = self.pid_interp(gradTg_v, dist)
        self.Tg[-1] = self.Tg[-2] + self.dx * gradTg

        # À droite :
        aip1_dc = 0.5 + self.ad/2.
        gradTip1_dc = (self.Td[1] - self.Tdc)/(aip1_dc*self.dx)
        gradTd_v = np.array([self.lda_gradTi/self.ldad, gradTip1_dc, self.gradTd[1]])
        dist = np.array([self.ad, np.abs(0.5 - aip1_dc/2.), 1.])*self.dx
        gradTd = self.pid_interp(gradTd_v, dist)
        self.Td[0] = self.Td[1] - self.dx * gradTd

    def _get_T_i_and_lda_grad_T_i(self, Tg: float, Td: float, dg: float, dd: float) -> (float, float):
        """
        On utilise la continuité de lad_grad_T pour interpoler linéairement à partir des valeurs en im32 et ip32
        On retourne les gradients suivants ::

                                 dg              dd
                            |-----------|-------------------|
                    +---------------+---------------+---------------+
                    |               |   |           |               |
                   -|>      +      -|>  |   +      -|>      +      -|>
                    |               |   |           |               |
                    +---------------+---------------+---------------+

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        T_i = (self.ldag/dg*Tg + self.ldad/dd*Td) / (self.ldag/dg + self.ldad/dd)
        lda_grad_T_ig = self.ldag * (T_i - Tg)/dg
        lda_grad_T_id = self.ldad * (Td - T_i)/dd
        return T_i, (lda_grad_T_ig + lda_grad_T_id)/2.

    def _get_T_i_and_lda_grad_T_i2(self, Tg: float, Td: float, Tgg: float, Tdd: float, dg: float,
                                   dd: float) -> (float, float, float, float):
        """
        On utilise la continuité de lad_grad_T pour interpoler linéairement à partir des valeurs en im32 et ip32
        On retourne les gradients suivants ::

                                          dgg    dg              dd      ddd
                            |---------------|-----------|-------------------|---------------|
                    +---------------+---------------+---------------+---------------+---------------+
                    |               |               |   |           |               |               |
                   -|>      +      -|>      +      -|>  |   +      -|>      +      -|>      +      -|>
                    |               |               |   |           |               |               |
                    +---------------+---------------+---------------+---------------+---------------+

        Returns:
            Calcule Ti, lda_grad_T ainsi que les dérivées d'ordre 2 g et d à l'interface
        """
        dgg = dg + self.dx
        ddd = dd + self.dx
        mat = np.array([[1., dg,                        dg**2/2.,   0.],
                        [1., dgg,                       dgg**2/2.,  0.],
                        [1., dd*self.ldag/self.ldad,    0.,         dd**2/2.],
                        [1., ddd*self.ldag/self.ldad,   0.,         ddd**2/2.]], dtype=np.float_)
        temp = np.array([Tg, Tgg, Td, Tdd], dtype=np.float_).T
        inv_mat = np.linalg.inv(mat)
        T_i, dTdxg, d2Tdx2g, d2Tdx2d = np.dot(inv_mat, temp)
        lda_grad_T = self.ldag * dTdxg
        return T_i, lda_grad_T, d2Tdx2g, d2Tdx2d

    def _get_T_i_and_lda_grad_T_i3(self, Tg: float, Td: float, Tgg: float, Tdd: float, Tggg: float, Tddd: float,
                                   dg: float, dd: float) -> (float, float, float, float, float, float):
        """
        On utilise la continuité de lad_grad_T et on écrit un DL à l'ordre 3 à droite et à gauche.
        On retourne les gradients suivants ::

                                           dggg   dgg    dg              dd      ddd        dddd
                    |---------------|---------------|-----------|-------------------|---------------|--------------|
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+
            |               |               |               |   |           |               |               |              |
            |>      +      -|>      +      -|>      +      -|>  |   +      -|>      +      -|>      +      -|>     +      -|>
            |               |               |               |   |           |               |               |              |
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+

        Returns:
            Calcule Ti, lda_grad_T ainsi que les dérivées d'ordre 2 g et d à l'interface
        """
        dgg = dg + self.dx
        dggg = dg + 2.*self.dx
        ddd = dd + self.dx
        dddd = dd + 2.*self.dx
        mat = np.array([[1., -dg,                      dg**2/2.,   0.,         -dg**3/6,   0.],
                        [1., -dgg,                     dgg**2/2.,  0.,         -dgg**3/6,  0.],
                        [1., -dggg,                    dggg**2/2., 0.,         -dggg**3/6, 0.],
                        [1., dd*self.ldag/self.ldad,   0.,         dd**2/2.,   0.,         dd**3/6.],
                        [1., ddd*self.ldag/self.ldad,  0.,         ddd**2/2.,  0.,         ddd**3/6.],
                        [1., dddd*self.ldag/self.ldad, 0.,         dddd**2/2., 0.,         dddd**3/6.]], dtype=np.float_)
        temp = np.array([Tg, Tgg, Tggg, Td, Tdd, Tddd], dtype=np.float_).T
        inv_mat = np.linalg.inv(mat)
        T_i, dTdxg, d2Tdx2g, d2Tdx2d, d3Tdx3g, d3Tdx3d = np.dot(inv_mat, temp)
        lda_grad_T = self.ldag * dTdxg
        return T_i, lda_grad_T, d2Tdx2g, d2Tdx2d, d3Tdx3g, d3Tdx3d

    def _get_T_i_and_lda_grad_T_i2_vol(self, Tg: float, Td: float, Tgg: float, Tdd: float, dg: float,
                                       dd: float) -> (float, float, float, float):
        """
        On utilise la continuité de lad_grad_T pour interpoler linéairement à partir des valeurs en im32 et ip32
        On retourne les gradients suivants ::

                                          dgg    dg              dd      ddd
                            |---------------|-----------|-------------------|---------------|
                    +---------------+---------------+---------------+---------------+---------------+
                    |               |               |   |           |               |               |
                   -|>      +      -|>      +      -|>  |   +      -|>      +      -|>      +      -|>
                    |               |               |   |           |               |               |
                    +---------------+---------------+---------------+---------------+---------------+

        Returns:
            Calcule Ti, lda_grad_T ainsi que les dérivées d'ordre 2 g et d à l'interface
        """
        dgg = dg + self.dx
        ddd = dd + self.dx
        mat = np.array([[1., -dg,                     dg**2 * 2./3.,  0.],
                        [1., -dgg,                    dgg**2 * 2./3., 0.],
                        [1., dd*self.ldag/self.ldad,  0.,             dd**2 * 2./3.],
                        [1., ddd*self.ldag/self.ldad, 0.,             ddd**2 * 2./3.]], dtype=np.float_)
        temp = np.array([Tg, Tgg, Td, Tdd], dtype=np.float_).T
        inv_mat = np.linalg.inv(mat)
        T_i, dTdxg, d2Tdx2g, d2Tdx2d = np.dot(inv_mat, temp)
        lda_grad_T = self.ldag * dTdxg
        return T_i, lda_grad_T, d2Tdx2g, d2Tdx2d

    def _get_T_i_and_lda_grad_T_i3_vol(self, Tg: float, Td: float, Tgg: float, Tdd: float, Tggg: float, Tddd: float,
                                       dg: float, dd: float) -> (float, float, float, float, float, float):
        """
        On utilise la continuité de lad_grad_T et on écrit un DL à l'ordre 3 à droite et à gauche.
        On retourne les gradients suivants ::

                                           dggg   dgg    dg              dd      ddd        dddd
                    |---------------|---------------|-----------|-------------------|---------------|--------------|
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+
            |               |               |               |   |           |               |               |              |
            |>      +      -|>      +      -|>      +      -|>  |   +      -|>      +      -|>      +      -|>     +      -|>
            |               |               |               |   |           |               |               |              |
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+

        Returns:
            Calcule Ti, lda_grad_T ainsi que les dérivées d'ordre 2 g et d à l'interface
        """
        dgg = dg + self.dx
        dggg = dg + 2.*self.dx
        ddd = dd + self.dx
        dddd = dd + 2.*self.dx
        mat = np.array([[1., -dg,                        dg**2 * 2./3.,   0.,             -dg**3 * 1./3.,   0.],
                        [1., -dgg,                       dgg**2 * 2./3.,  0.,             -dgg**3 * 1./3.,  0.],
                        [1., -dggg,                      dggg**2 * 2./3., 0.,             -dggg**3 * 1./3., 0.],
                        [1., dd*self.ldag/self.ldad,     0.,              dd**2 * 2./3.,   0.,              dd**3 * 1./3.],
                        [1., ddd*self.ldag/self.ldad,    0.,              ddd**2 * 2./3.,  0.,              ddd**3 * 1./3.],
                        [1., dddd * self.ldag/self.ldad, 0.,              dddd**2 * 2./3., 0.,              dddd**3 * 1./3.]], dtype=np.float_)
        temp = np.array([Tg, Tgg, Tggg, Td, Tdd, Tddd], dtype=np.float_).T
        inv_mat = np.linalg.inv(mat)
        T_i, dTdxg, d2Tdx2g, d2Tdx2d, d3Tdx3g, d3Tdx3d = np.dot(inv_mat, temp)
        lda_grad_T = self.ldag * dTdxg
        return T_i, lda_grad_T, d2Tdx2g, d2Tdx2d, d3Tdx3g, d3Tdx3d

    def _get_T_i_and_lda_grad_T_i3_1_vol(self, Tggg: float, Tgg: float, Tg: float, Td: float,
                                         dg: float, dd: float) -> (float, float, float, float):
        """
        On utilise la continuité de lad_grad_T et on écrit un DL à l'ordre 3 à droite et à gauche.
        On retourne les gradients suivants ::

                                           dggg   dgg    dg              dd      ddd        dddd
                    |---------------|---------------|-----------|-------------------|---------------|--------------|
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+
            |               |               |               |   |           |               |               |              |
            |>      +      -|>      +      -|>      +      -|>  |   +      -|>      +      -|>      +      -|>     +      -|>
            |               |               |               |   |           |               |               |              |
            +---------------+---------------+---------------+---------------+---------------+---------------+--------------+

        Returns:
            Calcule Ti, lda_grad_T ainsi que les dérivées d'ordre 2 g et d à l'interface
        """
        dgg = dg + self.dx
        dggg = dg + 2.*self.dx
        mat = np.array([[1., -dg,                        dg**2 * 2./3.,   0.],
                        [1., -dgg,                       dgg**2 * 2./3.,  0.],
                        [1., -dggg,                      dggg**2 * 2./3., 0.],
                        [1., dd*self.ldag/self.ldad,     0.,              dd**2 * 2./3.]], dtype=np.float_)
        temp = np.array([Tg, Tgg, Tggg, Td], dtype=np.float_).T
        inv_mat = np.linalg.inv(mat)
        T_i, dTdxg, d2Tdx2g, d2Tdx2d = np.dot(inv_mat, temp)
        lda_grad_T = self.ldag * dTdxg
        return T_i, lda_grad_T, d2Tdx2g, d2Tdx2d

    def _T_dlg(self, x: float) -> float:
        xi = self.ag*self.dx
        return self.Ti + (x - xi) * self._dTdxg + (x - xi)**2/2. * self._d2Tdx2g + (x - xi)**3/6. * self._d3Tdx3g

    def _T_dld(self, x: float) -> float:
        xi = self.ag*self.dx
        return self.Ti + (x - xi) * self._dTdxd + (x - xi)**2/2. * self._d2Tdx2d + (x - xi)**3/6. * self._d3Tdx3d

    def _gradT_dlg(self, x: float) -> float:
        xi = self.ag*self.dx
        return self._dTdxg + (x - xi) * self._d2Tdx2g + (x - xi)**2/2. * self._d3Tdx3g

    def _gradT_dld(self, x: float) -> float:
        xi = self.ag*self.dx
        return self._dTdxd + (x - xi) * self._d2Tdx2d + (x - xi)**2/2. * self._d3Tdx3d

    @staticmethod
    def _interp_upwind(T0: float, T1: float, x0: float, x1: float) \
            -> (float, float, float):
        """
        Dans cette méthode on veux que T0 et T1 soient amont de x_int
        C'est une interpolation d'ordre 0

        Args:
            T0:
            T1:
            T2:
            x0:
            x1:
            x2:
            x_int:

        Returns:

        """

        Tint = T0
        dTdx_int = (T1 - T0)/(x1 - x0)
        d2Tdx2_int = 0.
        return Tint, dTdx_int, d2Tdx2_int

    def _interp_lagrange_amont_vol(self, T0: float, T1: float, T2: float, x0: float, x1: float, x2: float, x_int: float) \
            -> (float, float, float):
        """
        Dans cette méthode on veux que T0 et T1 soient amont de x_int
        C'est une interpolation d'ordre 3

        Args:
            T0:
            T1:
            T2:
            x0:
            x1:
            x2:
            x_int:

        Returns:

        """
        d00 = x0 - x_int - 0.5 * self.dx
        d01 = x0 - x_int + 0.5 * self.dx
        d10 = x1 - x_int - 0.5 * self.dx
        d11 = x1 - x_int + 0.5 * self.dx
        d20 = x2 - x_int - 0.5 * self.dx
        d21 = x2 - x_int + 0.5 * self.dx
        # print('d : ', d0, d1, d2)

        mat = np.array([[1., (d01**2 - d00**2) / 2. / self.dx, (d01**3 - d00**3) / 6. / self.dx],
                        [1., (d11**2 - d10**2) / 2. / self.dx, (d11**3 - d10**3) / 6. / self.dx],
                        [1., (d21**2 - d20**2) / 2. / self.dx, (d21**3 - d20**3) / 6. / self.dx]], dtype=np.float_)
        inv_mat = np.linalg.inv(mat)
        # print(inv_mat)
        Tint, dTdx_int, d2Tdx2_int = np.dot(inv_mat, np.array([T0, T1, T2]))
        return Tint, dTdx_int, d2Tdx2_int

    def _interp_lagrange_amont_grad_vol(self, T0: float, T1: float, gradT0: float, x0: float, x1: float, x_int: float) \
            -> (float, float, float):

        """
        Schéma d'ordre 3 mais avec gradTi d'ordre 2, on peut dire qu'il est décentré car on utilise à la fois la valeur
        et le gradient amont.
        Args:
            T0:
            T1:
            gradT0:
            x0:
            x1:
            x_int:

        Returns:
            Les dérivées successives
        """
        d00 = x0 - x_int - 0.5 * self.dx
        d01 = x0 - x_int + 0.5 * self.dx
        d10 = x1 - x_int - 0.5 * self.dx
        d11 = x1 - x_int + 0.5 * self.dx

        mat = np.array([[1., (d01 ** 2 - d00 ** 2) / 2. / self.dx, (d01 ** 3 - d00 ** 3) / 6. / self.dx],
                        [1., (d11 ** 2 - d10 ** 2) / 2. / self.dx, (d11 ** 3 - d10 ** 3) / 6. / self.dx],
                        [0., 1., (d01 ** 2 - d00 ** 2) / 2. / self.dx]], dtype=np.float_)
        Tint, dTdx_int, d2Tdx2_int = np.dot(np.linalg.inv(mat), np.array([T0, T1, gradT0]))
        return Tint, dTdx_int, d2Tdx2_int

    @staticmethod
    def _interp_lagrange_amont(T0: float, T1: float, T2: float, x0: float, x1: float, x2: float, x_int: float) \
            -> (float, float, float):
        """
        Dans cette méthode on veux que T0 et T1 soient amont de x_int
        C'est une interpolation d'ordre 3

        Args:
            T0:
            T1:
            T2:
            x0:
            x1:
            x2:
            x_int:

        Returns:

        """
        d0 = x0 - x_int
        d1 = x1 - x_int
        d2 = x2 - x_int
        # print('d : ', d0, d1, d2)

        mat = np.array([[1., d0, d0**2 / 2.],
                        [1., d1, d1**2 / 2.],
                        [1., d2, d2**2 / 2.]], dtype=np.float_)
        Tint, dTdx_int, d2Tdx2_int = np.dot(np.linalg.inv(mat), np.array([T0, T1, T2]))
        return Tint, dTdx_int, d2Tdx2_int

    @staticmethod
    def _interp_lagrange_amont_grad(T0: float, T1: float, gradT0: float, x0: float, x1: float, x_int: float) \
            -> (float, float, float):

        """
        Schéma d'ordre 3 mais avec gradTi d'ordre 2, on peut dire qu'il est décentré car on utilise à la fois la valeur
        et le gradient amont.
        Args:
            T0:
            T1:
            gradT0:
            x0:
            x1:
            x_int:

        Returns:
            Les dérivées successives
        """
        d0 = x0 - x_int
        d1 = x1 - x_int

        mat = np.array([[1., d0, d0**2 / 2.],
                        [1., d1, d1**2 / 2.],
                        [0., 1., d0]], dtype=np.float_)
        Tint, dTdx_int, d2Tdx2_int = np.dot(np.linalg.inv(mat), np.array([T0, T1, gradT0]))
        return Tint, dTdx_int, d2Tdx2_int

    def _get_lda_grad_T_i_from_ldagradT_continuity(self, Tim2: float, Tim1: float, Tip1: float,
                                                   Tip2: float, dg: float, dd: float) -> (float, float, float,
                                                                                          float, float):
        """
        On utilise la continuité de lad_grad_T pour interpoler linéairement à partir des valeurs en im32 et ip32
        On retourne les gradients suivants ::

                             dg                      dd
                    |-------------------|---------------------------|
                    +---------------+---------------+---------------+
                    |               |   |           |               |
                   -|>      +    o -|>  |   +     o-|>      +      -|>
                    |               |   |           |               |
                    +---------------+---------------+---------------+
                 gradTi-3/2       gradTg          gradTd            gradTip32
                                gradTgi          gradTdi

        Warnings:
            Cette méthode est non convergente dans certains cas et d'ordre assez faible. Il est relativement faux de
            considérer qu'il y a linéarité de lda grad T étant donné qu'il n'y a pas continuité de la dérivée seconde.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        ldagradTgg = self.ldag*(Tim1 - Tim2)/self.dx
        ldagradTdd = self.ldad*(Tip2 - Tip1)/self.dx
        lda_gradTg = self.pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([self.dx, 2.*self.dx]))
        lda_gradTgi = self.pid_interp(np.array([ldagradTgg, ldagradTdd]),
                                      np.array([dg - (dg-0.5*self.dx)/2., dd + (dg-0.5*self.dx)/2.]))
        lda_gradTi = self.pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([dg, dd]))
        lda_gradTdi = self.pid_interp(np.array([ldagradTgg, ldagradTdd]),
                                      np.array([dg + (dd-0.5*self.dx)/2., dd - (dd-0.5*self.dx)/2.]))
        lda_gradTd = self.pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([2.*self.dx, self.dx]))
        return lda_gradTg, lda_gradTgi, lda_gradTi, lda_gradTdi, lda_gradTd

    def _get_lda_grad_T_i_from_ldagradT_interp(self, Tim3: float, Tim2: float, Tim1: float,
                                               Tip1: float, Tip2: float, Tip3: float,
                                               ag: float, ad: float) -> (float, float, float, float, float):
        """
        On utilise la continuité de lad_grad_T pour extrapoler linéairement par morceau à partir des valeurs en im52,
        im32, ip32 et ip52. On fait ensuite la moyenne des deux valeurs trouvées.
        On retourne les gradients suivants ::

                                      ag       dg
                                    |---|-----------|
                        dgi |-----o-----|---------o---------| ddi
                    +---------------+---------------+---------------+
                    |               |   |           |               |
                   -|>      +     o-|>  |   +     o-|>      +      -|>
                    |               |   |           |               |
                    +---------------+---------------+---------------+
                 gradTim32          gradTg          gradTd          gradTip32
                                 gradTgi         gradTdi

        Warnings:
            Cette méthode est non convergente dans certains cas, elle est décentrée ce qui n'est pas souvent une bonne
            idée.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        lda_gradTim52 = self.ldag*(Tim2 - Tim3)/self.dx
        lda_gradTim32 = self.ldag*(Tim1 - Tim2)/self.dx
        lda_gradTip32 = self.ldad*(Tip2 - Tip1)/self.dx
        lda_gradTip52 = self.ldad*(Tip3 - Tip2)/self.dx
        gradgradg = (lda_gradTim32 - lda_gradTim52)/self.dx
        gradgrad = (lda_gradTip52 - lda_gradTip32)/self.dx

        ldagradTig = lda_gradTim32 + gradgradg * (self.dx + ag*self.dx)
        ldagradTid = lda_gradTip32 - gradgrad * (self.dx + ad*self.dx)
        lda_gradTi = (ldagradTig + ldagradTid)/2.

        lda_gradTg = self.pid_interp(np.array([lda_gradTim32, lda_gradTi]), np.array([self.dx, ag*self.dx]))
        lda_gradTd = self.pid_interp(np.array([lda_gradTi, lda_gradTip32]), np.array([ad*self.dx, self.dx]))
        dgi = (1/2. + ag)*self.dx
        lda_gradTgi = self.pid_interp(np.array([lda_gradTim32, lda_gradTi]), np.array([self.dx/2 + dgi/2, dgi/2.]))
        ddi = (1./2 + ad)*self.dx
        lda_gradTdi = self.pid_interp(np.array([lda_gradTi, lda_gradTip32]), np.array([ddi/2., self.dx/2 + ddi/2.]))
        return lda_gradTg, lda_gradTgi, lda_gradTi, lda_gradTdi, lda_gradTd

    def _get_Ti_from_lda_grad_Ti(self, Tim1: float, Tip1: float, dg: float, dd: float,
                                 lda_gradTgi: float, lda_gradTdi: float) -> float:
        """
        Méthode d'extrapolation d'ordre 1, ou l'on récupère la température à l'interface à partir de la température et
        du gradient du côté du gradient le plus faible.

        Warnings:
            Déprécié car d'ordre 1 et extrapolation.

        Args:
            Tim1:
            Tip1:
            dg:
            dd:
            lda_gradTgi:
            lda_gradTdi:

        Returns:

        """
        if self.ldag > self.ldad:
            Ti = Tim1 + lda_gradTgi/self.ldag * dg
        else:
            Ti = Tip1 - lda_gradTdi/self.ldad * dd
        return Ti


class CellsSuiviInterface:
    """
    Cette classe contient des cellules j qui suivent l'interface ::

             Tg, gradTg                          Tghost
            +---------+----------+---------+---------+
            |         |          |         |   |     |
            |    +   -|>   +    -|>   +   -|>  |+    |
            |    0    |    1     |    2    |   |3    |
            +---------+----------+---------+---------+---------+---------+---------+
                                           |   |     |         |         |         |
                              Td, gradTd   |   |+   -|>   +   -|>   +   -|>   +    |
                                           |   |0    |    1    |    2    |    3    |
               +----------+----------+-----+---+-----+---+-----+---+-----+---+-----+
               |          |          |         |         |         |         |
               |    +     |    +     |    +    |    +    |    +    |    +    |
               |    jm2   |    jm1   |    j    |    jp1  |    jp2  |    jp3  |
               +----------+----------+---------+---------+---------+---------+
                                              T_I

    """

    def __init__(self, ldag=1., ldad=1., ag=1., dx=1., T=None, rhocpg=1., rhocpd=1., vdt=0., interp_type=None):
        self.cells_fixe = CellsInterface(ldag=ldag, ldad=ldad, ag=ag, dx=dx, T=T, rhocpg=rhocpg, rhocpd=rhocpd, vdt=vdt,
                                         interp_type=interp_type)
        self.dx = dx
        self.Tj = np.zeros((6,))
        self.Tjnp1 = np.zeros((4,))
        # Ici on calcule Tg et Td pour ensuite interpoler les valeurs à proximité de l'interface
        if self.cells_fixe.interp_type == 'Ti':
            self.cells_fixe.compute_from_Ti()
        elif self.cells_fixe.interp_type == 'Ti2':
            self.cells_fixe.compute_from_Ti2()
        elif self.cells_fixe.interp_type == 'Ti3':
            self.cells_fixe.compute_from_Ti3()
        elif self.cells_fixe.interp_type == 'gradTi':
            self.cells_fixe.compute_from_ldagradTi()
        else:
            self.cells_fixe.compute_from_ldagradTi_ordre2()

        # le zéro correspond à la position du centre de la maille i
        x_I = (ag - 1./2) * dx
        self.xj = np.linspace(-2, 3, 6)*dx + x_I - 1./2*dx

        self.Tj[:3] = self._interp_from_i_to_j_g(self.cells_fixe.Tg, self.cells_fixe.dx)
        self.Tj[3:] = self._interp_from_i_to_j_d(self.cells_fixe.Td, self.cells_fixe.dx)
        # self._lda_gradTj = None
        # self._Tj = None

    def _interp_from_i_to_j_g(self, Ti, dx):
        """
        On récupère un tableau de taille Ti - 1

        Args:
            Ti: la température à gauche
            dx (float):

        Returns:

        """
        Tj = np.empty((len(Ti) - 1,))
        xi = np.linspace(-3, 0, 4) * dx
        for j in range(len(Tj)):
            i = j+1
            d_im1_j_i = np.abs(xi[[i-1, i]] - self.xj[:3][j])
            Tj[j] = self.cells_fixe.pid_interp(Ti[[i-1, i]], d_im1_j_i)
        return Tj

    def _interp_from_i_to_j_d(self, Ti, dx):
        """
        On récupère un tableau de taille Ti - 1

        Args:
            Ti: la température à droite
            dx (float):

        Returns:

        """
        Tj = np.empty((len(Ti) - 1,))
        xi = np.linspace(0, 3, 4) * dx
        for j in range(len(Tj)):
            i = j+1
            d_im1_j_i = np.abs(xi[[i-1, i]] - self.xj[3:][j])
            Tj[j] = self.cells_fixe.pid_interp(Ti[[i-1, i]], d_im1_j_i)
        return Tj

    def timestep(self, diff, dt):
        gradT = (self.Tj[1:] - self.Tj[:-1])/self.cells_fixe.dx
        lda_grad_T = gradT * np.array([self.cells_fixe.ldag, self.cells_fixe.ldag, np.nan, self.cells_fixe.ldad,
                                       self.cells_fixe.ldad])  # taille 5
        lda_grad_T[2] = self.cells_fixe.lda_gradTi
        rho_cp_center = np.array([self.cells_fixe.rhocpg, self.cells_fixe.rhocpg, self.cells_fixe.rhocpd,
                                  self.cells_fixe.rhocpd])  # taille 4
        # le pas de temps de diffusion
        self.Tjnp1 = self.Tj[1:-1] + dt * 1/rho_cp_center * integrale_vol_div(lda_grad_T, self.dx) * diff

    def interp_T_from_j_to_i(self):
        """
        Ici on récupère un tableau de température centré en i. La valeur de température en i correspond à une
        interpolation entre la valeur à l'interface et la valeur du bon côté (j et j+1)

        Args:

        Returns:

        """
        Tj = self.Tjnp1
        Ti = np.empty((len(Tj) - 1,))
        xi = np.linspace(-1, 1, 3) * self.dx
        x_I = (self.cells_fixe.ag - 1./2) * self.dx + self.cells_fixe.vdt
        xj = np.linspace(-1, 2, 4)*self.dx + x_I - 1./2*self.dx
        for i in range(len(Ti)):
            j = i+1
            d_jm1_i_j = np.abs(xj[[j-1, j]] - xi[i])
            Ti[i] = self.cells_fixe.pid_interp(Tj[[j-1, j]], d_jm1_i_j)

        # Il faut traiter à part le cas de la cellule qu'on doit interpoler avec I
        # 3 possibilités se présentent :
        # soit l'interface est à gauche du milieu de la cellule i
        if x_I < 0.:
            # dans ce cas on fait l'interpolation entre I et j+1
            d = np.abs(np.array([x_I, xj[2]]))
            Ti[1] = self.cells_fixe.pid_interp(np.array([self.cells_fixe.Ti, Tj[2]]), d)
        # soit l'interface est à droite du milieu de la cellule i, mais toujours dans la cellule i
        elif x_I < 0.5*self.dx:
            # dans ce cas on fait l'interpolation entre j et I
            d = np.abs(np.array([xj[1], x_I]))
            Ti[1] = self.cells_fixe.pid_interp(np.array([Tj[1], self.cells_fixe.Ti]), d)
        # soit l'interface est passée à droite de la face i+1/2
        else:
            # dans ce cas on fait l'interpolation entre I et j+1 pour la température i+1
            d = np.abs(np.array([x_I, xj[3]]) - xi[2])
            Ti[2] = self.cells_fixe.pid_interp(np.array([self.cells_fixe.Ti, Tj[3]]), d)
        return Ti

    # @property
    # def lda_gradTj(self):
    #     if self._lda_gradTj is None:
    #         if self.interp_type == 'Ti':
    #             self._compute_from_Tj()
    #         else:
    #             raise NotImplementedError
    #     return self._lda_gradTj
    #
    # def _compute_from_Tj(self):
    #     """
    #     On calcule TIj et lda_gradTIj à partir de Tj et Tjp1
    #
    #     Returns:
    #         Calcule les gradients g, I, d, et Ti
    #     """
    #     self._Tj, self._lda_gradTj = _get_T_i_and_lda_grad_T_i(self.ldag, self.ldad, self.Tj[2], self.Tj[3],
    #                                                           (1. + self.ag) / 2. * self.dx,
    #                                                           (1. + self.ad) / 2. * self.dx)
