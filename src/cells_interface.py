import numpy as np
from src.main import grad, integrale_vol_div
from numba.experimental import jitclass
from numba import float64    # import the types


@jitclass([('Td', float64[:]), ('Tg', float64[:]), ('_rhocp_f', float64[:]), ('lda_f', float64[:])])
class CellsInterface:
    ldag: float
    ldad: float
    rhocpg: float
    rhocpd: float
    ag: float
    ad: float
    dx: float
    schema_conv: str
    interp_type: str
    vdt: float
    Tgc: float
    Tdc: float
    _Ti: float
    _lda_gradTi: float
    # Tg: np.ndarray((4,), dtype=float)
    # Td: np.ndarray((4,), dtype=float)
    # _rhocp_f: np.ndarray((6,), dtype=float)
    # lda_f: np.ndarray((6,), dtype=float)

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
                 schema_conv='upwind', interp_type='Ti'):
        self.ldag = ldag
        self.ldad = ldad
        self.rhocpg = rhocpg
        self.rhocpd = rhocpd
        self.ag = ag
        self.ad = 1. - ag
        self.dx = dx
        self.Tg = T[:4].copy()
        self.Tg[-1] = np.nan
        self.Td = T[3:].copy()
        self.Td[0] = np.nan
        self._rhocp_f = np.array([rhocpg, rhocpg, rhocpg, np.nan, rhocpd, rhocpd])
        self.lda_f = np.array([ldag, ldag, ldag, ldad, ldad, ldad])
        self._Ti = -1.
        self._lda_gradTi = 0.
        self.schema_conv = schema_conv
        self.vdt = vdt
        self.interp_type = interp_type
        self.Tgc = -1.
        self.Tdc = -1.

    @staticmethod
    def pid_interp(T: float64[:], d: float64[:]) -> float64:
        Tm = np.sum(T / d) / np.sum(1. / d)
        return Tm

    @property
    def T_f(self):
        if self.vdt > 0.:
            # on fait un calcul exacte pour la convection, cad qu'on calcule la température moyenne au centre du
            # volume qui va passer pour chaque face.
            # On fait une interpolation PID
            coeffg = 1./(self.dx - self.vdt/2.)
            coeffd = 1./(self.vdt/2.)
            summ = coeffd + coeffg
            coeffd /= summ
            coeffg /= summ
            T_intg = coeffg * self.Tg[:-1] + coeffd * self.Tg[1:]
            T_intd = coeffg * self.Td[:-1] + coeffd * self.Td[1:]
            return np.concatenate((T_intg, T_intd))
        elif self.schema_conv == 'upwind':
            return np.concatenate((self.Tg[:-1], self.Td[:-1]))
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
    def gradT(self) -> np.ndarray((6,), dtype=float):
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
        if self._Ti == -1.:
            if self.interp_type == 'Ti':
                self.compute_from_Ti()
            elif self.interp_type == 'gradTi':
                self.compute_from_ldagradTi()
            else:
                self.compute_from_ldagradTi_ordre2()
        return self._Ti

    @property
    def lda_gradTi(self) -> float:
        if self._Ti == -1.:
            if self.interp_type == 'Ti':
                self.compute_from_Ti()
            elif self.interp_type == 'gradTi':
                self.compute_from_ldagradTi()
            else:
                self.compute_from_ldagradTi_ordre2()
        return self._lda_gradTi

    def compute_from_ldagradTi(self):
        """
        On commence par récupérer lda_grad_Ti, gradTg, gradTd par continuité à partir de gradTim32 et gradTip32.
        On en déduit Ti par continuité à gauche et à droite.

        Returns:
            Calcule les gradients g, I, d, et Ti. gradTig et gradTid sont les gradients centrés entre TI et Tim1 et TI
            et Tip1
        """
        lda_gradTg, lda_gradTig, self._lda_gradTi, lda_gradTid, lda_gradTd = \
            self.get_lda_grad_T_i_from_ldagradT_continuity(self.Tg[1], self.Tg[2], self.Td[1], self.Td[2],
                                                           (3./2. + self.ag)*self.dx, (3./2. + self.ad)*self.dx)

        self._Ti = self.get_Ti_from_lda_grad_Ti(self.Tg[2], self.Td[1],
                                                (0.5+self.ag)*self.dx, (0.5+self.ad)*self.dx,
                                                lda_gradTig, lda_gradTid)
        self.Tg[-1] = self.Tg[2] + lda_gradTg/self.ldag * self.dx
        self.Td[0] = self.Td[1] - lda_gradTd/self.ldad * self.dx

    def compute_from_ldagradTi_ordre2(self):
        """
        On commence par récupérer lda_grad_Ti par continuité à partir de gradTim52 gradTim32 gradTip32
        et gradTip52.
        On en déduit Ti par continuité à gauche et à droite.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        lda_gradTg, lda_gradTig, self._lda_gradTi, lda_gradTid, lda_gradTd = \
            self.get_lda_grad_T_i_from_ldagradT_interp(self.Tg[0], self.Tg[1], self.Tg[2],
                                                       self.Td[1], self.Td[2], self.Td[3],
                                                       self.ag, self.ad)

        self._Ti = self.get_Ti_from_lda_grad_Ti(self.Tg[2], self.Td[1],
                                                (0.5+self.ag)*self.dx, (0.5+self.ad)*self.dx,
                                                lda_gradTig, lda_gradTid)
        self.Tg[-1] = self.Tg[2] + lda_gradTig/self.ldag * self.dx
        self.Td[0] = self.Td[1] - lda_gradTid/self.ldad * self.dx

    def compute_from_Ti(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1.
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        self._Ti, self._lda_gradTi = self.get_T_i_and_lda_grad_T_i(self.Tg[-2], self.Td[1],
                                                                   (1. + self.ag) / 2. * self.dx,
                                                                   (1. + self.ad) / 2. * self.dx)
        grad_Tg = self.pid_interp(np.array([self.gradTg[1], self._lda_gradTi/self.ldag]),
                                  np.array([1., self.ag])*self.dx)
        grad_Td = self.pid_interp(np.array([self._lda_gradTi/self.ldad, self.gradTd[1]]),
                                  np.array([self.ad, 1.])*self.dx)
        self.Tg[-1] = self.Tg[-2] + grad_Tg * self.dx
        self.Td[0] = self.Td[1] - grad_Td * self.dx

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
        self.Tgc, self.Tdc = np.dot(np.linalg.inv(system), np.array([h, T_mean])).squeeze()

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
        self._Ti, self._lda_gradTi = self.get_T_i_and_lda_grad_T_i(self.Tgc, self.Tdc, self.ag / 2. * self.dx,
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

    def get_T_i_and_lda_grad_T_i(self, Tg: float, Td: float, dg: float, dd: float) -> (float, float):
        T_i = (self.ldag/dg*Tg + self.ldad/dd*Td) / (self.ldag/dg + self.ldad/dd)
        lda_grad_T_ig = self.ldag * (T_i - Tg)/dg
        lda_grad_T_id = self.ldad * (Td - T_i)/dd
        return T_i, (lda_grad_T_ig + lda_grad_T_id)/2.

    def get_lda_grad_T_i_from_ldagradT_continuity(self, Tim2: float, Tim1: float, Tip1: float,
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

    def get_lda_grad_T_i_from_ldagradT_interp(self, Tim3: float, Tim2: float, Tim1: float,
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

    def get_Ti_from_lda_grad_Ti(self, Tim1: float, Tip1: float, dg: float, dd: float,
                                lda_gradTgi: float, lda_gradTdi: float) -> float:
        if self.ldag > self.ldad:
            Ti = Tim1 + lda_gradTgi/self.ldag * dg
        else:
            Ti = Tip1 - lda_gradTdi/self.ldad * dd
        return Ti


class CellsSuiviInterface:
    """
    Cette classe contient des cellules j qui suivent l'interface ::

           +----------+----------+---------+---------+---------+---------+---------+
           |          |          |         |   |     |         |         |         |
           |    +     |    +     |    +    |   |+    |    +    |    +    |    +    |
           |    im3   |    im2   |    im1  |   |i    |    ip1  |    ip2  |    ip3  |
           +---+----------+----------+---------+---------+---------+---------+-----+
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
        elif self.cells_fixe.interp_type == 'gradTi':
            self.cells_fixe.compute_from_ldagradTi()
        else:
            self.cells_fixe.compute_from_ldagradTi_ordre2()

        self.Tj[:3] = self._interp_from_i_to_j_g(self.cells_fixe.Tg, self.cells_fixe.ag, self.cells_fixe.dx)
        self.Tj[3:] = self._interp_from_i_to_j_d(self.cells_fixe.Td, self.cells_fixe.ag, self.cells_fixe.dx)
        # self._lda_gradTj = None
        # self._Tj = None

    def _interp_from_i_to_j_g(self, Ti, ag, dx):
        """
        On récupère un tableau de taille Ti - 1

        Args:
            Ti: la température à gauche
            ag (float):
            dx (float):

        Returns:

        """
        Tj = np.empty((len(Ti) - 1,))
        x_I = (ag - 1./2) * dx
        xj = np.linspace(-2, 0, 3)*dx + x_I - 1./2*dx
        xi = np.linspace(-3, 0, 4) * dx
        for j in range(len(Tj)):
            i = j+1
            d_im1_j_i = np.abs(xi[[i-1, i]] - xj[j])
            Tj[j] = self.cells_fixe.pid_interp(Ti[[i-1, i]], d_im1_j_i)
        return Tj

    def _interp_from_i_to_j_d(self, Ti, ag, dx):
        """
        On récupère un tableau de taille Ti - 1

        Args:
            Ti: la température à droite
            ag (float):
            dx (float):

        Returns:

        """
        Tj = np.empty((len(Ti) - 1,))
        x_I = (ag - 1./2) * dx
        xj = np.linspace(1, 3, 3)*dx + x_I - 1./2*dx
        xi = np.linspace(0, 3, 4) * dx
        for j in range(len(Tj)):
            i = j+1
            d_im1_j_i = np.abs(xi[[i-1, i]] - xj[j])
            Tj[j] = self.cells_fixe.pid_interp(Ti[[i-1, i]], d_im1_j_i)
        return Tj

    def timestep(self, diff, dt):
        gradT = grad(self.Tj, self.cells_fixe.dx)  # on se fiche
        lda_grad_T = gradT[1:-1] * np.array([self.cells_fixe.ldag, self.cells_fixe.ldag, np.nan, self.cells_fixe.ldad,
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
    #     self._Tj, self._lda_gradTj = get_T_i_and_lda_grad_T_i(self.ldag, self.ldad, self.Tj[2], self.Tj[3],
    #                                                           (1. + self.ag) / 2. * self.dx,
    #                                                           (1. + self.ad) / 2. * self.dx)
