import numpy as np


class CellsInterfaceBase:
    def __init__(self, ldag=1., ldad=1., ag=1., dx=1., T=None, rhocpg=1., rhocpd=1., vdt=None, schema_conv='upwind'):
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
        self.Ti = None
        self.lda_gradTi = None
        self.schema_conv = schema_conv
        self.vdt = vdt

    @property
    def T_f(self):
        if self.vdt is not None:
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
    def gradTg(self):
        return (self.Tg[1:] - self.Tg[:-1])/self.dx

    @property
    def gradTd(self):
        return (self.Td[1:] - self.Td[:-1])/self.dx

    @property
    def gradT(self):
        return np.concatenate((self.gradTg, self.gradTd))

    @property
    def rhocp_f(self):
        if self.vdt is not None:
            coeff_d = min(self.vdt, self.ad*self.dx)/self.vdt
            self._rhocp_f[3] = coeff_d * self.rhocpd + (1. - coeff_d) * self.rhocpg
            return self._rhocp_f
        else:
            self._rhocp_f[3] = self.rhocpd
            return self._rhocp_f

    @property
    def rhocpT_f(self):
        if self.vdt is not None:
            # TODO: implémenter une méthode qui renvoie rho * cp * T intégré sur le volume qui va passer d'une cellule à
            #  l'autre. Cette précision n'est peut-être pas nécessaire
            rhocpTf = self.rhocp_f * self.T_f
            return rhocpTf
        else:
            return self.rhocp_f * self.T_f


class CellsInterface1eq(CellsInterfaceBase):
    def __init__(self, ldag=1., ldad=1., ag=1., dx=1., T=None, rhocpg=1., rhocpd=1., vdt=None):
        """
        Cellule type ::

                                        Ti,
                                        lda_gradTi
                                         Ti0g
                                         Ti0d
                                    Tgf       Tdf
               +----------+---------+---------+---------+---------+
               |          |         |   |     |         |         |
               |    +    -|>   +   -|>  |+   -|>   +   -|>   +    |
               |          |         |   |     |         |         |
               +----------+---------+---------+---------+---------+
                        gradTi-3/2  gradTg    gradTd    gradTi+3/2

        Dans ce modèle on connait initialement les températures aux centes des cellules monophasiques. On ne se sert pas
        de la température de la cellule traversée par l'interface. Toutes les valeurs dans la cellule sont reconstruites
        avec les valeurs adjacentes.
        Il n'y a qu'une équation d'évolution, elle porte sur T, en dehors de la maille diphasique. La valeur de la
        maille diphasique ne sert pas entre 2 pas de temps tant que l'interface ne change pas de cellule.

        Args:
            ldag:
            ldad:
            ag:
            dx:
            T:
            rhocpg:
            rhocpd:
        """
        super().__init__(ldag=ldag, ldad=ldad, ag=ag, dx=dx, T=T, rhocpg=rhocpg, rhocpd=rhocpd, vdt=vdt,
                         schema_conv='upwind')

    def compute_from_ldagradTi(self):
        """
        On commence par récupérer lda_grad_Ti, gradTg, gradTd par continuité à partir de gradTim32 et gradTip32.
        On en déduit Ti par continuité à gauche et à droite.

        Returns:
            Calcule les gradients g, I, d, et Ti. gradTig et gradTid sont les gradients centrés entre TI et Tim1 et TI
            et Tip1
        """
        lda_gradTg, lda_gradTig, self.lda_gradTi, lda_gradTid, lda_gradTd = \
            get_lda_grad_T_i_from_ldagradT_continuity(self.ldag, self.ldad, *self.Tg[[1, 2]], *self.Td[[1, 2]],
                                                      (3./2. + self.ag)*self.dx, (3./2. + self.ad)*self.dx, self.dx)

        self.Ti = get_Ti_from_lda_grad_Ti(self.ldag, self.ldad, self.Tg[2], self.Td[1],
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
        lda_gradTg, lda_gradTig, self.lda_gradTi, lda_gradTid, lda_gradTd = \
            get_lda_grad_T_i_from_ldagradT_interp(self.ldag, self.ldad, *self.Tg[:-1], *self.Td[1:],
                                                  self.ag, self.ad, self.dx)

        self.Ti = get_Ti_from_lda_grad_Ti(self.ldag, self.ldad, self.Tg[2], self.Td[1],
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
        self.Ti, self.lda_gradTi = get_T_i_and_lda_grad_T_i(self.ldag, self.ldad, self.Tg[-2], self.Td[1],
                                                            (1.+self.ag)/2.*self.dx, (1.+self.ad)/2.*self.dx)
        grad_Tg = pid_interp(np.array([self.gradTg[1], self.lda_gradTi/self.ldag]), np.array([1., self.ag])*self.dx)
        grad_Td = pid_interp(np.array([self.lda_gradTi/self.ldad, self.gradTd[1]]), np.array([self.ad, 1.])*self.dx)
        self.Tg[-1] = self.Tg[-2] + grad_Tg * self.dx
        self.Td[0] = self.Td[1] - grad_Td * self.dx


class CellsInterface2eq(CellsInterfaceBase):
    def __init__(self, ldag=1., ldad=1., ag=1., dx=1., T=None, rhocpg=1., rhocpd=1., vdt=None):
        """
        Cellule type ::

                                        Ti,
                                        lda_gradTi
                                          Ti0g
                                          Ti0d
                                  Tgf Tgc  Tdc Tdf
               +----------+---------+---------+---------+---------+
               |          |         | gc|  dc |         |         |
               |    +    -|>   +   -|>* | +* -|>   +   -|>   +    |
               |          |         |   |     |         |         |
               +----------+---------+---------+---------+---------+
                        gradTi-3/2  gradTg    gradTd    gradTi+3/2

        Dans ce modèle on connait initialement les températures moyenne aux centes de toutes les cellules.
        On reconstruit les valeurs de Tgc et Tdc avec le système sur la valeur moyenne de température dans la maille et
        la valeur moyenne de l'énergie.
        Au contraire de la classe mère on utilise uniquement les valeurs de la mailles diphasiques pour trouver Ti et
        lda_grad_Ti.
        Ensuite évidemment on interpole là ou on en aura besoin.
        Il faudra faire 2 équations d'évolution dans la cellule i, une sur h et une sur T.

        Args:
            ldag:
            ldad:
            ag:
            dx:
            T:
            rhocpg:
            rhocpd:
        """
        super().__init__(ldag=ldag, ldad=ldad, ag=ag, dx=dx, T=T, rhocpg=rhocpg, rhocpd=rhocpd, vdt=vdt)
        if rhocpd == rhocpg:
            raise Exception('Le problème est à rho cp constant, il n a pas besoin et ne peut pas être traité avec cette'
                            'méthode')
        self.Tgc = None
        self.Tdc = None

    def _compute_Tgc_Tdc(self, h, T_mean):
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

    def compute_from_h_T(self, h, T_mean):
        """
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
        self.Ti, self.lda_gradTi = get_T_i_and_lda_grad_T_i(self.ldag, self.ldad, self.Tgc, self.Tdc,
                                                            self.ag/2.*self.dx, self.ad/2.*self.dx)
        # print('lda grad T i : ', self.lda_gradTi)

        # Calcul des gradient aux faces

        # À gauche :
        aim1_gc = 0.5 + self.ag/2.
        gradTim1_gc = (self.Tgc - self.Tg[-2])/(aim1_gc*self.dx)
        gradTg_v = np.array([self.gradTg[-2], gradTim1_gc, self.lda_gradTi/self.ldag])
        dist = np.array([1., np.abs(0.5 - aim1_gc/2.), self.ag])*self.dx
        gradTg = pid_interp(gradTg_v, dist)
        self.Tg[-1] = self.Tg[-2] + self.dx * gradTg

        # À droite :
        aip1_dc = 0.5 + self.ad/2.
        gradTip1_dc = (self.Td[1] - self.Tdc)/(aip1_dc*self.dx)
        gradTd_v = np.array([self.lda_gradTi/self.ldad, gradTip1_dc, self.gradTd[1]])
        dist = np.array([self.ad, np.abs(0.5 - aip1_dc/2.), 1.])*self.dx
        gradTd = pid_interp(gradTd_v, dist)
        self.Td[0] = self.Td[1] - self.dx * gradTd


def get_T_i_and_lda_grad_T_i(ldag, ldad, Tg, Td, dg, dd):
    T_i = (ldag/dg*Tg + ldad/dd*Td) / (ldag/dg + ldad/dd)
    lda_grad_T_ig = ldag * (T_i - Tg)/dg
    lda_grad_T_id = ldad * (Td - T_i)/dd
    return T_i, (lda_grad_T_ig + lda_grad_T_id)/2.


def get_lda_grad_T_i_from_ldagradT_continuity(ldag, ldad, Tim2, Tim1, Tip1, Tip2, dg, dd, dx):
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
    ldagradTgg = ldag*(Tim1 - Tim2)/dx
    ldagradTdd = ldad*(Tip2 - Tip1)/dx
    lda_gradTg = pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([dx, 2.*dx]))
    lda_gradTgi = pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([dg - (dg-0.5*dx)/2., dd + (dg-0.5*dx)/2.]))
    lda_gradTi = pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([dg, dd]))
    lda_gradTdi = pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([dg + (dd-0.5*dx)/2., dd - (dd-0.5*dx)/2.]))
    lda_gradTd = pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([2.*dx, dx]))
    return lda_gradTg, lda_gradTgi, lda_gradTi, lda_gradTdi, lda_gradTd


def get_lda_grad_T_i_from_ldagradT_interp(ldag, ldad, Tim3, Tim2, Tim1, Tip1, Tip2, Tip3, ag, ad, dx):
    """
    On utilise la continuité de lad_grad_T pour extrapoler linéairement par morceau à partir des valeurs en im52, im32,
    ip32 et ip52. On fait ensuite la moyenne des deux valeurs trouvées.
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
    lda_gradTim52 = ldag*(Tim2 - Tim3)/dx
    lda_gradTim32 = ldag*(Tim1 - Tim2)/dx
    lda_gradTip32 = ldad*(Tip2 - Tip1)/dx
    lda_gradTip52 = ldad*(Tip3 - Tip2)/dx
    gradgradg = (lda_gradTim32 - lda_gradTim52)/dx
    gradgrad = (lda_gradTip52 - lda_gradTip32)/dx

    ldagradTig = lda_gradTim32 + gradgradg * (dx + ag*dx)
    ldagradTid = lda_gradTip32 - gradgrad * (dx + ad*dx)
    lda_gradTi = (ldagradTig + ldagradTid)/2.

    lda_gradTg = pid_interp(np.array([lda_gradTim32, lda_gradTi]), np.array([dx, ag*dx]))
    lda_gradTd = pid_interp(np.array([lda_gradTi, lda_gradTip32]), np.array([ad*dx, dx]))
    dgi = (1/2. + ag)*dx
    lda_gradTgi = pid_interp(np.array([lda_gradTim32, lda_gradTi]), np.array([dx/2 + dgi/2, dgi/2.]))
    ddi = (1./2 + ad)*dx
    lda_gradTdi = pid_interp(np.array([lda_gradTi, lda_gradTip32]), np.array([ddi/2., dx/2 + ddi/2.]))
    return lda_gradTg, lda_gradTgi, lda_gradTi, lda_gradTdi, lda_gradTd


def get_Ti_from_lda_grad_Ti(ldag, ldad, Tim1, Tip1, dg, dd, lda_gradTgi, lda_gradTdi):
    if ldag > ldad:
        Ti = Tim1 + lda_gradTgi/ldag * dg
    else:
        Ti = Tip1 - lda_gradTdi/ldad * dd
    return Ti


def pid_interp(T, d):
    Tm = np.sum(T/d) / np.sum(1./d)
    return Tm
