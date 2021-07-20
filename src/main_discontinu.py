from src.main import *


class BulleTemperature(Bulles):
    def __init__(self, markers=None, phy_prop=None, n_bulle=None, Delta=1., x=None):
        super().__init__(markers, phy_prop, n_bulle, Delta)
        self.T = np.zeros_like(self.markers)
        self.Tg = np.zeros_like(self.markers)
        self.Td = np.zeros_like(self.markers)
        self.gradTg = np.zeros_like(self.markers)
        self.gradTd = np.zeros_like(self.markers)
        self.xg = np.zeros_like(self.markers)
        self.xd = np.zeros_like(self.markers)
        self.lda_grad_T = np.zeros_like(self.markers)
        self.ind = None
        if x is not None:
            self.set_indices_markers(x)
            print('initialisation des indices des cellules des marqueurs')

    def copy(self):
        copie = super().copy()
        copie.T = self.T.copy()
        copie.lda_grad_T = self.lda_grad_T.copy()
        if self.ind is None:
            copie.ind = None
        else:
            copie.ind = self.ind.copy()
        return copie

    def set_indices_markers(self, x):
        """
        Retourne les indices des cellules traveresées par l'interface.
        Il serait beaucoup plus économe de considérer que les marqueurs ne se déplacent pas de plus d'une cellule entre
        deux mise à jour et donc qu'il est facile de vérifier la position des marqueurs sur les 2 cellules voisines
        plutôt que partout dans le domaine. Cela dit en python il n'existe pas de moyen optimisé de faire ça.
        Args:
            x: le tableau des positions

        Returns:
            Met à jour self.ind, le tableau d'indices de la meme forme que self.markers
        """
        res = []
        dx = x[1] - x[0]
        for marks in self.markers:
            ind1 = (np.abs(marks[0] - x) < dx/2.).nonzero()[0][0]
            ind2 = (np.abs(marks[1] - x) < dx/2.).nonzero()[0][0]
            res.append([ind1, ind2])
        self.ind = np.array(res, dtype=np.int)


def get_prop(prop, i, liqu_a_gauche=True):
    if liqu_a_gauche:
        ldag = prop.phy_prop.lda1
        rhocpg = prop.phy_prop.rho_cp1
        ldad = prop.phy_prop.lda2
        rhocpd = prop.phy_prop.rho_cp2
        ag = prop.I[i]
        ad = 1. - prop.I[i]
    else:
        ldag = prop.phy_prop.lda2
        rhocpg = prop.phy_prop.rho_cp2
        ldad = prop.phy_prop.lda1
        rhocpd = prop.phy_prop.rho_cp1
        ag = 1. - prop.I[i]
        ad = prop.I[i]
    return ldag, rhocpg, ag, ldad, rhocpd, ad


class CellsInterface:
    def __init__(self, ldag=1., ldad=1., ag=1., dx=1., T=None, rhocpg=1., rhocpd=1.):
        """
        Cellule type ::

                                        Ti,
                                        lda_gradTi
                                          Ti0g
                                          Ti0d
                                    Tgf       Tdf
               +----------+---------+---------+---------+---------+
               |          |         |   |     |         |         |
               |    +    -|>   +   -|>  | +  -|>   +   -|>   +    |
               |    0     |    1    |   | 2   |    3    |    4    |
               +----------+---------+---------+---------+---------+
                          0         1         2         3
                        gradTi-3/2  gradTg    gradTd    gradTi+3/2

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
        self.ag = ag
        self.ad = 1. - ag
        self.dx = dx
        self.T = T.copy()
        self.T[2] = np.nan
        # self.x = np.array([-2., -1., 0., 1., 2.])*dx
        # self.x_f = np.array([-1.5, -0.5, 0.5, 1.5])*dx
        self.gradT = (self.T[1:] - self.T[:-1])/dx
        self.lda_f = np.array([ldag, ldag, ldad, ldad])
        self.rhocp_f = np.array([rhocpg, rhocpg, np.nan, rhocpd])
        self.Ti = None
        self.lda_gradTi = None
        self.Ti0d = None
        self.Ti0g = None

    def compute_from_ldagradTi(self):
        """
        On commence par récupérer lda_grad_Ti, gradTg, gradTd par continuité à partir de gradTim32 et gradTip32.
        On en déduit Ti par continuité à gauche et à droite.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        lda_gradTg, lda_gradTig, self.lda_gradTi, lda_gradTid, lda_gradTd = \
            get_lda_grad_T_i_from_ldagradT_continuity(self.ldag, self.ldad, *self.T[[0, 1, 3, 4]],
                                                      (3./2. + self.ag)*self.dx, (3./2. + self.ad)*self.dx, self.dx)
        grad_Tg = lda_gradTg/self.ldag
        grad_Td = lda_gradTd/self.ldad

        self.Ti = get_Ti_from_lda_grad_Ti(self.ldag, self.ldad, *self.T[[1, 3]],
                                          (0.5+self.ag)*self.dx, (0.5+self.ad)*self.dx,
                                          lda_gradTig, lda_gradTid)
        self.gradT[1:-1] = [grad_Tg, grad_Td]

    def compute_from_Ti(self):
        """
        On commence par calculer Ti et lda_grad_Ti à partir de Tim1 et Tip1.
        Ensuite on procède au calcul de grad_Tg et grad_Td en interpolant avec lda_grad_T_i et les gradients m32 et p32.
        Il y a une grande incertitude sur lda_grad_Ti, donc ce n'est pas terrible d'interpoler comme ça.

        Returns:
            Calcule les gradients g, I, d, et Ti
        """
        self.Ti, self.lda_gradTi = get_T_i_and_lda_grad_T_i(self.ldag, self.ldad, *self.T[[1, 3]],
                                                            (1.+self.ag)/2.*self.dx, (1.+self.ad)/2.*self.dx)
        grad_Tg = pid_interp(np.array([self.gradT[0], self.lda_gradTi/self.ldag]), np.array([1., self.ag])*self.dx)
        grad_Td = pid_interp(np.array([self.lda_gradTi/self.ldad, self.gradT[-1]]), np.array([self.ad, 1.])*self.dx)
        self.gradT[1:-1] = [grad_Tg, grad_Td]

    def set_Ti0_upwind(self):
        """
        Ici on choisit Ti0 ghost de manière à ce qu'il corresponde à la température ghost de la phase avale au centre de
        la cellule i0

        Returns:
            Set la température i0 à Tid ou Tig selon le sens de la vitesse
        """
        self.Ti0d = self.T[3] - self.gradT[2] * self.dx
        self.Ti0g = self.T[1] + self.gradT[1] * self.dx
        self.T[2] = self.Ti0d


def get_T_i_and_lda_grad_T_i(ldag, ldad, Tg, Td, dg, dd):
    T_i = (ldag/dg*Tg + ldad/dd*Td) / (ldag/dg + ldad/dd)
    lda_grad_T_ig = ldag * (T_i - Tg)/dg
    lda_grad_T_id = ldad * (Td - T_i)/dd
    return T_i, (lda_grad_T_ig + lda_grad_T_id)/2.


def get_lda_grad_T_i_from_ldagradT_continuity(ldag, ldad, Tim2, Tim1, Tip1, Tip2, dg, dd, dx):
    ldagradTgg = ldag*(Tim1 - Tim2)/dx
    ldagradTdd = ldad*(Tip2 - Tip1)/dx
    lda_gradTg = pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([dx, 2.*dx]))
    lda_gradTgi = pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([dg - (dg-0.5*dx)/2., dd + (dg-0.5*dx)/2.]))
    lda_gradTi = pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([dg, dd]))
    lda_gradTdi = pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([dg + (dd-0.5*dx)/2., dd - (dd-0.5*dx)/2.]))
    lda_gradTd = pid_interp(np.array([ldagradTgg, ldagradTdd]), np.array([2.*dx, dx]))
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


class ProblemDiscontinu(Problem):
    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None):
        """
        Cette classe résout le problème en 3 étapes :

        - on calcule le nouveau T comme avant (avec un stencil de 1 à proximité des interfaces par simplicité)
        - on calcule précisemment T1 et T2 ansi que les bons flux aux faces, on met à jour T
        - on met à jour T_i et lda_grad_T_i

        Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
        continuité de lda_grad_T et T à l'interface.

        Args:
            T0: la fonction initiale de température
            markers: les bulles
            num_prop: les propriétés numériques du calcul
            phy_prop: les propriétés physiques du calcul
        """
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        if self.num_prop.schema != 'upwind':
            raise Exception('Cette version ne marche que pour un schéma upwind')
        self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = 'gradTi'
        else:
            self.interp_type = interp_type

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(markers=markers, phy_prop=self.phy_prop)
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        elif isinstance(markers, Bulles):
            return BulleTemperature(markers=markers.markers, phy_prop=self.phy_prop, x=self.num_prop.x)
        else:
            print(markers)
            raise NotImplementedError

    def corrige_interface_aymeric1(self):
        """
        Dans cette approche on calclue Ti et lda_gradTi soit en utilisant la continuité avec Tim1 et Tip1, soit en
        utilisant la continuité des lda_grad_T calculés avec Tim2, Tim1, Tip1 et Tip2.
        Dans les deux cas il est à noter que l'on utilise pas les valeurs présentes dans la cellule de l'interface.
        On en déduit ensuite les gradients de température aux faces, les températures aux faces (en fait les
        températures au milieu parce qu'on fait du upwind).

        Returns:
            Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
            jour T_old
        """
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v*self.dt)
        Inp1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rhocp_np1 = self.phy_prop.rho_cp1 * Inp1 + self.phy_prop.rho_cp2 * (1.-Inp1)
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(self.bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im2, im1, i0, ip1, ip2 = cl_perio(len(self.T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(self, i, liqu_a_gauche=from_liqu_to_vap)
                cells = CellsInterface(ldag, ldad, ag, dx, self.T_old[[im2, im1, i0, ip1, ip2]], rhocpg=rhocpg,
                                       rhocpd=rhocpd)
                if self.interp_type == 'Ti':
                    cells.compute_from_Ti()
                else:
                    cells.compute_from_ldagradTi()
                cells.set_Ti0_upwind()
                # On vérifie que l'interface ne traverse pas la face, sinon on met à jour rho_cp_face_droite qui est
                # un mélange au prorato du temps de passage entre la phase de droite et la phase de gauche.
                if self.phy_prop.v > 0.:
                    dt1_dt = max(self.I[i0] * self.num_prop.dx / (self.phy_prop.v * self.dt), 1.)
                else:
                    dt1_dt = 1.
                dt2_dt = 1. - dt1_dt
                cells.rhocp_f[2] = dt1_dt*rhocpd + dt2_dt*rhocpd

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                _, grad_Tg, grad_Td, _ = cells.gradT

                # Tgf = pid_interp(np.array([self.T_old[im1], Ti]), np.array([0.5, ag/2.])*self.num_prop.dx)
                # Tdf = pid_interp(np.array([Ti, self.T_old[ip1]]), np.array([ad/2., 0.5])*self.num_prop.dx)

                self.bulles.Tg[i_int, ist] = cells.Ti0g
                self.bulles.Td[i_int, ist] = cells.Ti0d
                self.bulles.gradTg[i_int, ist] = grad_Tg
                self.bulles.gradTd[i_int, ist] = grad_Td

                dV = self.num_prop.dx*self.phy_prop.dS
                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # interpolation upwind
                # DONE: l'écrire en version flux pour être sûr de la conservation
                int_div_rhocpT_u = 1./dV * integrale_volume_div(cells.T[1:-1], cells.rhocp_f*self.phy_prop.v,
                                                                cl=0, dS=self.phy_prop.dS, schema='upwind',
                                                                cv_0=cells.T[0], cv_n=cells.T[-1])
                int_div_lda_grad_T = 1./dV * integrale_volume_div(np.ones((3,)), cells.lda_f*cells.gradT,
                                                                  schema='center', dS=self.phy_prop.dS, cl=1)

                # Correction des cellules
                ind_to_change = [im1, i0, ip1]
                self.T[ind_to_change] = (self.T_old[ind_to_change]*self.rho_cp_a[ind_to_change] +
                                         self.dt * (-int_div_rhocpT_u +
                                                    self.phy_prop.diff * int_div_lda_grad_T)) / rhocp_np1[ind_to_change]

        self.T_old = self.T.copy()

    def euler_timestep(self, debug=None, bool_debug=False):
        super().euler_timestep(debug=debug, bool_debug=bool_debug)
        self.corrige_interface_aymeric1()

    def update_markers(self):
        super().update_markers()
        self.bulles.set_indices_markers(self.num_prop.x)


def cl_perio(n, i):
    if i == 0:
        im2 = -2
        im1 = -1
        i = i
        ip1 = i+1
        ip2 = i+2
    elif i == 1:
        im2 = -1
        im1 = 0
        i = i
        ip1 = i+1
        ip2 = i+2
    elif i == n - 1:
        im1 = i-1
        im2 = i-2
        i = i
        ip1 = 0
        ip2 = 1
    elif i == n - 2:
        im1 = i - 1
        im2 = i - 2
        i = i
        ip1 = i+1
        ip2 = 0
    else:
        im1 = i - 1
        im2 = i - 2
        i = i
        ip1 = i + 1
        ip2 = i + 2
    return im2, im1, i, ip1, ip2
