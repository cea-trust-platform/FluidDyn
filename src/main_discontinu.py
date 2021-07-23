from src.main import *
from copy import deepcopy
from src.cells_interface import CellsInterface1eq, CellsInterface2eq


class BulleTemperature(Bulles):
    def __init__(self, markers=None, phy_prop=None, n_bulle=None, Delta=1., x=None):
        """
        On ajoute des champs afin de post-traiter les données calculées à l'interface.
        On peut maintenant savoir quelles sont les mailles diphasiques.

        Args:
            markers:
            phy_prop:
            n_bulle:
            Delta:
            x:
        """
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
            self.x = x
            self._set_indices_markers(x)
        else:
            raise Exception('x est un argument obligatoire')

    def copy(self):
        """
        Cette copie est récursive. De cette manière il n'y a pas de crainte à avoir à changer les valeurs de l'originale
        en changeant les valeurs dans la copie.

        Returns:
            Une copie récursive
        """
        copie = deepcopy(self)
        return copie

    def _set_indices_markers(self, x):
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

    def shift(self, dx):
        super().shift(dx)
        self._set_indices_markers(self.x)


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


class ProblemDiscontinuEnergieTemperature(Problem):
    T: np.ndarray
    I: np.ndarray
    h: np.ndarray
    bulles: BulleTemperature

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None):
        """
        Cette classe résout le problème en couplant une équation sur la température et une équation sur l'énergie
        interne au niveau des interfaces.
        On a donc un tableau T et un tableau h

        - on calcule dans les mailles diphasiques Tgc et Tdc les températures au centres de la partie remplie par la
        phase à gauche et la partie remplie par la phase à droite.
        - on en déduit en interpolant des flux aux faces
        - on met à jour T et h avec des flux exprimés de manière monophasique.

        Le problème de cette formulation est qu'elle fait intervenir l'équation sur la température alors qu'on sait
        que cette équation n'est pas terrible.

        Args:
            T0: la fonction initiale de température
            markers: les bulles
            num_prop: les propriétés numériques du calcul
            phy_prop: les propriétés physiques du calcul
        """
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        self.h = self.rho_cp_a*self.T
        self.intS_T_u_n_dS = np.empty_like(self.num_prop.x_f)
        self.intS_rho_cp_T_u_n_dS = np.empty_like(self.num_prop.x_f)
        self.intS_lda_gradT_n_dS = np.empty_like(self.num_prop.x_f)

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

    def euler_timestep(self, debug=None, bool_debug=False):
        # on devrait plutôt calculer les flux, les stocker
        self.intS_T_u_n_dS = interpolate_from_center_to_face_weno(self.T*self.phy_prop.v, cl=1) * \
            self.phy_prop.dS
        self.intS_rho_cp_T_u_n_dS = interpolate_from_center_to_face_weno(self.rho_cp_a*self.T*self.phy_prop.v, cl=1) * \
            self.phy_prop.dS
        self.intS_lda_gradT_n_dS = interpolate_from_center_to_face_weno(self.Lda_h) * grad(self.T, self.num_prop.dx) *\
            self.phy_prop.dS
        # Est ce qu'on fait entièrement en monofluide pour la température moyenne ou est-ce qu'on s'amuse à faire les
        # choses bien comme il faut au moins pour la diffusion ? (Avec des ldas*gradT purement monophasiques)
        # Faisons les choses bien
        for i_int, (i1, i2) in enumerate(self.bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(self.T), i)
                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(self, i, liqu_a_gauche=from_liqu_to_vap)

                cells = CellsInterface2eq(ldag, ldad, ag, self.num_prop.dx, self.T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                                          rhocpg=rhocpg, rhocpd=rhocpd, vdt=self.phy_prop.v*self.dt)
                cells.compute_from_h_T(self.h[i0], self.T[i0])

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi

                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                # Correction des flux entrant et sortant de la maille diphasique
                ind_flux_to_change = [im1, i0, ip1, ip2]
                self.intS_T_u_n_dS[ind_flux_to_change] = self.phy_prop.dS * self.phy_prop.v * cells.T_f[1:-1]
                self.intS_rho_cp_T_u_n_dS[ind_flux_to_change] = self.phy_prop.dS * self.phy_prop.v * \
                    cells.rhocp_f[1:-1] * cells.T_f[1:-1]
                self.intS_lda_gradT_n_dS[ind_flux_to_change] = self.phy_prop.dS * cells.lda_f[1:-1] * cells.gradT[1:-1]
                # TODO: bouger cette écriture et l'utiliser avec un autre type de cellule, ou on reconstruit la
                #       température exacte en Ti, et non avec la température moyenne de maille
                # self.intS_lda_gradT_n_dS[ind_flux_to_change] = self.phy_prop.dS * \
                #     ((cells.lda[1:] * cells.T[1:] - cells.lda[:-1] * cells.T[:-1]) -
                #      cells.T_f[1:3] * (cells.lda[1:] - cells.lda[:-1])) / self.num_prop.dx

        dV = self.phy_prop.dS * self.num_prop.dx
        int_div_T_u = 1/dV * (self.intS_T_u_n_dS[1:] - self.intS_T_u_n_dS[:-1])
        int_div_rho_cp_T_u = 1/dV * (self.intS_rho_cp_T_u_n_dS[1:] - self.intS_rho_cp_T_u_n_dS[:-1])
        int_div_lda_grad_T = 1/dV * (self.intS_lda_gradT_n_dS[1:] - self.intS_lda_gradT_n_dS[:-1])
        rho_cp_inv_h = 1. / self.rho_cp_h
        self.T += self.dt * (-int_div_T_u + self.phy_prop.diff * rho_cp_inv_h * int_div_lda_grad_T)
        self.h += self.dt * (-int_div_rho_cp_T_u + self.phy_prop.diff * int_div_lda_grad_T)


class ProblemDiscontinu(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

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
        self.intS_T_u_n_dS = np.empty_like(self.num_prop.x_f)
        self.intS_rho_cp_T_u_n_dS = np.empty_like(self.num_prop.x_f)
        self.intS_lda_gradT_n_dS = np.empty_like(self.num_prop.x_f)

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(markers=markers, phy_prop=self.phy_prop, x=self.num_prop.x)
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
        On en déduit ensuite les gradients de température aux faces, et les températures aux faces.

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
                im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(self.T), i)

                # On calcule gradTg, gradTi, Ti, gradTd

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(self, i, liqu_a_gauche=from_liqu_to_vap)
                cells = CellsInterface1eq(ldag, ldad, ag, dx, self.T_old[[im3, im2, im1, i0, ip1, ip2, ip3]],
                                          rhocpg=rhocpg, rhocpd=rhocpd)
                if self.interp_type == 'Ti':
                    cells.compute_from_Ti()
                elif self.interp_type == 'gradTi':
                    cells.compute_from_ldagradTi()
                else:
                    cells.compute_from_ldagradTi_ordre2()
                # On vérifie que l'interface ne traverse pas la face, sinon on met à jour rho_cp_face_droite qui est
                # un mélange au prorato du temps de passage entre la phase de droite et la phase de gauche.
                # Maintenant c'est fait dans la classe cells
                # if self.phy_prop.v > 0.:
                #     dt1_dt = max(self.I[i0] * self.num_prop.dx / (self.phy_prop.v * self.dt), 1.)
                # else:
                #     dt1_dt = 1.
                # dt2_dt = 1. - dt1_dt
                # cells.rhocp_f[2] = dt1_dt*rhocpd + dt2_dt*rhocpd

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi

                # Tgf = pid_interp(np.array([self.T_old[im1], Ti]), np.array([0.5, ag/2.])*self.num_prop.dx)
                # Tdf = pid_interp(np.array([Ti, self.T_old[ip1]]), np.array([ad/2., 0.5])*self.num_prop.dx)

                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                dV = self.num_prop.dx*self.phy_prop.dS
                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # interpolation upwind
                # DONE: l'écrire en version flux pour être sûr de la conservation
                # self.intS_rho_cp_T_u_n_dS =
                int_div_rhocpT_u = 1./dV * integrale_volume_div(np.ones((len(cells.gradT) - 1,)),
                                                                cells.rhocp_f*cells.T_f*self.phy_prop.v,
                                                                cl=1, dS=self.phy_prop.dS)
                int_div_lda_grad_T = 1./dV * integrale_volume_div(np.ones((len(cells.gradT) - 1,)),
                                                                  cells.lda_f*cells.gradT,
                                                                  dS=self.phy_prop.dS, cl=1)

                # Correction des cellules
                ind_to_change = [im2, im1, i0, ip1, ip2]
                self.T[ind_to_change] = (self.T_old[ind_to_change]*self.rho_cp_a[ind_to_change] +
                                         self.dt * (-int_div_rhocpT_u +
                                                    self.phy_prop.diff * int_div_lda_grad_T)) / rhocp_np1[ind_to_change]
        self.T_old = self.T.copy()

    def euler_timestep(self, debug=None, bool_debug=False):
        super().euler_timestep(debug=debug, bool_debug=bool_debug)
        self.corrige_interface_aymeric1()


def cl_perio(n, i):
    im1 = (i - 1) % n
    im2 = (i - 2) % n
    im3 = (i - 3) % n
    i0 = i % n
    ip1 = (i + 1) % n
    ip2 = (i + 2) % n
    ip3 = (i + 3) % n
    return im3, im2, im1, i0, ip1, ip2, ip3
