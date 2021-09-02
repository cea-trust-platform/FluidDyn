from src.main import *
from copy import deepcopy
from src.cells_interface import *


class BulleTemperature(Bulles):
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
        self.cells = np.zeros_like(self.markers, dtype=CellsSuiviInterface)
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
    bulles: BulleTemperature

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

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        self.h = self.rho_cp_a*self.T
        self.flux_conv_energie = np.zeros_like(self.flux_conv)

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
        self.flux_conv = interpolate(self.T, cl=1, schema='weno') * self.phy_prop.v
        self.flux_conv_energie = interpolate(self.rho_cp_a*self.T, cl=1, schema='weno') * self.phy_prop.v
        self.flux_diff = interpolate(self.Lda_h, cl=1, schema='weno') * grad(self.T, self.num_prop.dx)
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

                cells = CellsInterface(ldag, ldad, ag, self.num_prop.dx, self.T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                                       rhocpg=rhocpg, rhocpd=rhocpd, vdt=self.phy_prop.v*self.dt,
                                       interp_type='energie_temperature')
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
                self.flux_conv[ind_flux_to_change] = self.phy_prop.v * cells.T_f[1:-1]
                self.flux_conv_energie[ind_flux_to_change] = self.phy_prop.v * cells.rhocp_f[1:-1] * cells.T_f[1:-1]
                self.flux_diff[ind_flux_to_change] = cells.lda_f[1:-1] * cells.gradT[1:-1]

        dx = self.num_prop.dx
        int_div_T_u = integrale_vol_div(self.flux_conv, dx)
        int_div_rho_cp_T_u = integrale_vol_div(self.flux_conv_energie, dx)
        int_div_lda_grad_T = integrale_vol_div(self.flux_diff, dx)
        rho_cp_inv_h = 1. / self.rho_cp_h
        self.T += self.dt * (-int_div_T_u + self.phy_prop.diff * rho_cp_inv_h * int_div_lda_grad_T)
        self.h += self.dt * (-int_div_rho_cp_T_u + self.phy_prop.diff * int_div_lda_grad_T)

    @property
    def name(self):
        return 'Energie température ' + super().name


class ProblemDiscontinu(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

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

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        if self.num_prop.schema != 'upwind':
            raise Exception('Cette version ne marche que pour un schéma upwind')
        self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = 'Ti'
        else:
            self.interp_type = interp_type
        print(self.interp_type)

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

    def _corrige_interface_aymeric1(self):
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
                cells = CellsInterface(ldag, ldad, ag, dx, self.T_old[[im3, im2, im1, i0, ip1, ip2, ip3]],
                                       rhocpg=rhocpg, rhocpd=rhocpd, interp_type=self.interp_type)

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                int_div_rhocpT_u = integrale_vol_div(rhocpT_u, dx)
                lda_grad_T = cells.lda_f * cells.gradT
                int_div_lda_grad_T = integrale_vol_div(lda_grad_T, dx)

                # Correction des cellules
                ind_to_change = [im2, im1, i0, ip1, ip2]
                ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                self.flux_conv[ind_flux] = rhocpT_u
                self.flux_diff[ind_flux] = lda_grad_T
                self.T[ind_to_change] = (self.T_old[ind_to_change]*self.rho_cp_a[ind_to_change] +
                                         self.dt * (-int_div_rhocpT_u +
                                                    self.phy_prop.diff * int_div_lda_grad_T)) / rhocp_np1[ind_to_change]
        self.T_old = self.T.copy()

    @property
    def name(self):
        return 'CL energie ' + super().name

    def euler_timestep(self, debug=None, bool_debug=False):
        super().euler_timestep(debug=debug, bool_debug=bool_debug)
        self._corrige_interface_aymeric1()

    def rk4_timestep(self, debug=None, bool_debug=False):
        T_int = self.T.copy()
        K = [0.]
        T_u_l = []
        lda_gradT_l = []
        pas_de_temps = np.array([0, 0.5, 0.5, 1.])
        dx = self.num_prop.dx
        i0_tab = self.bulles.ind
        for h in pas_de_temps:
            markers_int = self.bulles.copy()
            markers_int.shift(self.phy_prop.v * h * self.dt)
            temp_I = markers_int.indicatrice_liquide(self.num_prop.x)
            T = T_int + h * self.dt * K[-1]
            T_u = interpolate(T, I=temp_I, schema=self.num_prop.schema) * self.phy_prop.v
            T_u_l.append(T_u)
            int_div_T_u = integrale_vol_div(T_u, self.num_prop.dx)

            Lda_h = 1. / (temp_I / self.phy_prop.lda1 + (1. - temp_I) / self.phy_prop.lda2)
            lda_grad_T = interpolate(Lda_h, I=temp_I, schema=self.num_prop.schema) * grad(T, self.num_prop.dx)
            lda_gradT_l.append(lda_grad_T)
            div_lda_grad_T = integrale_vol_div(lda_grad_T, self.num_prop.dx)

            rho_cp_f = interpolate(self.rho_cp_a, schema=self.num_prop.schema)
            int_div_rho_cp_u = integrale_vol_div(rho_cp_f, self.num_prop.dx)
            rho_cp_etoile = self.rho_cp_a - h * self.dt * int_div_rho_cp_u
            rho_cp_inv_h = temp_I / self.phy_prop.rho_cp1 + (1. - temp_I) / self.phy_prop.rho_cp2
            rho_cp_inv_int_div_lda_grad_T = self.phy_prop.diff * rho_cp_inv_h * div_lda_grad_T

            # correction de int_div_T_u et rho_cp_inv_int_div_lda_grad_T
            for i_int, (i1, i2) in enumerate(markers_int.ind):
                # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs
                # sur le maillage cartésien

                for ist, i in enumerate((i1, i2)):
                    if i == i1:
                        from_liqu_to_vap = True
                    else:
                        from_liqu_to_vap = False
                    im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                    # On calcule gradTg, gradTi, Ti, gradTd

                    ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(self, i, liqu_a_gauche=from_liqu_to_vap)
                    cells = CellsInterface(ldag, ldad, ag, dx, T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                                           rhocpg=rhocpg, rhocpd=rhocpd, interp_type=self.interp_type,
                                           vdt=h*self.dt*self.phy_prop.v)

                    # post-traitements
                    if h == 1.:
                        self.bulles.T[i_int, ist] = cells.Ti
                        self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                        self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                        self.bulles.Td[i_int, ist] = cells.Td[0]
                        self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                        self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                    # Correction des cellules i0 - 1 à i0 + 1 inclue
                    # DONE: l'écrire en version flux pour être sûr de la conservation
                    dx = self.num_prop.dx
                    cor_T_u = cells.T_f * self.phy_prop.v
                    cor_int_div_T_u = integrale_vol_div(cor_T_u, dx)
                    cor_lda_grad_T = cells.lda_f * cells.gradT
                    cor_int_div_lda_grad_T = integrale_vol_div(cor_lda_grad_T, dx)

                    # Correction des cellules
                    ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                    self.flux_conv[ind_flux] = cor_T_u
                    self.flux_diff[ind_flux] = cor_lda_grad_T

                    # TODO: changer pour être sûr de changer les bonnes cellules, cad les cellules qui correspondent et
                    #   pas celles qui ont suivi l'interface
                    rhocp_np1 = 0.
                    cor_int_div_rho_cp_u = 0.
                    int_div_lda_grad_T = 0.
                    int_div_rho_cp_T_u = 0.
                    km3, km2, km1, k0, kp1, kp2, kp3 = cl_perio(len(T), i0_tab[i_int, ist])
                    if i0 == k0:
                        ind_to_change = [km2, km1, k0, kp1, kp2]
                    else:
                        ind_to_change = [km1, k0, kp1, kp2, kp3]
                    cor_rho_cp_inv_h = rho_cp_inv_h[ind_to_change]

                    int_div_T_u[ind_to_change] = cor_int_div_T_u
                    rho_cp_inv_int_div_lda_grad_T[ind_to_change] = cor_int_div_lda_grad_T * cor_rho_cp_inv_h
                    int_div_rho_cp_u[ind_to_change] = cor_int_div_rho_cp_u
            # TODO: adapter cette formulation
            K.append(1/rhocp_np1 * (T * int_div_rho_cp_u - int_div_rho_cp_T_u + int_div_lda_grad_T))
        coeff = np.array([1. / 6, 1 / 3., 1 / 3., 1. / 6])
        self.flux_conv = np.sum(coeff * np.array(T_u_l).T, axis=-1)
        self.flux_diff = np.sum(coeff * np.array(lda_gradT_l).T, axis=-1)
        self.T += np.sum(self.dt * coeff * np.array(K[1:]).T, axis=-1)


class ProblemDiscontinu2(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

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

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        if self.num_prop.schema != 'upwind':
            raise Exception('Cette version ne marche que pour un schéma upwind')
        self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = 'Ti'
        else:
            self.interp_type = interp_type
        print(self.interp_type)

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

    def _corrige_interface_aymeric1(self):
        """
        Dans cette approche on calclue Ti et lda_gradTi soit en utilisant la continuité avec Tim1 et Tip1, soit en
        utilisant la continuité des lda_grad_T calculés avec Tim2, Tim1, Tip1 et Tip2.
        Dans les deux cas il est à noter que l'on utilise pas les valeurs présentes dans la cellule de l'interface.
        On en déduit ensuite les gradients de température aux faces, et les températures aux faces.

        Returns:
            Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
            jour T_old
        """
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
                cells = CellsInterface(ldag, ldad, ag, dx, self.T_old[[im3, im2, im1, i0, ip1, ip2, ip3]],
                                       rhocpg=rhocpg, rhocpd=rhocpd, interp_type=self.interp_type)

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                T_u = cells.T_f * self.phy_prop.v
                int_div_T_u = integrale_vol_div(T_u, dx)
                lda_grad_T = cells.lda_f * cells.gradT
                int_div_lda_grad_T = integrale_vol_div(lda_grad_T, dx)

                # Correction des cellules
                ind_to_change = [im2, im1, i0, ip1, ip2]
                ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                self.flux_conv[ind_flux] = T_u
                self.flux_diff[ind_flux] = lda_grad_T
                # on écrit l'équation en température, ça me semble peut être mieux ?
                # ce n'est pas conservatif en énergie mais ce n'est peut être pas très grave
                # Tnp1 = Tn + dt (- int_S_T_u + 1/rhocp * int_S_lda_grad_T)
                self.T[ind_to_change] = self.T_old[ind_to_change] + \
                    self.dt * (-int_div_T_u +
                               self.phy_prop.diff * int_div_lda_grad_T / self.rho_cp_a[ind_to_change])
        self.T_old = self.T.copy()

    def euler_timestep(self, debug=None, bool_debug=False):
        super().euler_timestep(debug=debug, bool_debug=bool_debug)
        self._corrige_interface_aymeric1()

    @property
    def name(self):
        return 'CL température ' + super().name

    def rk4_timestep(self, debug=None, bool_debug=False):
        T_int = self.T.copy()
        K = [0.]
        T_u_l = []
        lda_gradT_l = []
        pas_de_temps = np.array([0, 0.5, 0.5, 1.])
        dx = self.num_prop.dx
        i0_tab = self.bulles.ind
        for h in pas_de_temps:
            markers_int = self.bulles.copy()
            markers_int.shift(self.phy_prop.v * h * self.dt)
            temp_I = markers_int.indicatrice_liquide(self.num_prop.x)
            T = T_int + h * self.dt * K[-1]
            T_u = interpolate(T, I=temp_I, schema=self.num_prop.schema) * self.phy_prop.v
            T_u_l.append(T_u)
            int_div_T_u = integrale_vol_div(T_u, self.num_prop.dx)

            Lda_h = 1. / (temp_I / self.phy_prop.lda1 + (1. - temp_I) / self.phy_prop.lda2)
            lda_grad_T = interpolate(Lda_h, I=temp_I, schema=self.num_prop.schema) * grad(T, self.num_prop.dx)
            lda_gradT_l.append(lda_grad_T)
            div_lda_grad_T = integrale_vol_div(lda_grad_T, self.num_prop.dx)

            rho_cp_inv_h = temp_I / self.phy_prop.rho_cp1 + (1. - temp_I) / self.phy_prop.rho_cp2
            rho_cp_inv_int_div_lda_grad_T = self.phy_prop.diff * rho_cp_inv_h * div_lda_grad_T

            # correction de int_div_T_u et rho_cp_inv_int_div_lda_grad_T
            for i_int, (i1, i2) in enumerate(markers_int.ind):
                # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs
                # sur le maillage cartésien

                for ist, i in enumerate((i1, i2)):
                    if i == i1:
                        from_liqu_to_vap = True
                    else:
                        from_liqu_to_vap = False
                    im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)

                    # On calcule gradTg, gradTi, Ti, gradTd

                    ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(self, i, liqu_a_gauche=from_liqu_to_vap)
                    cells = CellsInterface(ldag, ldad, ag, dx, T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                                           rhocpg=rhocpg, rhocpd=rhocpd, interp_type=self.interp_type,
                                           vdt=h*self.dt*self.phy_prop.v)

                    # post-traitements
                    if h == 1.:
                        self.bulles.T[i_int, ist] = cells.Ti
                        self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                        self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                        self.bulles.Td[i_int, ist] = cells.Td[0]
                        self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                        self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                    # Correction des cellules i0 - 1 à i0 + 1 inclue
                    # DONE: l'écrire en version flux pour être sûr de la conservation
                    dx = self.num_prop.dx
                    cor_T_u = cells.T_f * self.phy_prop.v
                    cor_int_div_T_u = integrale_vol_div(cor_T_u, dx)
                    cor_lda_grad_T = cells.lda_f * cells.gradT
                    cor_int_div_lda_grad_T = integrale_vol_div(cor_lda_grad_T, dx)

                    # Correction des cellules
                    ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                    self.flux_conv[ind_flux] = cor_T_u
                    self.flux_diff[ind_flux] = cor_lda_grad_T

                    # TODO: changer pour être sûr de changer les bonnes cellules, cad les cellules qui correspondent et
                    #   pas celles qui ont suivi l'interface
                    km3, km2, km1, k0, kp1, kp2, kp3 = cl_perio(len(T), i0_tab[i_int, ist])
                    if i0 == k0:
                        ind_to_change = [km2, km1, k0, kp1, kp2]
                    else:
                        ind_to_change = [km1, k0, kp1, kp2, kp3]
                    cor_rho_cp_inv_h = rho_cp_inv_h[ind_to_change]

                    int_div_T_u[ind_to_change] = cor_int_div_T_u
                    rho_cp_inv_int_div_lda_grad_T[ind_to_change] = cor_int_div_lda_grad_T * cor_rho_cp_inv_h
            K.append(-int_div_T_u + rho_cp_inv_int_div_lda_grad_T)
        coeff = np.array([1. / 6, 1 / 3., 1 / 3., 1. / 6])
        self.flux_conv = np.sum(coeff * np.array(T_u_l).T, axis=-1)
        self.flux_diff = np.sum(coeff * np.array(lda_gradT_l).T, axis=-1)
        self.T += np.sum(self.dt * coeff * np.array(K[1:]).T, axis=-1)


class ProblemDiscontinuSautdTdt(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

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

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None, deb=False, delta_diff=1.,
                 delta_conv=1., int_Ti=1.):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        self.deb = deb
        if self.num_prop.schema != 'upwind':
            raise Exception('Cette version ne marche que pour un schéma upwind')
        self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = 'Ti'
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        self.delta_diff = delta_diff
        self.delta_conv = delta_conv
        self.int_Ti = int_Ti

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

    def _corrige_interface_aymeric1(self):
        """
        Dans cette approche on calclue Ti et lda_gradTi soit en utilisant la continuité avec Tim1 et Tip1, soit en
        utilisant la continuité des lda_grad_T calculés avec Tim2, Tim1, Tip1 et Tip2.
        Dans les deux cas il est à noter que l'on utilise pas les valeurs présentes dans la cellule de l'interface.
        On en déduit ensuite les gradients de température aux faces, et les températures aux faces.

        Returns:
            Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
            jour T_old
        """
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
                cells = CellsInterface(ldag, ldad, ag, dx, self.T_old[[im3, im2, im1, i0, ip1, ip2, ip3]],
                                       rhocpg=rhocpg, rhocpd=rhocpd, interp_type=self.interp_type)

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                rhocp_T_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                int_div_rhocp_T_u = integrale_vol_div(rhocp_T_u, dx)
                lda_grad_T = cells.lda_f * cells.gradT
                int_div_lda_grad_T = integrale_vol_div(lda_grad_T, dx)

                # propre à cette version particulière, on calule le saut de dT/dt à l'interface et int_S_Ti_v_n2_dS
                delta0 = self.delta_diff * (cells.grad_lda_gradT_n_d/rhocpd - cells.grad_lda_gradT_n_g/rhocpg) \
                    - self.delta_conv * cells.lda_gradTi * (1/ldad - 1/ldag) * self.phy_prop.v

                # pour rappel, ici on a divisé l'intégrale par le volume de la cellule comme toutes les intégrales
                # le signe - vient du fait qu'on calcule pour V2, avec le vecteur normal à I qui est donc dirigé en -x
                int_S_Ti_v_n2_dS_0 = -self.int_Ti * cells.Ti * self.phy_prop.v / self.num_prop.dx

                delta = np.array([0., 0., delta0, 0., 0.])
                int_S_Ti_v_n2_dS = np.array([0., 0., int_S_Ti_v_n2_dS_0, 0., 0.])

                # Correction des cellules
                ind_to_change = [im2, im1, i0, ip1, ip2]
                ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                self.flux_conv[ind_flux] = rhocp_T_u / self.rho_cp_a[ind_flux]
                self.flux_diff[ind_flux] = lda_grad_T
                if self.deb:
                    print('delta conv : ', cells.lda_gradTi * (1/ldad - 1/ldag) * self.phy_prop.v)
                    print('delta cond : ', (cells.grad_lda_gradT_n_d/rhocpd - cells.grad_lda_gradT_n_g/rhocpg))
                    print('delta * ... : ', delta0 * ad * (rhocpd - self.rho_cp_a[i0]))
                    print('int_I... : ', (rhocpd - rhocpg) * int_S_Ti_v_n2_dS_0)
                    print('int_I... + delta * ... : ', (rhocpd - rhocpg) * int_S_Ti_v_n2_dS_0 +
                          delta0 * ad * (rhocpd - self.rho_cp_a[i0]))
                    print('(int_I... + delta * ...)/rho_cp_a : ', ((rhocpd - rhocpg) * int_S_Ti_v_n2_dS_0 +
                                                                   delta0 * ad * (rhocpd - self.rho_cp_a[i0]))/self.rho_cp_a[i0])
                    print('int_div_lda_grad_T/rho_cp_a : ', int_div_lda_grad_T[2]/self.rho_cp_a[i0])
                    print('int_div_rhocp_T_u/rho_cp_a : ', int_div_rhocp_T_u[2]/self.rho_cp_a[i0])

                # on écrit l'équation en température, ça me semble peut être mieux ?
                # Tnp1 = Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T
                #                 - delta * I2 * (rhocp2 - rhocpa) - [rhocp] * int_S_Ti_v_n2_dS) / rhocpa
                self.T[ind_to_change] = self.T_old[ind_to_change] + \
                    self.dt * (-int_div_rhocp_T_u + self.phy_prop.diff * int_div_lda_grad_T
                               - delta * ad * (rhocpd - self.rho_cp_a[i0])
                               - (rhocpd - rhocpg) * int_S_Ti_v_n2_dS) / self.rho_cp_a[ind_to_change]
        self.T_old = self.T.copy()

    def euler_timestep(self, debug=None, bool_debug=False):
        super().euler_timestep(debug=debug, bool_debug=bool_debug)
        self._corrige_interface_aymeric1()

    @property
    def name(self):
        return 'CL température saut dTdt ' + super().name


class ProblemDiscontinuCoupleConserv(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

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

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        if self.num_prop.schema != 'upwind':
            raise Exception('Cette version ne marche que pour un schéma upwind')
        self.T_old = self.T.copy()
        self.flux_conv_ener = self.flux_conv.copy()
        self.h = self.rho_cp_a * self.T
        self.h_old = self.h.copy()
        if interp_type is None:
            self.interp_type = 'energie_temperature'
        else:
            self.interp_type = interp_type
        print(self.interp_type)

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

    def _corrige_interface(self):
        """
        Dans cette approche on calclue Ti et lda_gradTi soit en utilisant la continuité avec Tim1 et Tip1, soit en
        utilisant la continuité des lda_grad_T calculés avec Tim2, Tim1, Tip1 et Tip2.
        Dans les deux cas il est à noter que l'on utilise pas les valeurs présentes dans la cellule de l'interface.
        On en déduit ensuite les gradients de température aux faces, et les températures aux faces.

        Returns:
            Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
            jour T_old
        """
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
                cells = CellsInterface(ldag, ldad, ag, dx, self.T_old[[im3, im2, im1, i0, ip1, ip2, ip3]],
                                       rhocpg=rhocpg, rhocpd=rhocpd, interp_type='energie_temperature')
                cells.compute_from_h_T(self.h_old[i0], self.T_old[i0])

                # post-traitements

                self.bulles.T[i_int, ist] = cells.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells.Tg[-1]
                self.bulles.Td[i_int, ist] = cells.Td[0]
                self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
                self.bulles.gradTd[i_int, ist] = cells.gradTd[0]

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                rhocp_T_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                int_div_rhocp_T_u = integrale_vol_div(rhocp_T_u, dx)
                lda_grad_T = cells.lda_f * cells.gradT
                int_div_lda_grad_T = integrale_vol_div(lda_grad_T, dx)

                # propre à cette version particulière, on calule le saut de dT/dt à l'interface et int_S_Ti_v_n2_dS
                delta0 = (cells.grad_lda_gradT_n_d/rhocpd - cells.grad_lda_gradT_n_g/rhocpg) \
                         - cells.lda_gradTi * (1/ldad - 1/ldag) * self.phy_prop.v

                # pour rappel, ici on a divisé l'intégrale par le volume de la cellule comme toutes les intégrales
                # le signe - vient du fait qu'on calcule pour V2, avec le vecteur normal à I qui est donc dirigé en -x
                int_S_Ti_v_n2_dS_0 = -cells.Ti * self.phy_prop.v / self.num_prop.dx

                delta = np.array([0., 0., delta0, 0., 0.])
                int_S_Ti_v_n2_dS = np.array([0., 0., int_S_Ti_v_n2_dS_0, 0., 0.])

                # Correction des cellules
                ind_to_change = [im2, im1, i0, ip1, ip2]
                ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                self.flux_conv[ind_flux] = cells.T_f * self.phy_prop.v
                self.flux_conv_ener[ind_flux] = rhocp_T_u
                self.flux_diff[ind_flux] = lda_grad_T
                # on écrit l'équation en température, et en energie
                # Tnp1 = Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T
                #                 - delta * I2 * (rhocp2 - rhocpa) - [rhocp] * int_S_Ti_v_n2_dS) / rhocpa
                self.T[ind_to_change] = self.T_old[ind_to_change] + \
                    self.dt * (-int_div_rhocp_T_u + self.phy_prop.diff * int_div_lda_grad_T
                               - delta * ad * (rhocpd - self.rho_cp_a[i0])
                               - (rhocpd - rhocpg) * int_S_Ti_v_n2_dS) / self.rho_cp_a[ind_to_change]
                self.h[ind_to_change] = self.h_old[ind_to_change] + \
                    self.dt * (-int_div_rhocp_T_u +
                               self.phy_prop.diff * int_div_lda_grad_T)
        self.T_old = self.T.copy()
        self.h_old = self.h.copy()

    def euler_timestep(self, debug=None, bool_debug=False):
        self.flux_conv = interpolate(self.T, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        self.flux_conv_ener = interpolate(self.h, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        int_div_T_u = integrale_vol_div(self.flux_conv, self.num_prop.dx)
        int_div_rho_cp_T_u = integrale_vol_div(self.flux_conv_ener, self.num_prop.dx)
        self.flux_diff = interpolate(self.Lda_h, I=self.I, schema=self.num_prop.schema) * grad(self.T, self.num_prop.dx)
        int_div_lda_grad_T = integrale_vol_div(self.flux_diff, self.num_prop.dx)

        if (debug is not None) and bool_debug:
            debug.plot(self.num_prop.x, 1. / self.rho_cp_h, label='rho_cp_inv_h, time = %f' % self.time)
            debug.plot(self.num_prop.x, int_div_lda_grad_T, label='div_lda_grad_T, time = %f' % self.time)
            debug.xticks(self.num_prop.x_f)
            debug.grid(which='major')
            maxi = max(np.max(int_div_lda_grad_T), np.max(1. / self.rho_cp_h))
            mini = min(np.min(int_div_lda_grad_T), np.min(1. / self.rho_cp_h))
            for markers in self.bulles():
                debug.plot([markers[0]] * 2, [mini, maxi], '--')
                debug.plot([markers[1]] * 2, [mini, maxi], '--')
            debug.legend()
        rho_cp_inv_h = 1. / self.rho_cp_h
        self.T += self.dt * (-int_div_T_u + self.phy_prop.diff * rho_cp_inv_h * int_div_lda_grad_T)
        self.h += self.dt * (-int_div_rho_cp_T_u + self.phy_prop.diff * int_div_lda_grad_T)

        self._corrige_interface()

    @property
    def name(self):
        return 'CL température saut dTdt ' + super().name

def cl_perio(n, i):
    im1 = (i - 1) % n
    im2 = (i - 2) % n
    im3 = (i - 3) % n
    i0 = i % n
    ip1 = (i + 1) % n
    ip2 = (i + 2) % n
    ip3 = (i + 3) % n
    return im3, im2, im1, i0, ip1, ip2, ip3


class ProblemDiscontinuFT(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

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

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        if self.num_prop.schema != 'upwind':
            raise Exception('Cette version ne marche que pour un schéma upwind')
        self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = 'gradTi'
            print('interp type is :', self.interp_type)
        else:
            self.interp_type = interp_type

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

    def _corrige_interface_ft(self):
        """
        Dans cette correction, on calcule l'évolution de la température dans des cellules qui suivent l'interface (donc
        sans convection). Ensuite on réinterpole sur la grille fixe des température.

        Returns:
            Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
            jour T_old
        """
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
                cells_ft = CellsSuiviInterface(ldag, ldad, ag, dx, self.T_old[[im3, im2, im1, i0, ip1, ip2, ip3]],
                                               rhocpg=rhocpg, rhocpd=rhocpd, vdt=self.dt*self.phy_prop.v,
                                               interp_type=self.interp_type)
                # On commence par interpoler Ti sur Tj avec TI et lda_gradTi
                # On calcule notre pas de temps avec lda_gradTj entre j et jp1 (à l'interface)
                # On interpole Tj sur la grille i

                # Correction des cellules
                ind_to_change = [im1, i0, ip1]
                ind_flux = [im1, i0, ip1, ip2]
                self.flux_conv[ind_flux] = np.nan
                self.flux_diff[ind_flux] = np.nan

                cells_ft.timestep(self.dt, self.phy_prop.diff)
                T_i_np1_interp = cells_ft.interp_T_from_j_to_i()
                self.T[ind_to_change] = T_i_np1_interp

                # post-traitements

                self.bulles.T[i_int, ist] = cells_ft.cells_fixe.Ti
                self.bulles.lda_grad_T[i_int, ist] = cells_ft.cells_fixe.lda_gradTi
                self.bulles.Tg[i_int, ist] = cells_ft.cells_fixe.Tg[-1]
                self.bulles.Td[i_int, ist] = cells_ft.cells_fixe.Td[0]
                self.bulles.cells[i_int, ist] = cells_ft
                # print('cells :', self.bulles.cells[i_int, ist])

        self.T_old = self.T.copy()

    def euler_timestep(self, debug=None, bool_debug=False):
        super().euler_timestep(debug=debug, bool_debug=bool_debug)
        self._corrige_interface_ft()

    @property
    def name(self):
        return 'FT ' + super().name
