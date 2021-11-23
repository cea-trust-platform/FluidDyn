from src.main import *
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
        self.Ti = np.zeros_like(self.markers)
        self.cells = [0.]*(2*len(self.markers))  # type: list
        self.ind = None
        if x is not None:
            self.x = x
            self._set_indices_markers(x)
        else:
            raise Exception('x est un argument obligatoire')

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

    def copy(self):
        cls = self.__class__
        copie = cls(markers=self.markers.copy(), Delta=self.Delta, x=self.x.copy())
        copie.diam = self.diam
        return copie

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

    # def _euler_timestep(self, debug=None, bool_debug=False):
    #     # on devrait plutôt calculer les flux, les stocker
    #     self.flux_conv = interpolate(self.T, cl=1, schema='weno') * self.phy_prop.v
    #     self.flux_conv_energie = interpolate(self.rho_cp_a*self.T, cl=1, schema='weno') * self.phy_prop.v
    #     self.flux_diff = interpolate(self.Lda_h, cl=1, schema='weno') * grad(self.T, self.num_prop.dx)
    #     # Est ce qu'on fait entièrement en monofluide pour la température moyenne ou est-ce qu'on s'amuse à faire les
    #     # choses bien comme il faut au moins pour la diffusion ? (Avec des ldas*gradT purement monophasiques)
    #     # Faisons les choses bien
    #     for i_int, (i1, i2) in enumerate(self.bulles.ind):
    #         # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
    #         # le maillage cartésien
    #
    #         for ist, i in enumerate((i1, i2)):
    #             if i == i1:
    #                 from_liqu_to_vap = True
    #             else:
    #                 from_liqu_to_vap = False
    #             im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(self.T), i)
    #             ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(self, i, liqu_a_gauche=from_liqu_to_vap)
    #
    #             cells = CellsInterface(ldag, ldad, ag, self.num_prop.dx, self.T[[im3, im2, im1, i0, ip1, ip2, ip3]],
    #                                    rhocpg=rhocpg, rhocpd=rhocpd, vdt=self.phy_prop.v*self.dt,
    #                                    interp_type='energie_temperature', schema_conv=self.num_prop.schema)
    #             cells.compute_from_h_T(self.h[i0], self.T[i0])
    #
    #             # post-traitements
    #
    #             self.bulles.T[i_int, ist] = cells.Ti
    #             self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
    #
    #             self.bulles.Tg[i_int, ist] = cells.Tg[-1]
    #             self.bulles.Td[i_int, ist] = cells.Td[0]
    #             self.bulles.gradTg[i_int, ist] = cells.gradT[3]
    #             self.bulles.gradTd[i_int, ist] = cells.gradT[4]
    #
    #             # Correction des flux entrant et sortant de la maille diphasique
    #             ind_flux_to_change = [im1, i0, ip1, ip2]
    #             self.flux_conv[ind_flux_to_change] = self.phy_prop.v * cells.T_f[1:-1]
    #             self.flux_conv_energie[ind_flux_to_change] = self.phy_prop.v * cells.rhocp_f[1:-1] * cells.T_f[1:-1]
    #             self.flux_diff[ind_flux_to_change] = cells.lda_f[1:-1] * cells.gradT[1:-1]
    #
    #     dx = self.num_prop.dx
    #     int_div_T_u = integrale_vol_div(self.flux_conv, dx)
    #     int_div_rho_cp_T_u = integrale_vol_div(self.flux_conv_energie, dx)
    #     int_div_lda_grad_T = integrale_vol_div(self.flux_diff, dx)
    #     rho_cp_inv_h = 1. / self.rho_cp_h
    #     self.T += self.dt * (-int_div_T_u + self.phy_prop.diff * rho_cp_inv_h * int_div_lda_grad_T)
    #     self.h += self.dt * (-int_div_rho_cp_T_u + self.phy_prop.diff * int_div_lda_grad_T)

    def _corrige_interface(self):
        """
        Dans cette approche on calclue Ti et lda_gradTi à partir du système énergie température

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
                cells = CellsInterface(ldag, ldad, ag, dx, self.T[[im3, im2, im1, i0, ip1, ip2, ip3]],
                                       rhocpg=rhocpg, rhocpd=rhocpd, interp_type='energie_temperature',
                                       schema_conv='quick', vdt=self.phy_prop.v*self.dt)
                cells.compute_from_h_T(self.h[i0], self.T[i0])
                cells.compute_T_f_gradT_f_quick()
                # print(cells.rhocp_f)

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
                # int_div_rhocp_T_u = integrale_vol_div(rhocp_T_u, dx)
                lda_grad_T = cells.lda_f * cells.gradT
                # int_div_lda_grad_T = integrale_vol_div(lda_grad_T, dx)

                # Correction des cellules
                # ind_to_change = [im2, im1, i0, ip1, ip2]
                # ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                ind_flux = [im1, i0, ip1, ip2, ip3]
                self.flux_conv[ind_flux] = cells.T_f[1:] * self.phy_prop.v
                self.flux_conv_ener[ind_flux] = rhocp_T_u[1:]
                self.flux_diff[ind_flux] = lda_grad_T[1:]
                # on écrit l'équation en température, et en energie
                # Tnp1 = Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T
                #                 - delta * I2 * (rhocp2 - rhocpa) - [rhocp] * int_S_Ti_v_n2_dS) / rhocpa

    def _euler_timestep(self, debug=None, bool_debug=False):
        self.flux_conv = interpolate(self.T, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        self.flux_conv_ener = interpolate(self.h, I=self.I, schema=self.num_prop.schema) * self.phy_prop.v
        self.flux_diff = interpolate(self.Lda_h, I=self.I, schema=self.num_prop.schema) * grad(self.T, self.num_prop.dx)
        self._corrige_interface()
        int_div_T_u = integrale_vol_div(self.flux_conv, self.num_prop.dx)
        int_div_rho_cp_T_u = integrale_vol_div(self.flux_conv_ener, self.num_prop.dx)
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


    @property
    def name(self):
        return 'Energie température ' + super().name


class ProblemDiscontinuE(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Résolution en énergie.
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

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None, conv_interf=None):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        # if self.num_prop.schema != 'upwind':
        #     raise Exception('Cette version ne marche que pour un schéma upwind')
        if num_prop.time_scheme == 'rk3':
            print('RK3 is not implemented, changes to Euler')
            self.num_prop._time_scheme = 'euler'
        self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = 'Ti'
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf

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

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4
        Attention, lorsque cette méthode est surclassée et que les arguments changent il faut aussi surclasser
        _euler, _rk3 et _rk4_timestep

        Args:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

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
                                       schema_conv=self.conv_interf, vdt=self.dt*self.phy_prop.v)
                self.bulles.cells[2*i_int + ist] = cells

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation

                rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
                lda_grad_T = cells.lda_f * cells.gradT
                # self.bulles.gradTg[i_int, ist] = cells.gradT[3]
                # self.bulles.gradTd[i_int, ist] = cells.gradT[4]
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Ti[i_int, ist] = cells.Ti

                # print('rhocpTu : ', rhocpT_u)
                # print('lda_graT : ', lda_grad_T)
                # Correction des flux cellules
                # ind_to_change = [im2, im1, i0, ip1, ip2]
                # ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                ind_flux = [im1, i0, ip1, ip2, ip3]
                flux_conv[ind_flux] = rhocpT_u[1:]
                flux_diff[ind_flux] = lda_grad_T[1:]
                # Correction des coeffs
                # Pas de coeff à corriger
                # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v * self.dt)
        I_np1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_a_np1 = I_np1 * self.phy_prop.rho_cp1 + (1.-I_np1) * self.phy_prop.rho_cp2
        self.flux_conv = self._compute_convection_flux(self.rho_cp_a * self.T, self.bulles, bool_debug, debug)
        self.flux_diff = self._compute_diffusion_flux(self.T, self.bulles, bool_debug, debug)
        self._corrige_flux_coeff_interface(self.T, self.bulles, self.flux_conv, self.flux_diff)
        drhocpTdt = - integrale_vol_div(self.flux_conv, dx) \
            + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        # rho_cp_np1 * Tnp1 = rho_cp_n * Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T)
        self.T = (self.T * self.rho_cp_a + self.dt * drhocpTdt) / rho_cp_a_np1

    # # TODO: finir cette méthode, attention il faut trouver une solution pour mettre à jour T de manière cohérente
    # def _rk3_timestep(self, debug=None, bool_debug=False):
    #     T_int = self.T.copy()
    #     markers_int = self.bulles.copy()
    #     K = 0.
    #     # pas_de_temps = np.array([0, 1/3., 3./4])
    #     coeff_h = np.array([1./3, 5./12, 1./4])
    #     coeff_dTdtm1 = np.array([0., -5./9, -153./128])
    #     coeff_dTdt = np.array([1., 4./9, 15./32])
    #     for step, h in enumerate(coeff_h):
    #         I_step = markers_int.indicatrice_liquide(self.num_prop.x)
    #         rho_cp_a_step = I_step * self.phy_prop.rho_cp1 + (1. - I_step) * self.phy_prop.rho_cp2
    #         # convection, conduction, dTdt = self.compute_dT_dt(T_int, markers_int, bool_debug, debug)
    #         convection = self._compute_convection_flux(T_int, markers_int, bool_debug, debug)
    #         conduction = self._compute_diffusion_flux(T_int, markers_int, bool_debug, debug)
    #         self._corrige_flux_coeff_interface(T_int, markers_int, convection, conduction)
    #         markers_int_np1 = markers_int.copy()
    #         markers_int_np1.shift(self.phy_prop.v * h * self.dt)
    #         I_step_p1 = markers_int_np1.indicatrice_liquide(self.num_prop.x)
    #         rho_cp_a_step_p1 = I_step_p1 * self.phy_prop.rho_cp1 + (1. - I_step_p1) * self.phy_prop.rho_cp2
    #         drhocpTdt = - integrale_vol_div(convection, self.num_prop.dx) \
    #             + self.phy_prop.diff * integrale_vol_div(conduction, self.num_prop.dx)
    #         # On a dT = (- (rhocp_np1 - rhocp_n) * Tn + dt * (-conv + diff)) / rhocp_np1
    #         dTdt = (- (rho_cp_a_step_p1 - rho_cp_a_step) / (h * self.dt) * T_int + drhocpTdt) / rho_cp_a_step_p1
    #         K = K * coeff_dTdtm1[step] + dTdt
    #         if bool_debug and (debug is not None):
    #             print('step : ', step)
    #             print('dTdt : ', dTdt)
    #             print('K    : ', K)
    #         T_int += h * self.dt * K / coeff_dTdt[step]  # coeff_dTdt est calculé de
    #         # sorte à ce que le coefficient total devant les dérviées vale 1.
    #         markers_int.shift(self.phy_prop.v * h * self.dt)
    #     self.T = T_int
    #
    # def _rk4_timestep(self, debug=None, bool_debug=False):
    #     # T_int = self.T.copy()
    #     K = [0.]
    #     T_u_l = []
    #     lda_gradT_l = []
    #     pas_de_temps = np.array([0., 0.5, 0.5, 1.])
    #     dx = self.num_prop.dx
    #     for h in pas_de_temps:
    #         markers_int = self.bulles.copy()
    #         markers_int.shift(self.phy_prop.v * h * self.dt)
    #         I_step = markers_int.indicatrice_liquide(self.num_prop.x)
    #         rho_cp_a_step = I_step * self.phy_prop.rho_cp1 + (1. - I_step) * self.phy_prop.rho_cp2
    #         T = self.T + h * self.dt * K[-1]
    #         convection = self._compute_convection_flux(T, markers_int, bool_debug, debug)
    #         conduction = self._compute_diffusion_flux(T, markers_int, bool_debug, debug)
    #         self._corrige_flux_coeff_interface(T, markers_int, convection, conduction)
    #         T_u_l.append(convection)
    #         lda_gradT_l.append(conduction)
    #         # On a dT = (- (rhocp_np1 - rhocp_n) * Tn + dt * (-conv + diff)) / rhocp_np1
    #         # Probleme pour h = 0., on ne peut pas calculer drhocp/dt par différence de temps
    #         raise NotImplementedError
    #         K.append(- integrale_vol_div(convection, dx)
    #                  + self.phy_prop.diff * coeff_conduction * integrale_vol_div(conduction, dx))
    #     coeff = np.array([1. / 6, 1 / 3., 1 / 3., 1. / 6])
    #     self.flux_conv = np.sum(coeff * np.array(T_u_l).T, axis=-1)
    #     self.flux_diff = np.sum(coeff * np.array(lda_gradT_l).T, axis=-1)
    #     self.T += np.sum(self.dt * coeff * np.array(K[1:]).T, axis=-1)

    # def _corrige_interface_aymeric1(self):
    #     """
    #     Dans cette approche on calclue Ti et lda_gradTi soit en utilisant la continuité avec Tim1 et Tip1, soit en
    #     utilisant la continuité des lda_grad_T calculés avec Tim2, Tim1, Tip1 et Tip2.
    #     Dans les deux cas il est à noter que l'on utilise pas les valeurs présentes dans la cellule de l'interface.
    #     On en déduit ensuite les gradients de température aux faces, et les températures aux faces.
    #
    #     Returns:
    #         Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
    #         jour T_old
    #     """
    #     bulles_np1 = self.bulles.copy()
    #     bulles_np1.shift(self.phy_prop.v*self.dt)
    #     Inp1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
    #     rhocp_np1 = self.phy_prop.rho_cp1 * Inp1 + self.phy_prop.rho_cp2 * (1.-Inp1)
    #     dx = self.num_prop.dx
    #
    #     for i_int, (i1, i2) in enumerate(self.bulles.ind):
    #         # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
    #         # le maillage cartésien
    #
    #         for ist, i in enumerate((i1, i2)):
    #             if i == i1:
    #                 from_liqu_to_vap = True
    #             else:
    #                 from_liqu_to_vap = False
    #             im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(self.T), i)
    #
    #             # On calcule gradTg, gradTi, Ti, gradTd
    #
    #             ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(self, i, liqu_a_gauche=from_liqu_to_vap)
    #             cells = CellsInterface(ldag, ldad, ag, dx, self.T_old[[im3, im2, im1, i0, ip1, ip2, ip3]],
    #                                    rhocpg=rhocpg, rhocpd=rhocpd, interp_type=self.interp_type)
    #
    #             # post-traitements
    #
    #             self.bulles.T[i_int, ist] = cells.Ti
    #             self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
    #             self.bulles.Tg[i_int, ist] = cells.Tg[-1]
    #             self.bulles.Td[i_int, ist] = cells.Td[0]
    #             self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
    #             self.bulles.gradTd[i_int, ist] = cells.gradTd[0]
    #
    #             # Correction des cellules i0 - 1 à i0 + 1 inclue
    #             # DONE: l'écrire en version flux pour être sûr de la conservation
    #             dx = self.num_prop.dx
    #             rhocpT_u = cells.rhocp_f * cells.T_f * self.phy_prop.v
    #             int_div_rhocpT_u = integrale_vol_div(rhocpT_u, dx)
    #             lda_grad_T = cells.lda_f * cells.gradT
    #             int_div_lda_grad_T = integrale_vol_div(lda_grad_T, dx)
    #
    #             # Correction des cellules
    #             ind_to_change = [im2, im1, i0, ip1, ip2]
    #             ind_flux = [im2, im1, i0, ip1, ip2, ip3]
    #             self.flux_conv[ind_flux] = rhocpT_u
    #             self.flux_diff[ind_flux] = lda_grad_T
    #             self.T[ind_to_change] = (self.T_old[ind_to_change]*self.rho_cp_a[ind_to_change] +
    #                                      self.dt * (-int_div_rhocpT_u +
    #                                                 self.phy_prop.diff * int_div_lda_grad_T)) / rhocp_np1[ind_to_change]
    #     self.T_old = self.T.copy()

    @property
    def name(self):
        return 'EFC, ' + super().name + ', ' + self.interp_type.replace('_', '-')

    # def _rk4_timestep(self, debug=None, bool_debug=False):
    #     T_int = self.T.copy()
    #     K = [0.]
    #     T_u_l = []
    #     lda_gradT_l = []
    #     pas_de_temps = np.array([0, 0.5, 0.5, 1.])
    #     dx = self.num_prop.dx
    #     i0_tab = self.bulles.ind
    #     for h in pas_de_temps:
    #         markers_int = self.bulles.copy()
    #         markers_int.shift(self.phy_prop.v * h * self.dt)
    #         temp_I = markers_int.indicatrice_liquide(self.num_prop.x)
    #         T = T_int + h * self.dt * K[-1]
    #         T_u = interpolate(T, I=temp_I, schema=self.num_prop.schema) * self.phy_prop.v
    #         T_u_l.append(T_u)
    #         int_div_T_u = integrale_vol_div(T_u, self.num_prop.dx)
    #
    #         Lda_h = 1. / (temp_I / self.phy_prop.lda1 + (1. - temp_I) / self.phy_prop.lda2)
    #         lda_grad_T = interpolate(Lda_h, I=temp_I, schema=self.num_prop.schema) * grad(T, self.num_prop.dx)
    #         lda_gradT_l.append(lda_grad_T)
    #         div_lda_grad_T = integrale_vol_div(lda_grad_T, self.num_prop.dx)
    #
    #         rho_cp_f = interpolate(self.rho_cp_a, schema=self.num_prop.schema)
    #         int_div_rho_cp_u = integrale_vol_div(rho_cp_f, self.num_prop.dx)
    #         # rho_cp_etoile = self.rho_cp_a - h * self.dt * int_div_rho_cp_u
    #         rho_cp_inv_h = temp_I / self.phy_prop.rho_cp1 + (1. - temp_I) / self.phy_prop.rho_cp2
    #         rho_cp_inv_int_div_lda_grad_T = self.phy_prop.diff * rho_cp_inv_h * div_lda_grad_T
    #
    #         # correction de int_div_T_u et rho_cp_inv_int_div_lda_grad_T
    #         for i_int, (i1, i2) in enumerate(markers_int.ind):
    #             # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs
    #             # sur le maillage cartésien
    #
    #             for ist, i in enumerate((i1, i2)):
    #                 if i == i1:
    #                     from_liqu_to_vap = True
    #                 else:
    #                     from_liqu_to_vap = False
    #                 im3, im2, im1, i0, ip1, ip2, ip3 = cl_perio(len(T), i)
    #
    #                 # On calcule gradTg, gradTi, Ti, gradTd
    #
    #                 ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(self, i, liqu_a_gauche=from_liqu_to_vap)
    #                 cells = CellsInterface(ldag, ldad, ag, dx, T[[im3, im2, im1, i0, ip1, ip2, ip3]],
    #                                        rhocpg=rhocpg, rhocpd=rhocpd, interp_type=self.interp_type,
    #                                        vdt=h*self.dt*self.phy_prop.v)
    #
    #                 # post-traitements
    #                 if h == 1.:
    #                     self.bulles.T[i_int, ist] = cells.Ti
    #                     self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
    #                     self.bulles.Tg[i_int, ist] = cells.Tg[-1]
    #                     self.bulles.Td[i_int, ist] = cells.Td[0]
    #                     self.bulles.gradTg[i_int, ist] = cells.gradTg[-1]
    #                     self.bulles.gradTd[i_int, ist] = cells.gradTd[0]
    #
    #                 # Correction des cellules i0 - 1 à i0 + 1 inclue
    #                 # DONE: l'écrire en version flux pour être sûr de la conservation
    #                 dx = self.num_prop.dx
    #                 cor_T_u = cells.T_f * self.phy_prop.v
    #                 cor_int_div_T_u = integrale_vol_div(cor_T_u, dx)
    #                 cor_lda_grad_T = cells.lda_f * cells.gradT
    #                 cor_int_div_lda_grad_T = integrale_vol_div(cor_lda_grad_T, dx)
    #
    #                 # Correction des cellules
    #                 ind_flux = [im2, im1, i0, ip1, ip2, ip3]
    #                 self.flux_conv[ind_flux] = cor_T_u
    #                 self.flux_diff[ind_flux] = cor_lda_grad_T
    #
    #                 # TODO: changer pour être sûr de changer les bonnes cellules, cad les cellules qui correspondent et
    #                 #   pas celles qui ont suivi l'interface
    #                 rhocp_np1 = 0.
    #                 cor_int_div_rho_cp_u = 0.
    #                 int_div_lda_grad_T = 0.
    #                 int_div_rho_cp_T_u = 0.
    #                 km3, km2, km1, k0, kp1, kp2, kp3 = cl_perio(len(T), i0_tab[i_int, ist])
    #                 if i0 == k0:
    #                     ind_to_change = [km2, km1, k0, kp1, kp2]
    #                 else:
    #                     ind_to_change = [km1, k0, kp1, kp2, kp3]
    #                 cor_rho_cp_inv_h = rho_cp_inv_h[ind_to_change]
    #
    #                 int_div_T_u[ind_to_change] = cor_int_div_T_u
    #                 rho_cp_inv_int_div_lda_grad_T[ind_to_change] = cor_int_div_lda_grad_T * cor_rho_cp_inv_h
    #                 int_div_rho_cp_u[ind_to_change] = cor_int_div_rho_cp_u
    #         # TODO: adapter cette formulation
    #         K.append(1/rhocp_np1 * (T * int_div_rho_cp_u - int_div_rho_cp_T_u + int_div_lda_grad_T))
    #     coeff = np.array([1. / 6, 1 / 3., 1 / 3., 1. / 6])
    #     self.flux_conv = np.sum(coeff * np.array(T_u_l).T, axis=-1)
    #     self.flux_diff = np.sum(coeff * np.array(lda_gradT_l).T, axis=-1)
    #     self.T += np.sum(self.dt * coeff * np.array(K[1:]).T, axis=-1)


class ProblemDiscontinuT(Problem):
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

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None, conv_interf=None):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        # if self.num_prop.schema != 'upwind':
        #     raise Exception('Cette version ne marche que pour un schéma upwind')
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = 'Ti'
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf

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

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4

        Args:
            flux_conv:
            flux_diff:
            coeff_diff:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

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
                                       schema_conv=self.conv_interf)

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                T_u = cells.T_f * self.phy_prop.v
                lda_grad_T = cells.lda_f * cells.gradT
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Ti[i_int, ist] = cells.Ti

                # Correction des cellules
                # ind_to_change = [im2, im1, i0, ip1, ip2]
                # ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                ind_flux = [im1, i0, ip1, ip2, ip3]
                # print('Tu : ', T_u)
                # print('lda_graT : ', lda_grad_T)
                flux_conv[ind_flux] = T_u[1:]
                flux_diff[ind_flux] = lda_grad_T[1:]
                # Tnp1 = Tn + dt (- int_S_T_u + 1/rhocp * int_S_lda_grad_T)

    @property
    def name(self):
        return 'TFC, ' + super().name + ', ' + self.interp_type.replace('_', '-')


class ProblemDiscontinuT2(Problem):
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

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None, conv_interf=None):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        # if self.num_prop.schema != 'upwind':
        #     raise Exception('Cette version ne marche que pour un schéma upwind')
        # self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = 'Ti'
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        if conv_interf is None:
            conv_interf = self.num_prop.schema
        self.conv_interf = conv_interf

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

    def _corrige_flux_coeff_interface(self, T, bulles, *args):
        """
        Ici on corrige les flux sur place avant de les appliquer en euler, rk3 ou rk4

        Args:
            flux_conv:
            flux_diff:
            coeff_diff:

        Returns:

        """
        flux_conv, flux_diff = args
        dx = self.num_prop.dx

        for i_int, (i1, i2) in enumerate(bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

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
                                       schema_conv=self.conv_interf)

                # Correction des cellules i0 - 1 à i0 + 1 inclue
                # DONE: l'écrire en version flux pour être sûr de la conservation
                dx = self.num_prop.dx
                T_u = cells.T_f * self.phy_prop.v
                lda_over_rhocp_grad_T = cells.lda_f / cells.rhocp_f * cells.gradT
                self.bulles.lda_grad_T[i_int, ist] = cells.lda_gradTi
                self.bulles.Ti[i_int, ist] = cells.Ti

                # Correction des cellules
                # ind_to_change = [im2, im1, i0, ip1, ip2]
                # ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                ind_flux = [im1, i0, ip1, ip2, ip3]
                # print('Tu : ', T_u)
                # print('lda_graT : ', lda_grad_T)
                flux_conv[ind_flux] = T_u[1:]
                flux_diff[ind_flux] = lda_over_rhocp_grad_T[1:]
                # Tnp1 = Tn + dt (- int_S_T_u + 1/rhocp * int_S_lda_grad_T)

    def _euler_timestep(self, debug=None, bool_debug=False):
        dx = self.num_prop.dx
        self.flux_conv = self._compute_convection_flux(self.T, self.bulles, bool_debug, debug)
        self.flux_diff = self._compute_diffusion_flux(1. / self.rho_cp_a * self.T, self.bulles, bool_debug, debug)
        self._corrige_flux_coeff_interface(self.T, self.bulles, self.flux_conv, self.flux_diff)
        dTdt = - integrale_vol_div(self.flux_conv, dx) \
            + self.phy_prop.diff * integrale_vol_div(self.flux_diff, dx)
        self.T += self.dt * dTdt

    @property
    def name(self):
        return 'TFC2, ' + super().name + ', ' + self.interp_type.replace('_', '-')


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
                 delta_conv=1., int_Ti=1., delta_conv2=0.):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        self.deb = deb
        # if self.num_prop.schema != 'upwind':
        #     raise Exception('Cette version ne marche que pour un schéma upwind')
        self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = 'Ti'
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        self.delta_diff = delta_diff
        self.delta_conv = delta_conv
        self.delta_conv2 = delta_conv2
        self.int_Ti = int_Ti
        if num_prop.time_scheme != 'euler':
            print('%s time scheme not implemented, falling back to euler time scheme' % num_prop.time_scheme)
            self.num_prop._time_scheme = 'euler'

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
                                       rhocpg=rhocpg, rhocpd=rhocpd, interp_type=self.interp_type,
                                       schema_conv=self.num_prop.schema)

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
                    - self.delta_conv * cells.lda_gradTi * (1/ldad - 1/ldag) * self.phy_prop.v \
                    - self.delta_conv2 * cells.lda_gradTi * (1 / ldad + 1 / ldag) * self.phy_prop.v

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

    def _euler_timestep(self, debug=None, bool_debug=False):
        super()._euler_timestep(debug=debug, bool_debug=bool_debug)
        self._corrige_interface_aymeric1()

    @property
    def name(self):
        return 'SEFC ' + super().name


class ProblemDiscontinuSepIntT(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Cette classe résout le problème en 3 étapes :

        - on calcule le nouveau T comme avant 
        - on met à jour T_i et lda_grad_T_i avec l'ancienne température
        - on calcule précisemment selon la méthode choisie les flux à l'interface et les moyennes des propriétés 
          diphasiques à prendre ou non (on se base dans tous les cas sur les propriétés interfaciales
        - 

    Elle résout donc le problème de manière complètement monophasique et recolle à l'interface en imposant la
    continuité de lda_grad_T et T à l'interface.

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None, deb=False, delta_diff=1.,
                 delta_conv=1., int_Ti=1., delta_conv2=0.):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        self.deb = deb
        # if self.num_prop.schema != 'upwind':
        #     raise Exception('Cette version ne marche que pour un schéma upwind')
        self.T_old = self.T.copy()
        if interp_type is None:
            self.interp_type = 'Ti'
        else:
            self.interp_type = interp_type
        print(self.interp_type)
        self.delta_diff = delta_diff
        self.delta_conv = delta_conv
        self.delta_conv2 = delta_conv2
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
                                       rhocpg=rhocpg, rhocpd=rhocpd, interp_type=self.interp_type,
                                       schema_conv=self.num_prop.schema)
                rhocp = np.array([rhocpg, rhocpg, np.nan, rhocpd, rhocpd])  # le coeff du milieu est mis à nan pour
                # éviter de faire l'erreur de prendre en compte les valeurs intégrées dans la cellule diphasique.

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
                rhocp_inv_int_div_lda_grad_T = integrale_vol_div(lda_grad_T, dx) / rhocp
                rhocp_inv_int_div_lda_grad_T[2] = 1/dx * (-1./rhocpg*lda_grad_T[2] + 1./rhocpd*lda_grad_T[3]
                                                          + (1./rhocpd - 1./rhocpg)*cells.lda_gradTi)

                # Correction des cellules
                ind_to_change = [im2, im1, i0, ip1, ip2]
                ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                self.flux_conv[ind_flux] = T_u
                self.flux_diff[ind_flux] = lda_grad_T
                if self.deb:
                    pass

                # on écrit l'équation en température, ça me semble peut être mieux ?
                # Tnp1 = Tn + dt (- int_S_rho_cp_T_u + int_S_lda_grad_T
                #                 - delta * I2 * (rhocp2 - rhocpa) - [rhocp] * int_S_Ti_v_n2_dS) / rhocpa
                self.T[ind_to_change] = self.T_old[ind_to_change] \
                    + self.dt * (-int_div_T_u + self.phy_prop.diff * rhocp_inv_int_div_lda_grad_T)
        self.T_old = self.T.copy()

    def _euler_timestep(self, debug=None, bool_debug=False):
        super()._euler_timestep(debug=debug, bool_debug=bool_debug)
        self._corrige_interface()

    @property
    def name(self):
        return 'SEFC ' + super().name


class ProblemDiscontinuCoupleConserv(Problem):
    T: np.ndarray
    I: np.ndarray
    bulles: BulleTemperature

    """
    Attention cette classe n'est probablement pas finalisée

    Args:
        T0: la fonction initiale de température
        markers: les bulles
        num_prop: les propriétés numériques du calcul
        phy_prop: les propriétés physiques du calcul
    """

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None, interp_type=None):
        super().__init__(T0, markers, num_prop=num_prop, phy_prop=phy_prop)
        # if self.num_prop.schema != 'upwind':
        #     raise Exception('Cette version ne marche que pour un schéma upwind')
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
        Dans cette approche on calclue Ti et lda_gradTi à partir du système énergie température

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
                                       rhocpg=rhocpg, rhocpd=rhocpd, interp_type='energie_temperature',
                                       schema_conv='quick', vdt=self.phy_prop.v*self.dt)
                cells.compute_from_h_T(self.h_old[i0], self.T_old[i0])
                cells.compute_T_f_gradT_f_quick()

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
                # ind_flux = [im2, im1, i0, ip1, ip2, ip3]
                ind_flux = [im1, i0, ip1, ip2, ip3]
                self.flux_conv[ind_flux] = cells.T_f[1:] * self.phy_prop.v
                self.flux_conv_ener[ind_flux] = rhocp_T_u[1:]
                self.flux_diff[ind_flux] = lda_grad_T[1:]
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

    def _euler_timestep(self, debug=None, bool_debug=False):
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
                self.bulles.cells[2*i_int + ist] = cells_ft
                # print('cells :', self.bulles.cells[i_int, ist])

        self.T_old = self.T.copy()

    def _euler_timestep(self, debug=None, bool_debug=False):
        super()._euler_timestep(debug=debug, bool_debug=bool_debug)
        self._corrige_interface_ft()

    @property
    def name(self):
        return 'TFF ' + super().name  # température front-fitting
