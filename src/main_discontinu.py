from src.main import *


class BulleTemperature(Bulles):
    def __init__(self, markers=None, phy_prop=None, n_bulle=None, Delta=1., x=None):
        super().__init__(markers, phy_prop, n_bulle, Delta)
        self.T = np.zeros_like(self.markers)
        self.Tg = np.zeros_like(self.markers)
        self.Td = np.zeros_like(self.markers)
        self.lda_grad_T = np.zeros_like(self.markers)
        self.ind = None
        if x is not None:
            self.set_indices_markers(x)

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
            ind1 = (np.abs(marks[0] - x) < dx/2.).nonzero()
            ind2 = (np.abs(marks[1] - x) < dx/2.).nonzero()
            res.append([ind1, ind2])
        self.ind = np.array(res)

    def update_lda_grad_T_and_T(self, prob, T1, T2):
        """
        Met à jour lda_grad_T et T selon les nouvelles valeurs calculées de T1 et T2 (les valeurs de la température au
        centre des sous volumes de la cellule diphasique.
        Args:
            prob: the problem, with T and I
            T1: tableau des températures mises à jour dans les sous volumes de liquide (de la meme taille que
                self.markers)
            T2: idem pour la vapeur

        Returns:
            Met à jour les tableaux self.lda_grad_T et self.T
        """
        lda_grad_T_np1 = []
        T_np1 = []
        for i, marks in enumerate(self.markers):
            ind1 = self.ind[i, 0]
            lda_grad_T_1 = get_lda_grad_face_left1(ind1, T1[i, 0], prob)
            lda_grad_T_2 = get_lda_grad_face_right2(ind1, T2[i, 0], prob)
            lda_grad_T1_np1 = (1.-prob.I[ind1]) * lda_grad_T_1 + prob.I[ind1] * lda_grad_T_2  # on a pondéré par
            # l'indicatrice pour que la valeur de la face la plus proche de l'face compte plus
            T1 = get_temperature_from_left1(ind1, T1[i, 0], lda_grad_T1_np1, prob)
            T2 = get_temperature_from_right2(ind1, T2[i, 0], lda_grad_T1_np1, prob)
            T1_np1 = (T1 + T2)/2.
            ind2 = self.ind[i, 1]
            lda_grad_T_1 = get_lda_grad_face_left2(ind2, T1[i, 1], prob)
            lda_grad_T_2 = get_lda_grad_face_right1(ind2, T2[i, 1], prob)
            lda_grad_T2_np1 = (1. - prob.I[ind2]) * lda_grad_T_1 + prob.I[ind2] * lda_grad_T_2
            T1 = get_temperature_from_left2(ind2, T1[i, 1], lda_grad_T2_np1, prob)
            T2 = get_temperature_from_right1(ind2, T2[i, 1], lda_grad_T2_np1, prob)
            T2_np1 = (T1 + T2)/2.
            lda_grad_T_np1.append([lda_grad_T1_np1, lda_grad_T2_np1])
            T_np1.append([T1_np1, T2_np1])
        self.lda_grad_T = np.array(lda_grad_T_np1)
        self.T = np.array(T_np1)

    def shift(self, dx, x=None):
        super().shift(dx)
        if x is not None:
            self.set_indices_markers(x)
        else:
            raise Exception('J ai besoin de x pour recaculer les indices des cellules diphasiques')


def get_lda_grad_face_left1(ind, T, prob, cl=1):
    """
    On est dans la config suivante :
       liquide      vapeur
            +--------------+
            | T1  |   T2   |
           -|> +  | +  +  -|> avec à l'face T_i et lda_lda_grad_T_i
            |     | T      |
            +--------------+
    Args:
        ind: l'indice de la cellule diphasique
        T: la temperature T1 ou T2
        prob: le problème (qui nous donne les propriétés physiques, T et I)
        cl: la condition limite (1 = pério)

    Returns:
        le gradient de température sur la face gauche ou droite multiplié par lda de la bonne phase
    """
    lda = prob.phy_prop.lda1
    if cl == 1:
        _, im1, _, _, _ = cl_perio(len(prob.T), ind)
        Tim1 = prob.T[im1]
    else:
        raise NotImplementedError
    Delta_T = T - Tim1
    Delta_x = (0.5 + prob.I[ind]/2.) * prob.num_prop.dx
    return lda * Delta_T / Delta_x


def get_lda_grad_face_left2(ind, T, prob, cl=1):
    lda = prob.phy_prop.lda2
    if cl == 1:
        _, im1, _, _, _ = cl_perio(len(prob.T), ind)
        Tim1 = prob.T[im1]
    else:
        raise NotImplementedError
    Delta_T = T - Tim1
    Delta_x = (0.5 + (1-prob.I[ind])/2.) * prob.num_prop.dx
    return lda * Delta_T / Delta_x


def get_lda_grad_face_right1(ind, T, prob, cl=1):
    lda = prob.phy_prop.lda1
    if cl == 1:
        _, _, _, ip1, _ = cl_perio(len(prob.T), ind)
        Tip1 = prob.T[ip1]
    else:
        raise NotImplementedError
    Delta_T = Tip1 - T
    Delta_x = (0.5 + prob.I[ind]/2.) * prob.num_prop.dx
    return lda * Delta_T / Delta_x


def get_lda_grad_face_right2(ind, T, prob, cl=1):
    lda = prob.phy_prop.lda2
    if cl == 1:
        _, _, _, ip1, _ = cl_perio(len(prob.T), ind)
        Tip1 = prob.T[ip1]
    else:
        raise NotImplementedError
    Delta_T = Tip1 - T
    Delta_x = (0.5 + (1. - prob.I[ind])/2.) * prob.num_prop.dx
    return lda * Delta_T / Delta_x


def get_temperature_from_left1(ind, T, lda_grad_T, prob):
    """
    Cf. le dessin dans :method:`get_grad_interface_left1`, on calcule la température à l'interface avec lda_grad_T et T1
    ou T2
    Args:
        ind: l'indice de la cellule diphasique
        T: T1 ou T2
        lda_grad_T: c'est assez explicite
        prob: le prob qui nous donne accès à dx et à I pour caluler Delta_x

    Returns:
        La température de l'interface calculée en amont ou en aval
    """
    Delta_x = prob.I[ind]/2. * prob.num_prop.dx
    return T + lda_grad_T / prob.phy_prop.lda1 * Delta_x


def get_temperature_from_left2(ind, T, lda_grad_T, prob):
    Delta_x = (1. - prob.I[ind])/2. * prob.num_prop.dx
    return T + lda_grad_T / prob.phy_prop.lda2 * Delta_x


def get_temperature_from_right1(ind, T, lda_grad_T, prob):
    Delta_x = prob.I[ind]/2. * prob.num_prop.dx
    return T - lda_grad_T / prob.phy_prop.lda1 * Delta_x


def get_temperature_from_right2(ind, T, lda_grad_T, prob):
    Delta_x = (1. - prob.I[ind])/2. * prob.num_prop.dx
    return T - lda_grad_T / prob.phy_prop.lda2 * Delta_x


class ProblemDiscontinu(Problem):
    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None):
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
        self.T_new = self.T.copy()

    def _init_bulles(self, markers=None):
        if markers is None:
            return BulleTemperature(markers=markers, phy_prop=self.phy_prop)
        elif isinstance(markers, Bulles):
            return BulleTemperature(markers=markers.markers, phy_prop=self.phy_prop)
        elif isinstance(markers, BulleTemperature):
            return markers.copy()
        else:
            print(markers)
            raise NotImplementedError

    def corrige_interface(self):
        """

               +---------+---------+---------+---------+
               |         |         | T1|  T2 |         |
               |    +   -|>   +    | + | ++ -|>   +   -|>
               |    i-2  |    i-1  |   | i   |    i+1  |
               +---------+---------+---------+---------+

        Attention en tout il a 2 configs possibles, selon si on passe du liquide à la vapeur.
        Il faudrait factoriser le code par une méthode générale qui traite ces deux cas.
        Dans chaque cas, on doit calculer l'évolution des températures Ti-1, T1, T2 et Ti+1
        Il est à noter que si l'on se place dans un cas ou le stencil qui est à proximité de l'interface est de taille
        plus grande il faudra aussi corriger les cellules de ce stencil.

        Returns:

        """
        for i_int, (i1, i2) in enumerate(self.bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            #########################################
            # Cas du passage du liquide a la vapeur #
            #########################################

            im2, im1, i, ip1, ip2 = cl_perio(len(self.T), i1)

            # Correction de la cellule i1 - 1

            # on a une interpolation amont de la température donc on ne change rien
            int_div_T_u = 1./self.num_prop.dx * (self.T[im2] - self.T[im1]) * self.phy_prop.v

            # on calcule le gradient de température corrigé à la face i-1/2
            grad_T1 = (self.bulles.Tg[i_int, 0] - self.T[im1]) / ((0.5 + self.I[i]/2.)*self.num_prop.dx)
            grad_Tim1 = (self.T[im2] - self.T[im1])/self.num_prop.dx
            # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
            # monophasique à gauche. Les deux gradients sont monophasiques.
            int_rhocp_inv_div_lda_grad_T = 1/self.phy_prop.rho_cp1 * \
                1./self.num_prop.dx * self.phy_prop.lda1 * \
                (grad_Tim1 - grad_T1)
            self.T_new[im1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)

            # Correction de la cellule i1 à gauche
            # La cellule est de volume dS*I*dx

            # interpole en centre Tg ?
            Tg = (self.I[i]/(1. + self.I[i])) * self.T[im1] + (1. / (1. + self.I[i])) * self.bulles.Tg[i_int, 0]
            int_div_T_u = 1. / (self.I[i]*self.num_prop.dx) * (Tg - self.bulles.T[i_int, 0]) * self.phy_prop.v

            # on calcule le gradient de température corrigé à la face i-1/2
            # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
            # monophasique à gauche. Les deux gradients sont monophasiques.
            grad_T1 = (self.bulles.Tg[i_int, 0] - self.T[im1]) / ((0.5 + self.I[i] / 2.) * self.num_prop.dx)
            grad_Tint = self.bulles.lda_grad_T[i_int, 0]/self.phy_prop.lda1
            int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp1 * \
                1. / (self.I[i]*self.num_prop.dx) * self.phy_prop.lda1 * \
                (grad_T1 - grad_Tint)
            self.bulles.Tg[i_int, 0] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)

            # Correction de la cellule i1 à droite
            # La cellule est de volume dS*(1-I)*dx

            # on a une interpolation centre de Td entre Td_centre et Tip1
            Td = ((1. - self.I[i])/(2. - self.I[i])) * self.T[ip1] + (1. / (2. - self.I[i])) * self.bulles.Td[i_int, 0]
            int_div_T_u = 1. / ((1.-self.I[i]) * self.num_prop.dx) * (self.bulles.T[i_int, 0] - Td) * self.phy_prop.v

            # on calcule le gradient de température corrigé à la face i+1/2
            grad_Tint = self.bulles.lda_grad_T[i_int, 0] / self.phy_prop.lda2
            grad_T2 = (self.T[ip1] - self.bulles.Td[i_int, 0]) / ((0.5 + (1.-self.I[i])/2.) * self.num_prop.dx)
            # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
            # monophasique à gauche. Les deux gradients sont monophasiques.
            int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp2 * \
                1. / ((1.-self.I[i]) * self.num_prop.dx) * self.phy_prop.lda2 * \
                (grad_Tint - grad_T2)
            self.bulles.Td[i_int, 0] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)

            # On remplit la valeur au centre de la cellule i, qui vaut soit celle de la partie liquide, soit celle de
            # la partie vapeur selon la position de l'interface par rapport au centre de la cellule.
            if self.bulles.markers[i_int, 0] > self.num_prop.x[i]:
                self.T_new[i] = self.bulles.Tg[i_int, 0]
            else:
                self.T_new[i] = self.bulles.Td[i_int, 0]

            # Correction de la cellule i1 + 1

            # on a une interpolation amont de Ti qu'on prend à Td_centre
            Td = self.bulles.Td[i_int, 0]
            int_div_T_u = 1. / self.num_prop.dx * (Td - self.T[ip1]) * self.phy_prop.v

            # on calcule le gradient de température corrigé à la face i+1/2
            grad_T2 = (self.T[ip1] - self.bulles.Td[i_int, 0]) / ((0.5 + (1. - self.I[i]) / 2.) * self.num_prop.dx)
            grad_Tip1 = (self.T[ip2] - self.T[ip1]) / self.num_prop.dx
            # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
            # monophasique à gauche. Les deux gradients sont monophasiques.
            int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp2 * \
                1. / self.num_prop.dx * self.phy_prop.lda2 * \
                (grad_T2 - grad_Tip1)
            self.T_new[ip1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)

            ##########################################
            # Cas du passage de la vapeur au liquide #
            ##########################################

            im2, im1, i, ip1, ip2 = cl_perio(len(self.T), i2)

            # Correction de la cellule i2 - 1

            # on a une interpolation amont de la température donc on ne change rien
            int_div_T_u = 1. / self.num_prop.dx * (self.T[im2] - self.T[im1]) * self.phy_prop.v

            # on calcule le gradient de température corrigé à la face i-1/2
            grad_T1 = (self.bulles.Tg[i_int, 1] - self.T[im1]) / ((0.5 + (1.-self.I[i]) / 2.) * self.num_prop.dx)
            grad_Tim1 = (self.T[im2] - self.T[im1]) / self.num_prop.dx
            # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
            # monophasique à gauche. Les deux gradients sont monophasiques.
            int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp2 * \
                1. / self.num_prop.dx * self.phy_prop.lda2 * \
                (grad_Tim1 - grad_T1)
            self.T_new[im1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)

            # Correction de la cellule i2 à gauche
            # La cellule est de volume dS*(1-I)*dx

            # interpole en centre Tg ?
            Tg = ((1. - self.I[i]) / (2. - self.I[i])) * self.T[im1] + (1. / (2. - self.I[i])) * self.bulles.Tg[i_int,
                                                                                                                1]
            int_div_T_u = 1. / ((1.-self.I[i]) * self.num_prop.dx) * (Tg - self.bulles.T[i_int, 1]) * self.phy_prop.v

            # on calcule le gradient de température corrigé à la face i-1/2
            # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
            # monophasique à gauche. Les deux gradients sont monophasiques.
            grad_T1 = (self.bulles.Tg[i_int, 1] - self.T[im1]) / ((0.5 + (1. - self.I[i]) / 2.) * self.num_prop.dx)
            grad_Tint = self.bulles.lda_grad_T[i_int, 1] / self.phy_prop.lda2
            int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp2 * \
                1. / ((1.-self.I[i]) * self.num_prop.dx) * self.phy_prop.lda2 * \
                (grad_T1 - grad_Tint)
            self.bulles.Tg[i_int, 1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)

            # Correction de la cellule i2 à droite
            # La cellule est de volume dS*I*dx

            # on a une interpolation centre de Td entre Td_centre et Tip1
            Td = (self.I[i] / (1. + self.I[i])) * self.T[ip1] + (1. / (1. + self.I[i])) * self.bulles.Td[i_int, 1]
            int_div_T_u = 1. / (self.I[i] * self.num_prop.dx) * (self.bulles.T[i_int, 1] - Td) * self.phy_prop.v

            # on calcule le gradient de température corrigé à la face i+1/2
            grad_Tint = self.bulles.lda_grad_T[i_int, 1] / self.phy_prop.lda1
            grad_T2 = (self.T[ip1] - self.bulles.Td[i_int, 1]) / ((0.5 + self.I[i] / 2.) * self.num_prop.dx)
            # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
            # monophasique à gauche. Les deux gradients sont monophasiques.
            int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp1 * \
                1. / (self.I[i] * self.num_prop.dx) * self.phy_prop.lda1 * \
                (grad_Tint - grad_T2)
            self.bulles.Td[i_int, 1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)

            # On remplit la valeur au centre de la cellule i, qui vaut soit celle de la partie liquide, soit celle de
            # la partie vapeur selon la position de l'interface par rapport au centre de la cellule.
            if self.bulles.markers[i_int, 1] > self.num_prop.x[i]:
                self.T_new[i] = self.bulles.Tg[i_int, 1]
            else:
                self.T_new[i] = self.bulles.Td[i_int, 1]

            # Correction de la cellule i2 + 1

            # on a une interpolation amont de Ti qu'on prend à Td_centre
            Td = self.bulles.Td[i_int, 1]
            int_div_T_u = 1. / self.num_prop.dx * (Td - self.T[ip1]) * self.phy_prop.v

            # on calcule le gradient de température corrigé à la face i+1/2
            grad_T2 = (self.T[ip1] - self.bulles.Td[i_int, 1]) / ((0.5 + self.I[i] / 2.) * self.num_prop.dx)
            grad_Tip1 = (self.T[ip2] - self.T[ip1]) / self.num_prop.dx
            # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
            # monophasique à gauche. Les deux gradients sont monophasiques.
            int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp1 * \
                1. / self.num_prop.dx * self.phy_prop.lda1 * \
                (grad_T2 - grad_Tip1)
            self.T_new[ip1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)


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
