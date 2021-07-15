import numpy as np

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

    # def update_lda_grad_T_and_T(self, prob):
    #     """
    #     Met à jour lda_grad_T et T selon les nouvelles valeurs calculées de T1 et T2 (les valeurs de la température au
    #     centre des sous volumes de la cellule diphasique.
    #     Args:
    #         prob: the problem, with T and I
    #
    #     Returns:
    #         Met à jour les tableaux self.lda_grad_T et self.T
    #     """
    #     lda_grad_T_np1 = []
    #     T_np1 = []
    #     for i, marks in enumerate(self.markers):
    #         ind1 = self.ind[i, 0]
    #         lda_grad_T_1 = get_lda_grad_face_left1(ind1, self.Tg[i, 0], prob)
    #         lda_grad_T_2 = get_lda_grad_face_right2(ind1, self.Td[i, 0], prob)
    #         lda_grad_T1_np1 = (1.-prob.I[ind1]) * lda_grad_T_1 + prob.I[ind1] * lda_grad_T_2  # on a pondéré par
    #         # l'indicatrice pour que la valeur de la face la plus proche de l'face compte plus
    #         Tg = get_temperature_from_left1(ind1, self.Tg[i, 0], lda_grad_T1_np1, prob)
    #         Td = get_temperature_from_right2(ind1, self.Td[i, 0], lda_grad_T1_np1, prob)
    #         T1_np1 = (Tg + Td)/2.
    #         ind2 = self.ind[i, 1]
    #         lda_grad_T_1 = get_lda_grad_face_left2(ind2, self.Tg[i, 1], prob)
    #         lda_grad_T_2 = get_lda_grad_face_right1(ind2, self.Td[i, 1], prob)
    #         lda_grad_T2_np1 = (1. - prob.I[ind2]) * lda_grad_T_1 + prob.I[ind2] * lda_grad_T_2
    #         Tg = get_temperature_from_left2(ind2, self.Tg[i, 1], lda_grad_T2_np1, prob)
    #         Td = get_temperature_from_right1(ind2, self.Td[i, 1], lda_grad_T2_np1, prob)
    #         T2_np1 = (Tg + Td)/2.
    #         lda_grad_T_np1.append([lda_grad_T1_np1, lda_grad_T2_np1])
    #         T_np1.append([T1_np1, T2_np1])
    #     self.lda_grad_T = np.array(lda_grad_T_np1)
    #     self.T = np.array(T_np1)


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


def get_T_i_and_lda_grad_T_i(ldag, ldad, Tg, Td, dg, dd):
    T_i = (ldag/dg*Tg + ldad/dd*Td) / (ldag/dg + ldad/dd)
    lda_grad_T_i = ldag * (T_i - Tg)/dg
    lda_grad_T_id = ldad * (Td - T_i)/dd
    if lda_grad_T_id != lda_grad_T_i:
        print('Erreur, les lda_gradT sont différents')
        print('gauche : ', lda_grad_T_i)
        print('droite : ', lda_grad_T_id)
    return T_i, lda_grad_T_i


def pid_interp(T, d):
    Tm = np.sum(T/d) / np.sum(1./d)
    return Tm


# def compute_Tgdi(Tipm1, Tgd, agd, dx):
#     """
#     Comme il suffit d'inverser le sens des absisses pour que la situation soit la meme, la formule est valable pour
#     calculer T gauche i et T droite i
#     Args:
#         Tipm1: Tim1 ou Tip1
#         Tgd: Tg ou Td
#         agd: ag ou ad
#         dx:
#
#     Returns:
#         Tgi ou Tdi
#     """
#     dg_ipm1 = (0.5 + agd/2.) * dx
#     gradT = (Tgd - Tipm1)/dg_ipm1
#     dgd_i = (0.5 - agd/2.) * dx
#     Tgdi = Tgd + gradT * dgd_i
#     return Tgdi
#
#
# def get_lda_grad_face_left1(ind, T, prob, cl=1):
#     """
#     On est dans la config suivante :
#        liquide      vapeur
#             +--------------+
#             | T1  |   T2   |
#            -|> +  | +  +  -|> avec à l'face T_i et lda_lda_grad_T_i
#             |     | T      |
#             +--------------+
#     Args:
#         ind: l'indice de la cellule diphasique
#         T: la temperature T1 ou T2
#         prob: le problème (qui nous donne les propriétés physiques, T et I)
#         cl: la condition limite (1 = pério)
#
#     Returns:
#         le gradient de température sur la face gauche ou droite multiplié par lda de la bonne phase
#     """
#     lda = prob.phy_prop.lda1
#     if cl == 1:
#         _, im1, _, _, _ = cl_perio(len(prob.T), ind)
#         Tim1 = prob.T[im1]
#     else:
#         raise NotImplementedError
#     Delta_T = T - Tim1
#     Delta_x = (0.5 + prob.I[ind]/2.) * prob.num_prop.dx
#     return lda * Delta_T / Delta_x
#
#
# def get_lda_grad_face_left2(ind, T, prob, cl=1):
#     lda = prob.phy_prop.lda2
#     if cl == 1:
#         _, im1, _, _, _ = cl_perio(len(prob.T), ind)
#         Tim1 = prob.T[im1]
#     else:
#         raise NotImplementedError
#     Delta_T = T - Tim1
#     Delta_x = (0.5 + (1-prob.I[ind])/2.) * prob.num_prop.dx
#     return lda * Delta_T / Delta_x
#
#
# def get_lda_grad_face_right1(ind, T, prob, cl=1):
#     lda = prob.phy_prop.lda1
#     if cl == 1:
#         _, _, _, ip1, _ = cl_perio(len(prob.T), ind)
#         Tip1 = prob.T[ip1]
#     else:
#         raise NotImplementedError
#     Delta_T = Tip1 - T
#     Delta_x = (0.5 + prob.I[ind]/2.) * prob.num_prop.dx
#     return lda * Delta_T / Delta_x
#
#
# def get_lda_grad_face_right2(ind, T, prob, cl=1):
#     lda = prob.phy_prop.lda2
#     if cl == 1:
#         _, _, _, ip1, _ = cl_perio(len(prob.T), ind)
#         Tip1 = prob.T[ip1]
#     else:
#         raise NotImplementedError
#     Delta_T = Tip1 - T
#     Delta_x = (0.5 + (1. - prob.I[ind])/2.) * prob.num_prop.dx
#     return lda * Delta_T / Delta_x
#
#
# def get_temperature_from_left1(ind, T, lda_grad_T, prob):
#     """
#     Cf. le dessin dans :method:`get_grad_interface_left1`, on calcule la température à l'interface avec lda_grad_T et T1
#     ou T2
#     Args:
#         ind: l'indice de la cellule diphasique
#         T: T1 ou T2
#         lda_grad_T: c'est assez explicite
#         prob: le prob qui nous donne accès à dx et à I pour caluler Delta_x
#
#     Returns:
#         La température de l'interface calculée en amont ou en aval
#     """
#     Delta_x = prob.I[ind]/2. * prob.num_prop.dx
#     return T + lda_grad_T / prob.phy_prop.lda1 * Delta_x
#
#
# def get_temperature_from_left2(ind, T, lda_grad_T, prob):
#     Delta_x = (1. - prob.I[ind])/2. * prob.num_prop.dx
#     return T + lda_grad_T / prob.phy_prop.lda2 * Delta_x
#
#
# def get_temperature_from_right1(ind, T, lda_grad_T, prob):
#     Delta_x = prob.I[ind]/2. * prob.num_prop.dx
#     return T - lda_grad_T / prob.phy_prop.lda1 * Delta_x
#
#
# def get_temperature_from_right2(ind, T, lda_grad_T, prob):
#     Delta_x = (1. - prob.I[ind])/2. * prob.num_prop.dx
#     return T - lda_grad_T / prob.phy_prop.lda2 * Delta_x
#
#
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
        if self.num_prop.schema != 'weno upwind':
            raise Exception('Cette version ne marche que pour un stencil de 2 à proximité de l interface')
        self.T_old = self.T.copy()

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

    def corrige_interface_adrien1(self):
        """
        Cellule type ::

                             Ti,
                             lda_gradTi
                               Ti0g
                               Ti0d
                        Tgf        Tdf
               +---------+---------+---------+
               |         |   |     |         |
              -|>   +    |   | +  -|>   +   -|>
               |    i-1  |   | i   |    i+1  |
               +---------+---------+---------+
           gradTi-3/2                    gradTi+3/2
                      gradTg      gradTd

        Dans cette approche on calclue Ti et lda_gradTi avec Tim1 et Tip1.
        On en déduit Ti-1/2, Ti+1/2 corrige. On peut utiliser une interpolation proportionnelle à l'inverse de la
        distance pour en déduire lda_gradTgf et lda_gradTdf avec lda_gradTi et gradTi-3/2 et gradTi+3/2.

        Returns:
            Rien, mais met à jour T en le remplaçant par les nouvelles valeurs à proximité de l'interface, puis met à
            jour T_old
        """
        bulles_np1 = self.bulles.copy()
        bulles_np1.shift(self.phy_prop.v*self.dt)
        Inp1 = bulles_np1.indicatrice_liquide(self.num_prop.x)
        rhocp_np1 = self.phy_prop.rho_cp1 * Inp1 + self.phy_prop.rho_cp2 * (1.-Inp1)

        for i_int, (i1, i2) in enumerate(self.bulles.ind):
            # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
            # le maillage cartésien

            for ist, i in enumerate((i1, i2)):
                if i == i1:
                    from_liqu_to_vap = True
                else:
                    from_liqu_to_vap = False
                im2, im1, i0, ip1, ip2 = cl_perio(len(self.T), i)

                # On calcule gradTgf, Tgf, gradTi, Ti, gradTdf, Tdf

                ldag, rhocpg, ag, ldad, rhocpd, ad = get_prop(self, i, liqu_a_gauche=from_liqu_to_vap)
                Ti, lda_gradTi = get_T_i_and_lda_grad_T_i(ldag, ldad, self.T_old[im1], self.T_old[ip1],
                                                          (1.+ag)/2.*self.num_prop.dx, (1.+ad)/2.*self.num_prop.dx)
                self.bulles.T[i_int, ist] = Ti
                self.bulles.lda_grad_T[i_int, ist] = lda_gradTi
                # print('Ti : ', Ti)
                # print('ldagradTi : ', lda_gradTi)
                grad_Tim32 = (self.T_old[im1] - self.T_old[im2])/self.num_prop.dx
                grad_Tip32 = (self.T_old[ip2] - self.T_old[ip1]) / self.num_prop.dx
                # print('gradTim32 : ', grad_Tim32)
                grad_Tg = pid_interp(np.array([grad_Tim32, lda_gradTi/ldag]), np.array([1., ag])*self.num_prop.dx)
                # print('grad tg', grad_Tg.shape)
                grad_Td = pid_interp(np.array([lda_gradTi/ldad, grad_Tip32]), np.array([ad, 1.])*self.num_prop.dx)
                # Tgf = pid_interp(np.array([self.T_old[im1], Ti]), np.array([0.5, ag/2.])*self.num_prop.dx)
                # Tdf = pid_interp(np.array([Ti, self.T_old[ip1]]), np.array([ad/2., 0.5])*self.num_prop.dx)
                Ti0d = self.T_old[ip1] - grad_Td*self.num_prop.dx
                Ti0g = self.T_old[im1] + grad_Tg*self.num_prop.dx

                self.bulles.Tg[i_int, ist] = Ti0g
                self.bulles.Td[i_int, ist] = Ti0d
                self.bulles.gradTg[i_int, ist] = grad_Tg
                self.bulles.gradTd[i_int, ist] = grad_Td

                dV = self.num_prop.dx
                # Correction de la cellule i0 - 1
                # interpolation upwind
                int_div_T_u = 1./dV * (self.T_old[im2] - self.T_old[im1]) * self.phy_prop.v
                # gradients centres
                int_rhocp_inv_div_lda_grad_T = 1./dV * ldag/rhocpg * (grad_Tim32 - grad_Tg)
                self.T[im1] += self.dt * (-int_div_T_u + self.phy_prop.diff * int_rhocp_inv_div_lda_grad_T)

                # Correction de la cellule i0
                int_div_rhocp_T_u = 1./dV * (rhocpg*self.T_old[im1] -
                                             rhocpd*Ti0d) * self.phy_prop.v
                int_div_lda_grad_T = 1./dV * (ldag * grad_Tg - ldad * grad_Td)
                self.T[i] = (self.T_old[i]*self.rho_cp_a[i] +
                             self.dt * (-int_div_rhocp_T_u + self.phy_prop.diff * int_div_lda_grad_T)) / rhocp_np1[i]

                # Correction de la cellule i0 + 1
                int_div_T_u = 1. / dV * (Ti0d - self.T_old[ip1]) * self.phy_prop.v
                int_rhocp_inv_div_lda_grad_T = 1. / rhocpd / dV * ldad * (grad_Td - grad_Tip32)
                self.T[ip1] += self.dt * (-int_div_T_u + self.phy_prop.diff * int_rhocp_inv_div_lda_grad_T)
        self.T_old = self.T.copy()

    # def corrige_interface_broken(self):
    #     """
    #
    #            +---------+---------+---------+---------+
    #            |         |         | T1|  T2 |         |
    #            |    +   -|>   +    | + | ++ -|>   +   -|>
    #            |    i-2  |    i-1  |   | i   |    i+1  |
    #            +---------+---------+---------+---------+
    #
    #     Attention en tout il a 2 configs possibles, selon si on passe du liquide à la vapeur.
    #     Il faudrait factoriser le code par une méthode générale qui traite ces deux cas.
    #     Dans chaque cas, on doit calculer l'évolution des températures Ti-1, T1, T2 et Ti+1
    #     Il est à noter que si l'on se place dans un cas ou le stencil qui est à proximité de l'interface est de taille
    #     plus grande il faudra aussi corriger les cellules de ce stencil.
    #
    #     Returns:
    #         Rien, mais met à jour T en le remplaçant par T_new
    #     """
    #     for i_int, (i1, i2) in enumerate(self.bulles.ind):
    #         # i_int sert à aller chercher les valeurs aux interfaces, i1 et i2 servent à aller chercher les valeurs sur
    #         # le maillage cartésien
    #
    #         #########################################
    #         # Cas du passage du liquide a la vapeur #
    #         #########################################
    #
    #         im2, im1, i, ip1, ip2 = cl_perio(len(self.T), i1)
    #         # Correction de la cellule i1 - 1
    #
    #         # on a une interpolation amont de la température donc on ne change rien
    #         int_div_T_u = 1./self.num_prop.dx * (self.T_old[im2] - self.T_old[im1]) * self.phy_prop.v
    #
    #         # on calcule le gradient de température corrigé à la face i-1/2
    #         grad_T1 = (self.bulles.Tg[i_int, 0] - self.T_old[im1]) / ((0.5 + self.I[i]/2.)*self.num_prop.dx)
    #         grad_Tim1 = (self.T_old[im2] - self.T_old[im1])/self.num_prop.dx
    #         # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
    #         # monophasique à gauche. Les deux gradients sont monophasiques.
    #         int_rhocp_inv_div_lda_grad_T = 1/self.phy_prop.rho_cp1 * \
    #             1./self.num_prop.dx * self.phy_prop.lda1 * \
    #             (grad_Tim1 - grad_T1)
    #         self.T[im1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)
    #
    #         # Correction de la cellule i1 à gauche
    #         # La cellule est de volume dS*I*dx
    #
    #         # interpole en centre Tg ?
    #         Tg = (self.I[i]/(1. + self.I[i])) * self.T_old[im1] + (1. / (1. + self.I[i])) * self.bulles.Tg[i_int, 0]
    #         int_div_T_u = 1. / (self.I[i]*self.num_prop.dx) * (Tg - self.bulles.T[i_int, 0]) * self.phy_prop.v
    #
    #         # on calcule le gradient de température corrigé à la face i-1/2
    #         # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
    #         # monophasique à gauche. Les deux gradients sont monophasiques.
    #         grad_T1 = (self.bulles.Tg[i_int, 0] - self.T_old[im1]) / ((0.5 + self.I[i] / 2.) * self.num_prop.dx)
    #         grad_Tint = self.bulles.lda_grad_T[i_int, 0]/self.phy_prop.lda1
    #         int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp1 * \
    #             1. / (self.I[i]*self.num_prop.dx) * self.phy_prop.lda1 * \
    #             (grad_T1 - grad_Tint)
    #         self.bulles.Tg[i_int, 0] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)
    #
    #         # Correction de la cellule i1 à droite
    #         # La cellule est de volume dS*(1-I)*dx
    #
    #         # on a une interpolation centre de Td entre Td_centre et Tip1
    #         Td = ((1. - self.I[i])/(2. - self.I[i])) * self.T_old[ip1] + (1. / (2. - self.I[i])) * self.bulles.Td[i_int, 0]
    #         int_div_T_u = 1. / ((1.-self.I[i]) * self.num_prop.dx) * (self.bulles.T[i_int, 0] - Td) * self.phy_prop.v
    #
    #         # on calcule le gradient de température corrigé à la face i+1/2
    #         grad_Tint = self.bulles.lda_grad_T[i_int, 0] / self.phy_prop.lda2
    #         grad_T2 = (self.T_old[ip1] - self.bulles.Td[i_int, 0]) / ((0.5 + (1.-self.I[i])/2.) * self.num_prop.dx)
    #         # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
    #         # monophasique à gauche. Les deux gradients sont monophasiques.
    #         int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp2 * \
    #             1. / ((1.-self.I[i]) * self.num_prop.dx) * self.phy_prop.lda2 * \
    #             (grad_Tint - grad_T2)
    #         self.bulles.Td[i_int, 0] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)
    #
    #         # On remplit la valeur au centre de la cellule i, qui vaut soit celle de la partie liquide, soit celle de
    #         # la partie vapeur selon la position de l'interface par rapport au centre de la cellule.
    #         if self.bulles.markers[i_int, 0] > self.num_prop.x[i]:
    #             self.T[i] = self.bulles.Tg[i_int, 0]
    #         else:
    #             self.T[i] = self.bulles.Td[i_int, 0]
    #
    #         # Correction de la cellule i1 + 1
    #
    #         # on a une interpolation amont de Ti qu'on prend à Td_centre
    #         Td = self.bulles.Td[i_int, 0]
    #         int_div_T_u = 1. / self.num_prop.dx * (Td - self.T_old[ip1]) * self.phy_prop.v
    #
    #         # on calcule le gradient de température corrigé à la face i+1/2
    #         grad_T2 = (self.T_old[ip1] - self.bulles.Td[i_int, 0]) / ((0.5 + (1. - self.I[i]) / 2.) * self.num_prop.dx)
    #         grad_Tip1 = (self.T_old[ip2] - self.T_old[ip1]) / self.num_prop.dx
    #         # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
    #         # monophasique à gauche. Les deux gradients sont monophasiques.
    #         int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp2 * \
    #             1. / self.num_prop.dx * self.phy_prop.lda2 * \
    #             (grad_T2 - grad_Tip1)
    #         self.T[ip1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)
    #
    #         ##########################################
    #         # Cas du passage de la vapeur au liquide #
    #         ##########################################
    #
    #         im2, im1, i, ip1, ip2 = cl_perio(len(self.T), i2)
    #
    #         # Correction de la cellule i2 - 1
    #
    #         # on a une interpolation amont de la température donc on ne change rien
    #         int_div_T_u = 1. / self.num_prop.dx * (self.T_old[im2] - self.T_old[im1]) * self.phy_prop.v
    #
    #         # on calcule le gradient de température corrigé à la face i-1/2
    #         grad_T1 = (self.bulles.Tg[i_int, 1] - self.T_old[im1]) / ((0.5 + (1.-self.I[i]) / 2.) * self.num_prop.dx)
    #         grad_Tim1 = (self.T_old[im2] - self.T_old[im1]) / self.num_prop.dx
    #         # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
    #         # monophasique à gauche. Les deux gradients sont monophasiques.
    #         int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp2 * \
    #             1. / self.num_prop.dx * self.phy_prop.lda2 * \
    #             (grad_Tim1 - grad_T1)
    #         self.T[im1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)
    #
    #         # Correction de la cellule i2 à gauche
    #         # La cellule est de volume dS*(1-I)*dx
    #
    #         # interpole en centre Tg ?
    #         Tg = ((1. - self.I[i]) / (2. - self.I[i])) * self.T_old[im1] + (1. / (2. - self.I[i])) * self.bulles.Tg[i_int,
    #                                                                                                             1]
    #         int_div_T_u = 1. / ((1.-self.I[i]) * self.num_prop.dx) * (Tg - self.bulles.T[i_int, 1]) * self.phy_prop.v
    #
    #         # on calcule le gradient de température corrigé à la face i-1/2
    #         # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
    #         # monophasique à gauche. Les deux gradients sont monophasiques.
    #         grad_T1 = (self.bulles.Tg[i_int, 1] - self.T_old[im1]) / ((0.5 + (1. - self.I[i]) / 2.) * self.num_prop.dx)
    #         grad_Tint = self.bulles.lda_grad_T[i_int, 1] / self.phy_prop.lda2
    #         int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp2 * \
    #             1. / ((1.-self.I[i]) * self.num_prop.dx) * self.phy_prop.lda2 * \
    #             (grad_T1 - grad_Tint)
    #         self.bulles.Tg[i_int, 1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)
    #
    #         # Correction de la cellule i2 à droite
    #         # La cellule est de volume dS*I*dx
    #
    #         # on a une interpolation centre de Td entre Td_centre et Tip1
    #         Td = (self.I[i] / (1. + self.I[i])) * self.T_old[ip1] + (1. / (1. + self.I[i])) * self.bulles.Td[i_int, 1]
    #         int_div_T_u = 1. / (self.I[i] * self.num_prop.dx) * (self.bulles.T[i_int, 1] - Td) * self.phy_prop.v
    #
    #         # on calcule le gradient de température corrigé à la face i+1/2
    #         grad_Tint = self.bulles.lda_grad_T[i_int, 1] / self.phy_prop.lda1
    #         grad_T2 = (self.T_old[ip1] - self.bulles.Td[i_int, 1]) / ((0.5 + self.I[i] / 2.) * self.num_prop.dx)
    #         # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
    #         # monophasique à gauche. Les deux gradients sont monophasiques.
    #         int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp1 * \
    #             1. / (self.I[i] * self.num_prop.dx) * self.phy_prop.lda1 * \
    #             (grad_Tint - grad_T2)
    #         self.bulles.Td[i_int, 1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)
    #
    #         # On remplit la valeur au centre de la cellule i, qui vaut soit celle de la partie liquide, soit celle de
    #         # la partie vapeur selon la position de l'interface par rapport au centre de la cellule.
    #         if self.bulles.markers[i_int, 1] > self.num_prop.x[i]:
    #             self.T[i] = self.bulles.Tg[i_int, 1]
    #         else:
    #             self.T[i] = self.bulles.Td[i_int, 1]
    #
    #         # Correction de la cellule i2 + 1
    #
    #         # on a une interpolation amont de Ti qu'on prend à Td_centre
    #         Td = self.bulles.Td[i_int, 1]
    #         int_div_T_u = 1. / self.num_prop.dx * (Td - self.T_old[ip1]) * self.phy_prop.v
    #
    #         # on calcule le gradient de température corrigé à la face i+1/2
    #         grad_T2 = (self.T_old[ip1] - self.bulles.Td[i_int, 1]) / ((0.5 + self.I[i] / 2.) * self.num_prop.dx)
    #         grad_Tip1 = (self.T_old[ip2] - self.T_old[ip1]) / self.num_prop.dx
    #         # on calcule la divergence de diffusion avec ce nouveau gradient à droite et l'ancien qui est bien
    #         # monophasique à gauche. Les deux gradients sont monophasiques.
    #         int_rhocp_inv_div_lda_grad_T = 1 / self.phy_prop.rho_cp1 * \
    #             1. / self.num_prop.dx * self.phy_prop.lda1 * \
    #             (grad_T2 - grad_Tip1)
    #         self.T[ip1] += self.dt * (-int_div_T_u + int_rhocp_inv_div_lda_grad_T)

    def euler_timestep(self, debug=None, bool_debug=False):
        super().euler_timestep(debug=debug, bool_debug=bool_debug)
        self.corrige_interface_adrien1()
        # self.bulles.update_lda_grad_T_and_T(self)

    def update_markers(self):
        self.bulles.shift(self.phy_prop.v * self.dt)
        self.bulles.set_indices_markers(self.num_prop.x)
        self.I = self.update_I()


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
