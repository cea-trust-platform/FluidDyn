import numpy as np
from scipy import optimize as opt


# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def integrale_volume_div(center_value, face_value, I=None, cl=1, dS=1., schema='weno', cv_0=0., cv_n=0.):
    """
    Calcule le delta de convection aux bords des cellules

    Args:
        I: s'il faut prendre en compte si le stencil traverse une interface
        dS:
        schema:
        center_value: les valeurs présentes aux centres des cellules
        face_value: les valeurs présentes aux faces des cellules
        cl: si cl = 1, on prend des gradients nuls aux bords du domaine, si cl = 0 on utilise cv_0 et cv_n
        cv_0 : la valeur au centre au bord en -1
        cv_n : la valeur au centre au bord en n+1

    Returns:
        le delta au centre
    """
    if len(center_value.shape) != 1 or len(face_value.shape) != 1:
        raise NotImplementedError
    if center_value.shape[0] != face_value.shape[0] - 1:
        print('Le tableau des valeurs aux faces ne correspond pas au tableau des valeurs au centre')
        print(center_value.shape, face_value.shape)
        raise NotImplementedError
    if schema is 'upwind':
        interpolated_value = interpolate_from_center_to_face_upwind(center_value, cl=cl, cv_0=cv_0)
    elif schema is 'center':
        interpolated_value = interpolate_from_center_to_face_center(center_value, cl=cl, cv_0=cv_0, cv_n=cv_n)
    elif schema is 'weno':
        interpolated_value = interpolate_form_center_to_face_weno(center_value, cl=cl, cv_0=cv_0, cv_n=cv_n)
    elif schema == 'weno upwind':
        if I is None:
            raise NotImplementedError
        interpolated_value = interpolate_center_value_weno_to_face_upwind_interface(center_value, I, cl=cl, cv_0=cv_0,
                                                                                    cv_n=cv_n)
    else:
        raise NotImplementedError
    flux = interpolated_value * face_value
    delta_center_value = flux[1:] - flux[:-1]
    return dS * delta_center_value


def interpolate_from_center_to_face_center(center_value, cl=1, cv_0=0., cv_n=0.):
    interpolated_value = np.zeros((center_value.shape[0] + 1,))
    interpolated_value[1:-1] = (center_value[:-1] + center_value[1:]) / 2.
    if cl == 1:
        interpolated_value[0] = (center_value[0] + center_value[-1]) / 2.
        interpolated_value[-1] = (center_value[0] + center_value[-1]) / 2.
    elif cl == 2:
        interpolated_value[0] = interpolated_value[1]
        interpolated_value[-1] = interpolated_value[-2]
    elif cl == 0:
        interpolated_value[0] = (center_value[0] + cv_0) / 2.
        interpolated_value[-1] = (center_value[-1] + cv_n) / 2.
    else:
        raise NotImplementedError
    return interpolated_value


def interpolate_from_center_to_face_upwind(center_value, cl=1, cv_0=0.):
    interpolated_value = np.zeros((center_value.shape[0] + 1,))
    interpolated_value[1:] = center_value
    if cl == 2:
        interpolated_value[0] = interpolated_value[1]
        interpolated_value[-1] = interpolated_value[-2]
    elif cl == 1:
        interpolated_value[0] = center_value[-1]
    elif cl == 0:
        interpolated_value[0] = cv_0
    else:
        raise NotImplementedError
    return interpolated_value


def interpolate_form_center_to_face_weno(a, cl=1, cv_0=0., cv_n=0.):
    """
    Weno scheme

    Args:
        cv_n: center value at n+1
        cv_0: center value at -1
        a: the scalar value at the center of the cell
        cl: conditions aux limites, cl = 1: périodicité, cl=0 valeurs imposées aux bords à cv_0 et cv_n avec gradients
            nuls

    Returns:
        les valeurs interpolées aux faces de la face -1/2 à la face n+1/2
    """
    center_values = np.empty(a.size + 5)
    if cl == 1:
        center_values[:3] = a[-3:]
        center_values[3:-2] = a
        center_values[-2:] = a[:2]
    elif cl == 0:
        center_values[:3] = cv_0
        center_values[3:-2] = a
        center_values[-2:] = cv_n
    else:
        raise NotImplementedError
    ujm2 = center_values[:-4]
    ujm1 = center_values[1:-3]
    uj = center_values[2:-2]
    ujp1 = center_values[3:-1]
    ujp2 = center_values[4:]
    f1 = 1. / 3 * ujm2 - 7. / 6 * ujm1 + 11. / 6 * uj
    f2 = -1. / 6 * ujm1 + 5. / 6 * uj + 1. / 3 * ujp1
    f3 = 1. / 3 * uj + 5. / 6 * ujp1 - 1. / 6 * ujp2
    eps = np.array(10. ** -6)
    b1 = 13. / 12 * (ujm2 - 2 * ujm1 + uj) ** 2 + 1. / 4 * (ujm2 - 4 * ujm1 + 3 * uj) ** 2
    b2 = 13. / 12 * (ujm1 - 2 * uj + ujp1) ** 2 + 1. / 4 * (ujm1 - ujp1) ** 2
    b3 = 13. / 12 * (uj - 2 * ujp1 + ujp2) ** 2 + 1. / 4 * (3 * uj - 4 * ujp1 + ujp2) ** 2
    w1 = 1. / 10 / (eps + b1) ** 2
    w2 = 3. / 5 / (eps + b2) ** 2
    w3 = 3. / 10 / (eps + b3) ** 2
    sum_w = w1 + w2 + w3
    w1 /= sum_w
    w2 /= sum_w
    w3 /= sum_w
    interpolated_value = f1 * w1 + f2 * w2 + f3 * w3
    return interpolated_value


def interpolate_center_value_weno_to_face_upwind_interface(a, I, cl=1, cv_0=0., cv_n=0.):
    """
    interpolate the center value a[i] at the face res[i+1] (corresponding to the upwind scheme) on diphasic cells

    Args:
        cv_n:
        cv_0:
        cl: the limit condition, 1 is periodic
        a: the center values
        I: the phase indicator

    Returns:
        res
    """
    res = interpolate_form_center_to_face_weno(a, cl)
    # print('a : ', a.shape)
    # print('res : ', res.shape)
    center_values = np.empty(a.size + 5)
    phase_indicator = np.empty(I.size + 5)
    if cl == 1:
        center_values[:3] = a[-3:]
        center_values[3:-2] = a
        center_values[-2:] = a[:2]
        phase_indicator[:3] = I[-3:]
        phase_indicator[3:-2] = I
        phase_indicator[-2:] = I[:2]
        center_diph = (phase_indicator * (1. - phase_indicator) != 0.)
    elif cl == 0:
        center_values[:3] = cv_0
        center_values[3:-2] = a
        center_values[-2:] = cv_n
        # en cas d'utilisation du schéma avec des conditions aux limites on n'est pas diphasique aux bords
        phase_indicator[:3] = 0.
        phase_indicator[3:-2] = I
        phase_indicator[-2:] = 0.
        center_diph = (phase_indicator * (1. - phase_indicator) != 0.)
    else:
        raise NotImplementedError
    diph_jm2 = center_diph[:-4]
    diph_jm1 = center_diph[1:-3]
    diph_j = center_diph[2:-2]
    diph_jp1 = center_diph[3:-1]
    diph_jp2 = center_diph[4:]
    # f diphasic correspond aux faces dont le stencil utilisé par le WENO traverse l'interface
    f_diph = diph_jm2 | diph_jm1 | diph_j | diph_jp1 | diph_jp2

    # interpolation upwind
    res[f_diph] = center_values[2:-2][f_diph]
    return res


def grad(center_value, dx=1., cl=1):
    """
    Calcule le gradient aux faces
    :param center_value: globalement lambda
    :param cl: si cl = 1 les gradients aux bords sont périodiques
    :param dx: le delta x
    :return: le gradient aux faces
    """
    if len(center_value.shape) != 1:
        raise NotImplementedError
    gradient = np.zeros(center_value.shape[0] + 1)
    gradient[1:-1] = (center_value[1:] - center_value[:-1]) / dx
    if cl == 1:
        gradient[0] = (center_value[0] - center_value[-1]) / dx
        gradient[-1] = (center_value[0] - center_value[-1]) / dx
    if cl == 2:
        gradient[0] = 0
        gradient[-1] = 0
    return gradient


def grad_center(center_value, dx=1., cl=1):
    """
    Ce schéma calcule les gradients aux éléments, c'est expérimental pour avoir un lda_grad_T plus continu
    :param center_value:
    :param dx:
    :param cl:
    :return: les gradients au centres des cellules
    """
    if len(center_value.shape) != 1:
        raise NotImplementedError
    if cl == 1:
        center_extended = np.r_[center_value[-1], center_value, center_value[0]]
    else:
        raise NotImplementedError
    gradient = (center_extended[2:] - center_extended[:-2]) / (2. * dx)
    return gradient


# def indicatrice_liquide(x, markers=None):
#     i = np.ones_like(x)
#     if markers is None:
#         return i
#     dx = x[1] - x[0]
#     if markers[0] < markers[1]:
#         i[(x > markers[0]) & (x < markers[1])] = 0.
#     else:
#         i[(x > markers[0]) | (x < markers[1])] = 0.
#     diph0 = (np.abs(x - markers[0]) < dx / 2.)
#     i[diph0] = (markers[0] - x[diph0]) / dx + 1. / 2.
#     diph1 = (np.abs(x - markers[1]) < dx / 2.)
#     i[diph1] = -(markers[1] - x[diph1]) / dx + 1 / 2.
#     return i


class Bulles:
    def __init__(self, markers=None, phy_prop=None, n_bulle=None, Delta=1.):
        if markers is None:
            self.markers = []
            if n_bulle is None:
                if phy_prop.a_i is None:
                    raise Exception('On ne peut pas déterminer auto la géométrie des bulles sans le rapport surfacique')
                else:
                    # On détermine le nombre de bulle pour avoir une aire interfaciale donnée.
                    # On considère ici une géométrie 1D comme l'équivalent d'une situation 3D
                    n_bulle = int(phy_prop.a_i / 2. * phy_prop.Delta) + 1
            if phy_prop.alpha is None:
                raise Exception('On ne peut pas déterminer auto la géométrie des bulles sans le taux de vide')
            else:
                # Avec le taux de vide on en déduit le diamètre d'une bulle. On va considérer que le taux de vide
                # s'exprime en 1D, cad : phy_prop.alpha = n*d*dS/(Dx*dS)
                self.diam = phy_prop.alpha * phy_prop.Delta / n_bulle
                centers = np.linspace(self.diam, phy_prop.Delta + self.diam, n_bulle + 1)[:-1]
                for center in centers:
                    self.markers.append((center - self.diam / 2., center + self.diam / 2.))
                self.markers = np.array(self.markers)
        else:
            self.markers = np.array(markers)
            self.diam = None

        if phy_prop is not None:
            self.Delta = phy_prop.Delta
        else:
            self.Delta = Delta

        depasse = (self.markers > self.Delta) | (self.markers < 0.)
        if np.any(depasse):
            print('Delta : ', self.Delta)
            print('markers : ', self.markers)
            print('depasse : ', depasse)
            raise Exception('Les marqueurs dépassent du domaine')

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
        :param x: les positions des centres des mailles
        :return: l'indicatrice
        """
        i = np.ones_like(x)
        dx = x[1] - x[0]
        for markers in self.markers:
            if markers[0] < markers[1]:
                i[(x > markers[0]) & (x < markers[1])] = 0.
            else:
                i[(x > markers[0]) | (x < markers[1])] = 0.
            diph0 = (np.abs(x - markers[0]) < dx / 2.)
            i[diph0] = (markers[0] - x[diph0]) / dx + 1. / 2.
            diph1 = (np.abs(x - markers[1]) < dx / 2.)
            i[diph1] = -(markers[1] - x[diph1]) / dx + 1 / 2.
        return i

    def shift(self, dx):
        """
        On déplace les marqueurs vers la droite
        Args:
            dx: la distance du déplacement

        Returns:

        """
        self.markers += dx
        depasse = self.markers > self.Delta
        self.markers[depasse] -= self.Delta


class PhysicalProperties:
    def __init__(self, Delta=1., lda1=1., lda2=0., rho_cp1=1., rho_cp2=1., v=1., diff=1., a_i=358., alpha=0.06, dS=1.):
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
        if self._v == 0.:
            self._cas = 'diffusion'
        elif self._diff == 0.:
            self._cas = 'convection'
        else:
            self._cas = 'mixte'

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


class NumericalProperties:
    def __init__(self, dx=0.1, dt=1., cfl=1., fo=1., schema='weno', time_scheme='euler', phy_prop=None):
        if phy_prop is None:
            print('Attention : les valeurs par défaut ont été prises pour Delta et les autres params physiques')
            phy_prop = PhysicalProperties()
        self._cfl_lim = cfl
        self._fo_lim = fo
        self._schema = schema
        self._time_scheme = time_scheme
        self._dx_lim = dx
        nx = int(phy_prop.Delta / dx)
        dx = phy_prop.Delta / nx
        self._dx = dx
        self._x = np.linspace(dx / 2., phy_prop.Delta - dx / 2., nx)
        self._x_f = np.linspace(0., phy_prop.Delta, nx + 1)
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


class Problem:
    T: np.ndarray
    I: np.ndarray

    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None):
        if phy_prop is None:
            print('Attention, les propriétés physiques par défaut sont utilisées')
            phy_prop = PhysicalProperties()
        if num_prop is None:
            print('Attention, les propriétés numériques par défaut sont utilisées')
            num_prop = NumericalProperties()
        self.phy_prop = phy_prop
        self.num_prop = num_prop
        self.bulles = self._init_bulles(markers)
        self.T = T0(self.num_prop.x, markers=self.bulles, phy_prop=self.phy_prop)
        self.dt = self.get_time()
        self.time = 0.
        self.I = self.update_I()
        self.iter = 0

    def _init_bulles(self, markers=None):
        if markers is None:
            return Bulles(markers=markers, phy_prop=self.phy_prop)
        elif isinstance(markers, Bulles):
            return markers.copy()
        else:
            print(markers)
            raise NotImplementedError

    @property
    def name(self):
        if self.phy_prop.v == 0.:
            return 'Cas : %s, %s, %s, dx = %g, dt = %g' % (self.phy_prop.cas, self.num_prop.time_scheme,
                                                           self.num_prop.schema, self.num_prop.dx,
                                                           self.dt)
        elif self.phy_prop.diff == 0.:
            return 'Cas : %s, %s, %s, dx = %g, cfl = %g' % (self.phy_prop.cas, self.num_prop.time_scheme,
                                                            self.num_prop.schema, self.num_prop.dx,
                                                            self.cfl)
        else:
            return 'Cas : %s, %s, %s, dx = %g, dt = %g, cfl = %g' % (self.phy_prop.cas, self.num_prop.time_scheme,
                                                                     self.num_prop.schema,
                                                                     self.num_prop.dx, self.dt, self.cfl)

    @property
    def cfl(self):
        return self.phy_prop.v * self.dt / self.num_prop.dx

    @property
    def Lda_h(self):
        # lda = np.ones_like(self.num_prop.x)*self.phy_prop.lda1
        # lda[(self.num_prop.x > self.markers[0]) & (self.num_prop.x < self.markers[1])] = self.phy_prop.lda2
        return 1. / (self.I / self.phy_prop.lda1 + (1. - self.I) / self.phy_prop.lda2)

    @property
    def rho_cp_a(self):
        return self.I * self.phy_prop.rho_cp1 + (1. - self.I) * self.phy_prop.rho_cp2

    @property
    def rho_cp_h(self):
        return 1. / (self.I / self.phy_prop.rho_cp1 + (1. - self.I) / self.phy_prop.rho_cp2)

    def update_I(self):
        i = self.bulles.indicatrice_liquide(self.num_prop.x)
        return i

    def get_time(self):
        # nombre CFL = 1. par défaut
        if self.phy_prop.v > 10 ** (-15):
            dt_cfl = self.num_prop.dx / self.phy_prop.v * self.num_prop.cfl_lim
        else:
            dt_cfl = 10 ** 15
        # nombre de fourier = 1. par défaut
        dt_fo = self.num_prop.dx ** 2 / max(self.phy_prop.lda1, self.phy_prop.lda2) * \
            min(self.phy_prop.rho_cp2, self.phy_prop.rho_cp1) * self.num_prop.fo_lim
        # dt_fo = dx**2/max(lda1/rho_cp1, lda2/rho_cp2)*fo
        # minimum des 3
        list_dt = [self.num_prop.dt_min, dt_cfl, dt_fo]
        i_dt = np.argmin(list_dt)
        temps = ['dt min', 'dt cfl', 'dt fourier'][i_dt]
        dt = list_dt[i_dt]
        print(temps)
        print(dt)
        return dt

    @property
    def energy(self):
        return np.sum(self.rho_cp_a * self.T * self.phy_prop.dS * self.num_prop.dx)

    @property
    def energy_m(self):
        return np.sum(self.rho_cp_a * self.T * self.num_prop.dx) / self.phy_prop.Delta

    def update_markers(self):
        self.bulles.shift(self.phy_prop.v * self.dt)
        self.I = self.update_I()

    def timestep(self, n=None, t_fin=None, plot_for_each=1, number_of_plots=None, plotter=None, debug=None):
        if (n is None) and (t_fin is None):
            raise NotImplementedError
        if t_fin is not None:
            n = int(t_fin / self.dt)
        if number_of_plots is not None:
            plot_for_each = int((n - 1) / number_of_plots)
        if plot_for_each == 0:
            plot_for_each = 1
        plotter.plot(self)
        energy = np.zeros((n + 1,))
        t = np.linspace(0, n * self.dt, n + 1)
        energy[0] = self.energy
        for i in range(n):
            if self.num_prop.time_scheme is 'euler':
                self.euler_timestep(debug=debug, bool_debug=(i % plot_for_each == 0))
            elif self.num_prop.time_scheme is 'rk4':
                self.rk4_timestep(debug=debug, bool_debug=(i % plot_for_each == 0))
            self.update_markers()
            # TODO: vérifier que le rho cp correspond au rho cp np1 utilisé dans ProblemDiscontinu
            self.time += self.dt
            self.iter += 1
            energy[i + 1] = self.energy
            if (i % plot_for_each == 0) and (plotter is not None) and ((i != 0) or (n == 1)):
                plotter.plot(self)
        return t, energy

    def euler_timestep(self, debug=None, bool_debug=False):
        int_div_T_u = 1 / (self.phy_prop.dS * self.num_prop.dx) * \
            integrale_volume_div(self.T, self.phy_prop.v * np.ones((self.T.shape[0] + 1,)), I=self.I,
                                 dS=self.phy_prop.dS, schema=self.num_prop.schema)
        int_div_lda_grad_T = 1. / (self.phy_prop.dS * self.num_prop.dx) * \
            integrale_volume_div(self.Lda_h, grad(self.T, dx=self.num_prop.dx), I=self.I,
                                 dS=self.phy_prop.dS, schema=self.num_prop.schema)
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

    def rk4_timestep(self, debug=None, bool_debug=False):
        T_int = self.T.copy()
        K = [0.]
        pas_de_temps = np.array([0, 0.5, 0.5, 1.])
        for h in pas_de_temps:
            markers_int = self.bulles.copy()
            markers_int.shift(self.phy_prop.v * h * self.dt)
            temp_I = markers_int.indicatrice_liquide(self.num_prop.x)
            T = T_int + h * self.dt * K[-1]
            int_div_T_u = -1 / (self.phy_prop.dS * self.num_prop.dx) * \
                integrale_volume_div(T, self.phy_prop.v * np.ones((T.shape[0] + 1,)), I=temp_I,
                                     dS=self.phy_prop.dS, schema=self.num_prop.schema)
            Lda_h = 1. / (temp_I / self.phy_prop.lda1 + (1. - temp_I) / self.phy_prop.lda2)
            rho_cp_inv_h = temp_I / self.phy_prop.rho_cp1 + (1. - temp_I) / self.phy_prop.rho_cp2
            div_lda_grad_T = 1 / (self.phy_prop.dS * self.num_prop.dx) * \
                integrale_volume_div(Lda_h, grad(T, dx=self.num_prop.dx), I=temp_I,
                                     dS=self.phy_prop.dS, schema=self.num_prop.schema)
            rho_cp_inv_int_div_lda_grad_T = self.phy_prop.diff * rho_cp_inv_h * div_lda_grad_T
            K.append(int_div_T_u + rho_cp_inv_int_div_lda_grad_T)
            if (debug is not None) and bool_debug:
                debug.set_title('sous-pas de temps %f' % (len(K) - 2))
                debug.plot(self.num_prop.x_f,
                           interpolate_form_center_to_face_weno(Lda_h) * grad(T, dx=self.num_prop.dx),
                           label='lda_h grad T, time = %f' % self.time)
                debug.plot(self.num_prop.x, rho_cp_inv_h, label='rho_cp_inv_h, time = %f' % self.time)
                debug.plot(self.num_prop.x, div_lda_grad_T, label='div_lda_grad_T, time = %f' % self.time)
                maxi = max(np.max(div_lda_grad_T), np.max(rho_cp_inv_int_div_lda_grad_T), np.max(rho_cp_inv_h))
                mini = min(np.min(div_lda_grad_T), np.min(rho_cp_inv_int_div_lda_grad_T), np.min(rho_cp_inv_h))
                for markers in self.bulles():
                    debug.plot([markers[0] + self.phy_prop.v * h * self.dt] * 2, [mini, maxi], '--')
                    debug.plot([markers[1] + self.phy_prop.v * h * self.dt] * 2, [mini, maxi], '--')
                    debug.set_xticks(self.num_prop.x_f)
                    debug.grid(b=True, which='major')
                    debug.legend()
        coeff = np.array([1. / 6, 1 / 3., 1 / 3., 1. / 6])
        self.T += np.sum(self.dt * coeff * np.array(K[1:]).T, axis=-1)


class ProblemConserv(Problem):
    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None):
        super().__init__(T0, markers=markers, num_prop=num_prop, phy_prop=phy_prop)

    @property
    def name(self):
        return 'Forme conservative 1, ' + super().name

    def euler_timestep(self, debug=None, bool_debug=False):
        markers_int = self.bulles.copy()
        markers_int.shift(self.phy_prop.v * self.dt)
        Inp1 = markers_int.indicatrice_liquide(self.num_prop.x)
        rho_cp_np1 = self.phy_prop.rho_cp1 * Inp1 + self.phy_prop.rho_cp2 * (1. - Inp1)
        int_div_rho_cp_T_u = 1 / (self.phy_prop.dS * self.num_prop.dx) * \
            integrale_volume_div(self.rho_cp_a * self.T,
                                 self.phy_prop.v * np.ones((self.T.shape[0] + 1,)), I=self.I,
                                 dS=self.phy_prop.dS, schema=self.num_prop.schema)
        int_div_lda_grad_T = 1. / (self.phy_prop.dS * self.num_prop.dx) * \
            integrale_volume_div(self.Lda_h, grad(self.T, dx=self.num_prop.dx), I=self.I,
                                 dS=self.phy_prop.dS, schema=self.num_prop.schema)
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
        self.T = self.rho_cp_a * self.T / rho_cp_np1 + self.dt / rho_cp_np1 * (
                -int_div_rho_cp_T_u + self.phy_prop.diff * int_div_lda_grad_T)

    def rk4_timestep(self, debug=None, bool_debug=False):
        T_int = self.T.copy()
        K = [0.]
        pas_de_temps = np.array([0, 0.5, 0.5, 1.])
        for h in pas_de_temps:
            markers_int = self.bulles.copy()
            markers_int.shift(self.phy_prop.v * self.dt * h)
            temp_I = markers_int.indicatrice_liquide(self.num_prop.x)
            rho_cp_int = self.phy_prop.rho_cp1 * temp_I + self.phy_prop.rho_cp2 * (1. - temp_I)
            T = T_int + h * self.dt * K[-1]
            int_div_rho_cp_T_u = 1 / (self.phy_prop.dS * self.num_prop.dx) * \
                integrale_volume_div(rho_cp_int * T, self.phy_prop.v * np.ones((T.shape[0] + 1,)),
                                     dS=self.phy_prop.dS, schema=self.num_prop.schema)
            Lda_h = 1. / (temp_I / self.phy_prop.lda1 + (1. - temp_I) / self.phy_prop.lda2)
            # rho_cp_inv_h = temp_I / self.phy_prop.rho_cp1 + (1. - temp_I) / self.phy_prop.rho_cp2
            div_lda_grad_T = 1 / (self.phy_prop.dS * self.num_prop.dx) * \
                integrale_volume_div(Lda_h, grad(T, dx=self.num_prop.dx),
                                     dS=self.phy_prop.dS, schema=self.num_prop.schema)
            int_div_lda_grad_T = self.phy_prop.diff * div_lda_grad_T
            K.append(-int_div_rho_cp_T_u + int_div_lda_grad_T)
            if (debug is not None) and bool_debug:
                debug.set_title('sous-pas de temps %f' % (len(K) - 2))
                debug.plot(self.num_prop.x_f,
                           interpolate_form_center_to_face_weno(Lda_h) * grad(T, dx=self.num_prop.dx),
                           label='lda_h grad T, time = %f' % self.time)
                debug.plot(self.num_prop.x, div_lda_grad_T, label='div_lda_grad_T, time = %f' % self.time)
                maxi = max(np.max(div_lda_grad_T), np.max(int_div_lda_grad_T))
                mini = min(np.min(div_lda_grad_T), np.min(int_div_lda_grad_T))
                for markers in self.bulles():
                    debug.plot([markers[0] + self.phy_prop.v * h * self.dt] * 2, [mini, maxi], '--')
                    debug.plot([markers[1] + self.phy_prop.v * h * self.dt] * 2, [mini, maxi], '--')
                    debug.xticks(self.num_prop.x_f)
                    debug.grid(b=True, which='major')
                    debug.legend()
        markers_np1 = self.bulles.copy()
        markers_np1.shift(self.phy_prop.v * self.dt)
        temp_I = markers_np1.indicatrice_liquide(self.num_prop.x)
        rho_cp_np1 = self.phy_prop.rho_cp1 * temp_I + self.phy_prop.rho_cp2 * (1. - temp_I)
        coeff = np.array([1. / 6, 1 / 3., 1 / 3., 1. / 6])
        self.T = self.rho_cp_a * self.T / rho_cp_np1 + 1. / rho_cp_np1 * np.sum(self.dt * coeff * np.array(K[1:]).T,
                                                                                axis=-1)


class ProblemConserv2(Problem):
    def __init__(self, T0, markers=None, num_prop=None, phy_prop=None):
        super().__init__(T0, markers=markers, num_prop=num_prop, phy_prop=phy_prop)

    @property
    def name(self):
        return 'Forme conservative boniou, ' + super().name

    def euler_timestep(self, debug=None, bool_debug=False):
        markers_np1 = self.bulles.copy()
        markers_np1.shift(self.phy_prop.v * self.dt)
        int_div_rho_cp_u = 1 / (self.phy_prop.dS * self.num_prop.dx) * \
            integrale_volume_div(self.rho_cp_a, self.phy_prop.v * np.ones((self.T.shape[0] + 1,)),
                                 I=self.I,
                                 dS=self.phy_prop.dS, schema=self.num_prop.schema)
        rho_cp_etoile = self.rho_cp_a + self.dt * int_div_rho_cp_u
        int_div_rho_cp_T_u = 1 / (self.phy_prop.dS * self.num_prop.dx) * \
            integrale_volume_div(self.rho_cp_a * self.T,
                                 self.phy_prop.v * np.ones((self.T.shape[0] + 1,)), I=self.I,
                                 dS=self.phy_prop.dS, schema=self.num_prop.schema)
        int_div_lda_grad_T = 1. / (self.phy_prop.dS * self.num_prop.dx) * \
            integrale_volume_div(self.Lda_h, grad(self.T, dx=self.num_prop.dx), I=self.I,
                                 dS=self.phy_prop.dS, schema=self.num_prop.schema)
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
        self.T += self.dt / rho_cp_etoile * (int_div_rho_cp_u * self.T +
                                             (- int_div_rho_cp_T_u + self.phy_prop.diff * int_div_lda_grad_T))

    def rk4_timestep(self, debug=None, bool_debug=False):
        K = [np.array(0.)]  # type: list[np.ndarray]
        K_rhocp = [0.]
        pas_de_temps = np.array([0, 0.5, 0.5, 1.])
        for h in pas_de_temps:
            markers_int = self.bulles.copy()
            markers_int.shift(self.phy_prop.v * self.dt * h)
            temp_I = markers_int.indicatrice_liquide(self.num_prop.x)

            # On s'occupe de calculer d_rho_cp

            # rho_cp_markers = self.phy_prop.rho_cp1 * temp_I + self.phy_prop.rho_cp2 * (1. - temp_I)
            rho_cp = self.rho_cp_a + h * self.dt * K_rhocp[-1]
            int_div_rho_cp_u = 1 / (self.phy_prop.dS * self.num_prop.dx) * \
                integrale_volume_div(rho_cp, self.phy_prop.v * np.ones((self.T.shape[0] + 1,)), I=temp_I,
                                     dS=self.phy_prop.dS, schema=self.num_prop.schema)
            rho_cp_etoile = rho_cp - int_div_rho_cp_u * self.dt * h

            # On s'occupe de calculer d_rho_cp_T

            T = self.T + h * self.dt * K[-1]
            int_div_rho_cp_T_u = 1 / (self.phy_prop.dS * self.num_prop.dx) * \
                integrale_volume_div(rho_cp * T, self.phy_prop.v * np.ones((T.shape[0] + 1,)),
                                     I=temp_I,
                                     dS=self.phy_prop.dS, schema=self.num_prop.schema)
            Lda_h = 1. / (temp_I / self.phy_prop.lda1 + (1. - temp_I) / self.phy_prop.lda2)
            int_div_lda_grad_T = 1 / (self.phy_prop.dS * self.num_prop.dx) * \
                integrale_volume_div(Lda_h, grad(T, dx=self.num_prop.dx), I=temp_I,
                                     dS=self.phy_prop.dS, schema=self.num_prop.schema)
            int_div_lda_grad_T = self.phy_prop.diff * int_div_lda_grad_T
            K.append(1. / rho_cp_etoile * (T * int_div_rho_cp_u - int_div_rho_cp_T_u + int_div_lda_grad_T))

        coeff = np.array([1. / 6, 1 / 3., 1 / 3., 1. / 6])
        # d_rhocp = np.sum(self.dt * coeff * np.array(K_rhocp[1:]).T, axis=-1)
        d_rhocpT = np.sum(self.dt * coeff * np.array(K[1:]).T, axis=-1)
        # rho_cp_etoile = rho_cp_markers
        # rho_cp_etoile = self.rho_cp_a + d_rhocp
        self.T += d_rhocpT


def get_T(x, markers=None, phy_prop=None):
    if phy_prop is None:
        raise Exception
    else:
        lda_1 = phy_prop.lda1
        lda_2 = phy_prop.lda2
    dx = x[1] - x[0]
    Delta = x[-1] + dx / 2.
    if markers is None:
        marker = np.array([[0.25 * Delta, 0.75 * Delta]])
    elif isinstance(markers, Bulles):
        marker = markers.markers
    else:
        marker = markers.copy()

    if len(marker) > 1:
        raise Exception('Le cas pour plus d une bulle n est pas enore implémenté')
    marker = marker.squeeze()
    if marker[0] < marker[1]:
        m = np.mean(marker)
    else:
        m = np.mean([marker[0], marker[1] + Delta])
        if m > Delta:
            m -= Delta
    T1 = lda_2 * np.cos(2 * np.pi * (x - m) / Delta)
    w = opt.fsolve(
        lambda y: y * np.sin(2 * np.pi * y * (marker[0] - m) / Delta) - np.sin(2 * np.pi * (marker[0] - m) / Delta),
        np.array(1.))
    b = lda_2 * np.cos(2 * np.pi / Delta * (marker[0] - m)) - lda_1 * np.cos(2 * np.pi * w / Delta * (marker[0] - m))
    T2 = lda_1 * np.cos(w * 2 * np.pi * ((x - m) / Delta)) + b
    T = T1.copy()
    if marker[0] < marker[1]:
        bulle = (x > marker[0]) & (x < marker[1])
    else:
        bulle = (x < marker[1]) | (x > marker[0])
    T[bulle] = T2[bulle]
    T -= np.min(T)
    T /= np.max(T)
    return T


def get_T_creneau(x, markers=None, phy_prop=None):
    if phy_prop is None:
        raise Exception('Attetion, il faut des propriétés thermiques pour déterminer auto le nbre de bulles')
    if markers is None:
        markers = Bulles(markers=markers, phy_prop=phy_prop)
    T = 1. - markers.indicatrice_liquide(x)
    return T

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     Delta = 10.
#     dx = 0.2
#     lda_1 = 2.
#     lda_2 = 2.
#     rho_cp_1 = 1.
#     rho_cp_2 = 1.
#     markers = np.array([0.4 * Delta, 0.55 * Delta])
#     v = 0.
#     dt = 1.
#     fo = 0.5
#
#     t_fin = 1.
#     t_m = []
#     e_m = []
#     Dx = 10. ** np.linspace(-1.5, -1, 1)
#     Cfl = 10. ** np.linspace(-1, -0.5, 1)
#
#     Cas_test = itertools.product(Dx, Cfl)
#
#     for dx, cfl in Cas_test:
#         # x, T = get_T(dx=dx, Delta=Delta, lda_1=lda_1, lda_2=lda_2, markers=markers)
#         # x, T = get_T_creneau(dx=dx, Delta=Delta, markers=markers)
#
#         prob = Problem(Delta, dx, lda_1, lda_2, rho_cp_1, rho_cp_2, markers, get_T_creneau, v, dt, cfl, fo,
#                        diff=0., schema='upwind', time_scheme='rk4')
#         t, e = prob.timestep(n=10000, number_of_plots=3, debug=None)
#         t_m.append(t)
#         e_m.append(e)
#     i = 0
#     Cas_test = itertools.product(Dx, Cfl)
#     for dx, cfl in Cas_test:
#         debug.figure('energie')
#         debug.plot(t_m[i], e_m[i], label='dx = %f, cfl = %f' % (dx, cfl))
#         debug.legend()
#         dedt = (e_m[i][-1] - e_m[i][0]) / (t_m[i][-1] - t_m[i][0])
#         print('dx = %f, cfl = %f, dE/dt = %f' % (dx, cfl, dedt))
#         i += 1
#     debug.show()
