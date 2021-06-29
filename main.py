# This is a sample Python script.
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import itertools

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import fileinput


def integrale_volume_div(center_value, face_value, cl=1, dS=1., schema='center'):
    """
    Calcule le delta de convection aux bords des cellules
    :param center_value: les valeurs présentes aux centres des cellules
    :param face_value: les valeurs présentes aux faces des cellules
    :param cl: si cl = 1, on prend des gradients nuls aux bords du domaine
    :param dV: le volume de la cellule
    :return: le delta au centre
    """
    if len(center_value.shape) != 1 or len(face_value.shape) != 1:
        raise NotImplementedError
    if center_value.shape[0] != face_value.shape[0] - 1:
        print('Le tableau des valeurs aux faces ne correspond pas au tableau des valeurs au centre')
        print(center_value.shape, face_value.shape)
        raise NotImplementedError
    if schema is 'upwind':
        interpolated_value = interpolate_from_center_to_face_upwind(center_value, cl)
    elif schema is 'center':
        interpolated_value = interpolate_from_center_to_face_center(center_value, cl)
    elif schema is 'weno':
        interpolated_value = interpolate_form_center_to_face_weno(center_value, cl=1)
    else:
        raise NotImplementedError
    flux = interpolated_value*face_value*dS
    # plt.figure(1)
    # plt.plot(flux)
    # plt.show(block=False)
    delta_center_value = flux[1:] - flux[:-1]
    return dS*delta_center_value


def interpolate_from_center_to_face_center(center_value, cl=1):
    interpolated_value = np.zeros((center_value.shape[0]+1,))
    interpolated_value[1:-1] = (center_value[:-1] + center_value[1:])/2.
    if cl == 1:
        interpolated_value[0] = (center_value[0] + center_value[-1])/2.
        interpolated_value[-1] = (center_value[0] + center_value[-1])/2.
    elif cl == 2:
        interpolated_value[0] = interpolated_value[1]
        interpolated_value[-1] = interpolated_value[-2]
    else:
        raise NotImplementedError
    return interpolated_value


def interpolate_from_center_to_face_upwind(center_value, cl=1):
    interpolated_value = np.zeros((center_value.shape[0]+1,))
    interpolated_value[1:] = center_value
    if cl == 2:
        interpolated_value[0] = interpolated_value[1]
        interpolated_value[-1] = interpolated_value[-2]
    elif cl == 1:
        interpolated_value[0] = center_value[-1]
    else:
        raise NotImplementedError
    return interpolated_value


def interpolate_form_center_to_face_weno(a, cl=1):
    """
    Weno scheme
    :param a: the scalar value at the center of the cell
    :param f: the scalar value at the faces
    :param cl: conditions aux limites, cl = 1: périodicité
    :return: les valeurs interpolées aux faces de la face -1/2 à la face n+1/2
    """
    if cl==1:
        center_values = np.r_[a[-3:], a, a[:2]]
    else:
        raise NotImplementedError
    ujm2 = center_values[:-4]
    ujm1 = center_values[1:-3]
    uj = center_values[2:-2]
    ujp1 = center_values[3:-1]
    ujp2 = center_values[4:]
    f1 = 1./3*ujm2 - 7./6*ujm1 + 11./6*uj
    f2 = -1./6*ujm1 + 5./6*uj + 1./3*ujp1
    f3 = 1./3*uj + 5./6*ujp1 - 1./6*ujp2
    eps = np.array(10.**-6)
    b1 = 13./12*(ujm2 - 2*ujm1 + uj)**2 + 1./4 * (ujm2 - 4*ujm1 + 3*uj)**2
    b2 = 13./12*(ujm1 - 2*uj + ujp1)**2 + 1./4 * (ujm1 - ujp1)**2
    b3 = 13./12*(uj - 2*ujp1 + ujp2)**2 + 1./4 * (3*uj - 4*ujp1 + ujp2)**2
    w1 = 1./10 / (eps + b1)**2
    w2 = 3./5 / (eps + b2)**2
    w3 = 3./10 / (eps + b3)**2
    sum = w1 + w2 + w3
    w1 /= sum
    w2 /= sum
    w3 /= sum
    interpolated_value = f1*w1 + f2*w2 + f3*w3
    return interpolated_value


def grad(center_value, dx=1, cl=1):
    """
    Calcule le gradient aux faces
    :param center_value: globalement lambda
    :param cl: si cl = 1 les gradients aux bords sont nuls
    :param dx: le delta x
    :return: le gradient aux faces
    """
    if len(center_value.shape) != 1:
        raise NotImplementedError
    gradient = np.zeros(center_value.shape[0] + 1)
    gradient[1:-1] = (center_value[1:] - center_value[:-1])/dx
    if cl == 1:
        gradient[0] = (center_value[0] - center_value[-1])/dx
        gradient[-1] = (center_value[0] - center_value[-1])/dx
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
    gradient = (center_extended[2:] - center_extended[:-2])/(2.*dx)
    return gradient


def indicatrice_liquide(x, markers=None):
    i = np.ones_like(x)
    if markers is None:
        return i
    dx = x[1] - x[0]
    if markers[0] < markers[1]:
        i[(x > markers[0]) & (x < markers[1])] = 0.
    else:
        i[(x > markers[0]) | (x < markers[1])] = 0.
    diph0 = (np.abs(x - markers[0]) < dx / 2.)
    i[diph0] = (markers[0] - x[diph0]) / dx + 1. / 2.
    diph1 = (np.abs(x - markers[1]) < dx / 2.)
    i[diph1] = -(markers[1] - x[diph1]) / dx + 1 / 2.
    return i


class Problem:
    def __init__(self, Delta, dx, lda1, lda2, rho_cp1, rho_cp2, markers, T0, v, dt, cfl=1., fo=1., schema='center'):
        self.lda1 = lda1
        self.lda2 = lda2
        self.rho_cp1 = rho_cp1
        self.rho_cp2 = rho_cp2
        self.Delta = Delta
        # self.x = np.linspace(0, Delta, int(Delta/dx)+1)
        self.x = np.linspace(dx / 2., Delta - dx / 2., int(Delta / dx))
        self.x_f = np.linspace(0, Delta, int(Delta/dx)+1)
        dx = self.x[1] - self.x[0]
        self.dx = dx
        self.markers = np.array(markers)
        self.T = np.array(T0)
        self.v = v
        self.cfl = cfl
        self.dt = get_time(cfl, fo, dt, v, dx, rho_cp1, rho_cp2, lda1, lda2)
        self.schema = schema
        self.time = 0.

    @property
    def Lda_h(self):
        # lda = np.ones_like(self.x)*self.lda1
        # lda[(self.x > self.markers[0]) & (self.x < self.markers[1])] = self.lda2
        return 1. / (self.I / self.lda1 + (1. - self.I) / self.lda2)

    @property
    def rho_cp_a(self):
        return self.I * self.rho_cp1 + (1. - self.I)*self.rho_cp2

    @property
    def rho_cp_h(self):
        return 1. / (self.I / self.rho_cp1 + (1. - self.I) / self.rho_cp2)

    @property
    def I(self):
        i = indicatrice_liquide(self.x, self.markers)
        return i

    def update_markers(self):
        self.markers = self.markers + self.v*self.dt
        self.markers[self.markers > self.Delta] -= self.Delta

    def compute_energy(self):
        return np.sum(self.rho_cp_a*self.T) / np.size(self.T)

    def timestep(self, n=None, t_fin=None, plot_for_each=1, number_of_plots=None, diff=1., time_scheme='euler', plot=True, debug=False):
        if (n is None) and (t_fin is None):
            raise NotImplementedError
        if t_fin is not None:
            n = int(t_fin/self.dt)
        if number_of_plots is not None:
            plot_for_each = int(n / number_of_plots)
        dS = 1.
        energy = np.zeros((n+1,))
        t = np.linspace(0, n*self.dt, n+1)
        energy[0] = self.compute_energy()
        plt.figure('T, dx = %f, cfl = %f' % (self.dx, self.cfl))
        self.plt_T_I()
        plt.xticks(self.x_f)
        plt.grid(which='major')
        for i in range(n):
            if time_scheme is 'euler':
                self.euler_timestep(dS=dS, diff=diff, debug=(debug & (i % plot_for_each == 0)))
            elif time_scheme is 'rk4':
                self.rk4_timestep(dS=1., diff=diff, debug=(debug & (i % plot_for_each == 0)))
            self.time += self.dt
            energy[i+1] = self.compute_energy()
            if (i % plot_for_each == 0) and plot:
                plt.figure('T, dx = %f, cfl = %f' % (self.dx, self.cfl))
                self.plt_T_I()
        return t, energy

    def euler_timestep(self, dS=1., diff=1., debug=False):
        int_div_T_u = 1 / (dS * self.dx) * integrale_volume_div(self.T, self.v * np.ones((self.T.shape[0] + 1,)), dS=dS,
                                                                schema=self.schema)
        int_div_lda_grad_T = 1. / (dS * self.dx) * integrale_volume_div(self.Lda_h, grad(self.T, dx=self.dx), dS=dS)
        if debug:
            plt.figure()
            plt.plot(self.x, 1. / self.rho_cp_h, label='rho_cp_inv_h, time = %f' % self.time)
            plt.plot(self.x, int_div_lda_grad_T, label='div_lda_grad_T, time = %f' % self.time)
            plt.xticks(self.x_f)
            plt.grid(which='major')
            maxi = max(np.max(int_div_lda_grad_T), np.max(1./self.rho_cp_h))
            mini = min(np.min(int_div_lda_grad_T), np.min(1./self.rho_cp_h))
            plt.plot([self.markers[0]] * 2, [mini, maxi], '--')
            plt.plot([self.markers[1]] * 2, [mini, maxi], '--')
            plt.legend()
        self.T += self.dt * (-int_div_T_u + diff * 1. / self.rho_cp_h * int_div_lda_grad_T)
        self.update_markers()

    def rk4_timestep(self, dS=1., diff=1., debug=False):
        T_int = self.T.copy()
        K = [0.]
        pas_de_temps = np.array([0, 0.5, 0.5, 1.])
        for h in pas_de_temps:
            T = T_int + h*self.dt*K[-1]
            int_div_T_u = -1 / (dS * self.dx) * integrale_volume_div(T, self.v * np.ones((T.shape[0] + 1,)), dS=dS,
                                                                     schema=self.schema)
            temp_I = indicatrice_liquide(self.x, self.markers + self.v*h*self.dt)
            Lda_h = 1./(temp_I/self.lda1 + (1. - temp_I)/self.lda2)
            rho_cp_inv_h = temp_I/self.rho_cp1 + (1.-temp_I)/self.rho_cp2
            div_lda_grad_T = 1/(dS*self.dx) * integrale_volume_div(Lda_h, grad(T, dx=self.dx), dS=dS,
                                                                   schema=self.schema)
            rho_cp_inv_int_div_lda_grad_T = diff * rho_cp_inv_h * div_lda_grad_T
            K.append(int_div_T_u + rho_cp_inv_int_div_lda_grad_T)
            if debug:
                plt.figure('sous-pas de temps %f' % (len(K)-2))
                # plt.plot(self.x, T, label='T')
                # plt.plot(self.x, Lda_h, label='lda_h')
                # plt.plot(self.x_f, grad(T, dx=self.dx), label='grad T')
                plt.plot(self.x_f, interpolate_form_center_to_face_weno(Lda_h)*grad(T, dx=self.dx), label='lda_h grad T, time = %f' % self.time)
                # Lda_a = temp_I*self.lda1 + (1.-temp_I)*self.lda2
                # plt.plot(self.x_f, interpolate_form_center_to_face_weno(Lda_a)*grad(T, dx=self.dx), label='lda_a grad T')
                plt.plot(self.x, rho_cp_inv_h, label='rho_cp_inv_h, time = %f' % self.time)
                plt.plot(self.x, div_lda_grad_T, label='div_lda_grad_T, time = %f' % self.time)
                # plt.plot(self.x, rho_cp_inv_int_div_lda_grad_T, label='rho_cp_inv_div_lda_grad_T, time = %f' % self.time)
                maxi = max(np.max(div_lda_grad_T), np.max(rho_cp_inv_int_div_lda_grad_T), np.max(rho_cp_inv_h))
                mini = min(np.min(div_lda_grad_T), np.min(rho_cp_inv_int_div_lda_grad_T), np.min(rho_cp_inv_h))
                plt.plot([self.markers[0]+self.v*h] * 2, [mini, maxi], '--')
                plt.plot([self.markers[1]+self.v*h] * 2, [mini, maxi], '--')
                plt.xticks(self.x_f)
                plt.grid(b=True, which='major')
                plt.legend()
        coeff = np.array([1./6, 1/3., 1/3., 1./6])
        self.T += np.sum(self.dt*coeff*np.array(K[1:]).T, axis=-1)
        self.update_markers()

    def plt_T_I(self):
        c = plt.plot(self.x, self.I, '+')
        col = c[-1].get_color()
        plt.plot(self.x, decale_perio(self.x, self.T, self.time*v), c=col, label='time %f' % self.time)
        # x, T = get_T(self.dx, self.Delta, self.lda1, self.lda2, self.markers)
        # plt.plot(x, T, '--', c=col, label='solution ana, time %f' % self.time)
        # maxi = max(np.max(self.T), np.max(self.I))
        # mini = min(np.min(self.T), np.min(self.I))
        # plt.plot([self.markers[0]]*2, [mini, maxi], '--', c=col)
        # plt.plot([self.markers[1]]*2, [mini, maxi], '--', c=col)
        plt.legend()
        # plt.show(block=False)


def get_time(cfl, fo, dt, v, dx, rho_cp1, rho_cp2, lda1, lda2):
    # nombre CFL = 0.5
    if v > 10**(-15):
        dt_cfl = dx/v*cfl
    else:
        dt_cfl = 10**15
    # nombre de fourier = 0.5
    dt_fo = dx**2/max(lda1, lda2)*min(rho_cp2, rho_cp1)*fo
    # dt_fo = dx**2/max(lda1/rho_cp1, lda2/rho_cp2)*fo
    # minimum des 3
    dt = min(dt, dt_cfl, dt_fo)
    print(dt)
    return dt


def decale_perio(x, T, x0=0.):
    """
    décale de x0 vers la gauche la courbe T en interpolant entre les décalages direct de n*dx < x0 < (n+1)*dx
    avec la formule suivante : T_interp = T_n * a + T_np1 * (1 - a) avec a = (x0 - n*dx)/dx
    :param x:
    :param T:
    :param x0:
    :return: T decalé
    """
    dx = x[1] - x[0]
    Delta = x[-1] + dx/2.
    while x0 > Delta:
        x0 -= Delta
    n = int(x0/dx)
    a = (x0 - n*dx)/dx
    T_n = np.r_[T[n:], T[:n]]
    T_np1 = np.r_[T[n+1:], T[:n+1]]
    T_decale = a*T_n + (1.-a)*T_np1
    return T_decale

def get_T(dx=0.1, Delta=10., lda_1=1., lda_2=1., markers=None):
    if markers is None:
        markers = np.array([0.25*Delta, 0.75*Delta])
    m = np.mean(markers)
    x = np.linspace(dx / 2., Delta - dx / 2., int(Delta / dx))
    T1 = lda_2 * np.cos(2*np.pi*(x - m)/Delta)
    w = opt.fsolve(lambda y: y*np.sin(2*np.pi*y*(markers[0] - m)/Delta) - np.sin(2*np.pi*(markers[0]-m)/Delta), np.array(1.))
    b = lda_2 * np.cos(2*np.pi/Delta*(markers[0]-m)) - lda_1 * np.cos(2*np.pi*w/Delta*(markers[0]-m))
    T2 = lda_1 * np.cos(w*2*np.pi*((x - m)/Delta)) + b
    T = T1.copy()
    if markers[0] < markers[1]:
        T[(x > markers[0]) & (x < markers[1])] = T2[(x > markers[0]) & (x < markers[1])]
    else:
        T[(x < markers[1]) | (x > markers[0])] = T2[(x > markers[0]) | (x < markers[1])]
    return x, T


def get_T_creneau(dx=0.1, Delta=10., markers=None):
    if markers is None:
        markers = np.array([0.25*Delta, 0.75*Delta])
    x = np.linspace(dx / 2., Delta - dx / 2., int(Delta / dx))
    T = 1. - indicatrice_liquide(x, markers)
    return x, T

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Delta = 10.
    dx = 0.2
    lda_1 = 2.
    lda_2 = 2.
    rho_cp_1 = 1.
    rho_cp_2 = 1.
    markers = np.array([0.4*Delta,0.55*Delta])
    v = 1.
    dt = 1.
    fo = 5000.

    t_fin = 1.
    tl = []
    el = []
    Dx = 10.**np.linspace(-1.5, -1, 1)
    Cfl = 10.**np.linspace(-1, -0.5, 1)

    Cas_test = itertools.product(Dx, Cfl)

    for dx, cfl in Cas_test:
        #x, T = get_T(dx=dx, Delta=Delta, lda_1=lda_1, lda_2=lda_2, markers=markers)
        x, T = get_T_creneau(dx=dx, Delta=Delta, markers=markers)

        prob = Problem(Delta, dx, lda_1, lda_2, rho_cp_1, rho_cp_2, markers, T, v, dt, cfl, fo, schema='weno')
        t, e = prob.timestep(n=10000, number_of_plots=3, diff=0., time_scheme='rk4', debug=True)
        tl.append(t)
        el.append(e)
    i = 0
    Cas_test = itertools.product(Dx, Cfl)
    for dx, cfl in Cas_test:
        plt.figure('energie')
        plt.plot(tl[i], el[i], label='dx = %f, cfl = %f' % (dx, cfl))
        plt.legend()
        dedt = (el[i][-1] - el[i][0])/(tl[i][-1] - tl[i][0])
        print('dx = %f, cfl = %f, dE/dt = %f' % (dx, cfl, dedt))
        i += 1
    plt.show()
