import machinespirit as ms
import numpy as np
from ._Generic import GenericDynsys


def b_fun(s, a, dynsys=None):
    if dynsys:
        s = s.reshape(dynsys.S.D, -1)
        a = a.reshape(dynsys.A.D, -1)
    phi, phidot = s

    return np.array([phidot, a - np.sin(phi)])


def quadratic_R_fun(s, a, dynsys=None):  # per unit of time!
    # print('quardatic R_fun _Pendulums: pre-reshape')
    # print('s = %s'%s)
    # print('a = %s'%a)

    if dynsys:
        s = s.reshape(dynsys.S.D, -1)
        a = a.reshape(dynsys.A.D, -1)

    # print('post-reshape')
    # print('s = %s'%s)
    # print('a = %s'%a)

    x, p = s
    cx = 1
    cp = 0.8
    ca = 0.01
    timecost = 0.0
    ans = -cx * (x) ** 2 - cp * p ** 2 - ca * a ** 2 - timecost
    return ans.reshape(-1, 1)


A = ms.Box([[-0.3, 0.3]], [[51]])


class pend1(GenericDynsys):
    def __init__(self, R_fun=None, sigma2_fun=None, absorber=None, gamma=1.0 - 1e-3):
        S = ms.Box(
            [[-np.pi, np.pi], [-np.pi, np.pi]],
            BCS_PERIODIC=[1, 0],
            N_cores=[[151], [151]],
        )
        if R_fun is None:
            R_fun = quadratic_R_fun
        self = super().__init__(
            S=S,
            A=A,
            b_fun=b_fun,
            R_fun=R_fun,
            sigma2_fun=sigma2_fun,
            absorber=absorber,
            gamma=gamma,
        )


class pend3(GenericDynsys):
    def __init__(self, R_fun=None, sigma2_fun=None, absorber=None, gamma=1.0 - 1e-3):
        S = ms.Box(
            [[-3 * np.pi, 3 * np.pi], [-np.pi, np.pi]],
            BCS_PERIODIC=[0, 0],
            N_cores=[[301], [101]],
        )
        if R_fun is None:
            R_fun = quadratic_R_fun
        self = super().__init__(
            S=S,
            A=A,
            b_fun=b_fun,
            R_fun=R_fun,
            sigma2_fun=sigma2_fun,
            absorber=absorber,
            gamma=gamma,
        )
