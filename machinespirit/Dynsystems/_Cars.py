import machinespirit as ms
import numpy as np
from ._Generic import GenericDynsys


class car3(GenericDynsys):
    def __init__(
        self, sigma2_fun=None, absorber=None, gamma=1.0 - 1e-3, S=None, A=None
    ):
        if not S:
            S = ms.Box(
                [[-100, 100], [-100, 100], [-np.pi, np.pi]],
                BCS_PERIODIC=[0, 0, 1],
                N_cores=[[71], [61], [51]],
            )

        if not A:
            A = ms.Box([[0, 1], [-np.pi / 3, np.pi / 3]], [[2], [21]])

        def b_fun(s, a):
            L = 10
            x, y, phi = s
            u, alpha = a
            return np.array([u * np.cos(phi), u * np.sin(phi), u / L * np.tan(alpha)])

        def R_fun(s, a, dynsys=None):  # per unit of time!
            """
            s is Ds x B, where Ds - dimensionality of S, B - batch size;
            a is Da x B, where Da - dimensionality of A, B - batch size;
            """

            if dynsys:
                s = s.reshape(dynsys.S.D, -1)
                a = a.reshape(dynsys.A.D, -1)

            x, y, phi = s
            u, alpha = a
            cx = 1e-4
            cy = 1e-4
            cphi = 1e-5
            cu = 1e-3
            timecost = 0
            return -cx * x ** 2 - cy * y ** 2 - cphi * phi ** 2 - cu * u ** 2 - timecost

        self = super().__init__(
            S=S,
            A=A,
            b_fun=b_fun,
            R_fun=R_fun,
            sigma2_fun=None,
            absorber=None,
            gamma=gamma,
        )


class car4(GenericDynsys):
    def __init__(
        self, sigma2_fun=None, absorber=None, gamma=1.0 - 1e-3, S=None, A=None
    ):
        if not S:
            S = ms.Box(
                [[-100, 100], [-100, 100], [-100, 100], [-np.pi, np.pi]],
                BCS_PERIODIC=[0, 0, 0, 1],
                N_cores=[[59], [57], [55], [53]],
            )

        if not A:
            A = ms.Box([[0, 1], [-np.pi / 3, np.pi / 3]], [[2], [3]])

        def b_fun(s, a):
            L = 10
            x, y, v, phi = s
            u, alpha = a
            return np.array(
                [v * np.cos(phi), v * np.sin(phi), u, v / L * np.tan(alpha)]
            )

        def R_fun(s, a):  # per unit of time!
            x, y, v, phi = s
            u, alpha = a
            c_coord = 1
            c_vel = 0.1
            cphi = 0.1
            cu = 0.01
            timecost = 1e-5

            return (
                -c_coord * x ** 2
                - c_coord * y ** 2
                - cphi * phi ** 2
                - cu * u ** 2
                - c_vel * v ** 2
                - timecost
            )

        self = super().__init__(
            S=S,
            A=A,
            b_fun=b_fun,
            R_fun=R_fun,
            sigma2_fun=None,
            absorber=None,
            gamma=gamma,
        )
