import numpy as np
import sdeint
import matplotlib.pyplot as plt
import mytt


class GenericDynsys:
    # Dynsys is a mathematical, not numerical, representation of the system, reward and the corresponding statespace.
    # It only stores vectorial entities of S,A,b,sigma2, Rfun,
    # Absorber_indicator_fun
    def __init__(self, S, A, b_fun, R_fun, sigma2_fun=None, absorber=None, gamma=1.0):
        self.S = S
        self.A = A
        self.b_fun = b_fun
        self.R_fun = R_fun  # rfun is s,a, not sas
        self.absorber = absorber
        self.type = "SDE"  # or 'gym'
        self.gamma = gamma

        if absorber is not None:
            if callable(absorber):  # if absorber is function, do nothing
                self.absorber_fun = absorber
            else:  # if absorber is an array, cook a function and attach it immidiately
                print("cooking absorber function from boundaries")
                absorb_region_bounds = np.array(absorber)  # [[-.9,0],[0,.75]])

                def hat(xyz, a):
                    # print('xyz=%s'%xyz)
                    buf = np.zeros_like(xyz)
                    for i in range(self.S.D):
                        # print(i)
                        buf[i] = (
                            mytt.softsign(xyz[i] - absorb_region_bounds[i, 0], a)
                            * mytt.softsign(absorb_region_bounds[i, 1] - xyz[i], a)
                            + 1e-14
                        )

                    # print('buf=%s'%buf)
                    return np.prod(buf, axis=0)

                self.absorber_fun = hat

        # UNCKECKED! In this part maybe it will work, maybe not
        # Сигма, это, конечно, скаляр Но ДЛЯ КАЖДОЙ КОМПОНЕНТЫ ОТДЕЛЬНЫЙ. И это
        # если диагональная.
        if sigma2_fun is None:

            def s2(s0, a):
                buf = np.array([np.zeros_like(s0[0])])
                return np.array([np.zeros_like(s0[0])] * S.D)

            self.sigma2_fun = s2

        elif isinstance(sigma2_fun, _numbers.Number):
            sigma2const = copy.copy(sigma2_fun)

            def s2(s0, a):
                buf = np.array([np.ones_like(s0[0]) * sigma2const] * S.D)
                return buf

            self.sigma2_fun = s2

        else:
            self.sigma2_fun = sigma2_fun

    def policyfun_zero(self, s):
        return np.zeros(self.A.D, dtype=np.float)

    def CreateTrajectory(
        self,
        s0,
        policyfun=None,
        testnoise=0,
        timespan=10,
        plot_timeseries=False,
        Nstoch=1,
        sde_dt=None,
    ):
        # policy_fun is such that u=policy_fun(s)  such that b(s, policy_fun(s))
        # does not crash
        if policyfun is None:
            print("No policy is specified. Using policy of zeros")
            policyfun = self.policyfun_zero

        if sde_dt is None:
            # danger danger
            sde_dt = timespan / 1000.0

        if self.type == "SDE":
            TsSDE = np.arange(0, timespan, sde_dt, dtype=np.float64)
            trajectories = []
            b = self.b_fun

            for i in range(Nstoch):
                trajectories.append(
                    sdeint.itoSRI2(
                        lambda s, t: b(s, policyfun(s), dtype=np.float),
                        lambda s, t: testnoise * np.eye(s.shape[0]),
                        np.array(s0, dtype=np.float),
                        TsSDE,
                    )
                )

            trajectories = np.array(trajectories)

            if plot_timeseries:
                self.PlotTimeseries(trajectories, TsSDE)

            return trajectories, TsSDE
        else:
            raise NotImplementedError(
                "Simulator for non-SDE envinronments is not implemented"
            )

    def PlotTimeseries(self, trajectories, times, fignum=0):
        N = trajectories.shape[0]
        fig = plt.figure(fignum)
        plt.xlabel = "Time, s"
        plt.ylabel = "State"

        for i in range(N):
            plt.plot(times, trajectories[i])

    def AnimateTrajectory(self, times, states, controls=None):
        #           """Raise error as certain methods need to be over-rided."""
        raise NotImplementedError("method needs to be defined by sub-class")

    def fun_wrapper4cross(self, s, a, fun):
        # this is supposed to make vanilla bfuns and Rfuns cross-friendly
        Ds = self.S.D
        Da = self.S.D
        s = s.reshape(self.S.D, -1)
        a = a.reshape(self.A.D, -1)
