import numpy as np

NCONST = 51


class Box:
    # vectorial entity of D-dimensional box on a grid
    @property
    def L(self):
        return np.array(self.bounds[:, 1] - self.bounds[:, 0], dtype=np.float).T

    @property
    def D(self):
        return self.L.shape[0]

    @property
    def N_i(self):
        buf = np.zeros([self.D], dtype=int)
        for i in range(self.D):
            buf[i] = np.prod(self.N_cores[i])
        return buf

    @property
    def h(self):
        buf = np.zeros([self.D], dtype=float)

        for i in range(self.D):
            buf[i] = np.array(
                (self.bounds[i, 1] - self.bounds[i, 0]) / (self.N_i[i] - 1),
                dtype=np.float,
            )
        return buf

    @property
    def N(self):
        return np.prod(self.N_i)

    def __init__(self, bounds, N_cores=None, BCS_PERIODIC=None):
        self.bounds = np.array(bounds)
        if N_cores == None:
            self.N_cores = np.array([[NCONST] * self.D], dtype=int).T
        elif type(N_cores) == int:
            self.N_cores = np.array([[N_cores] * self.D], dtype=int).T
        else:
            self.N_cores = np.array(N_cores)

        if BCS_PERIODIC == None:
            self.BCS_PERIODIC = np.zeros(self.D, dtype=bool)
        else:
            self.BCS_PERIODIC = np.array(BCS_PERIODIC)

        # for i_d in range(self.D):
        #     if self.BCS_PERIODIC[i_d]:
        #         self.N_cores[i_d, :] -= 1 # this is done to maintain pure zero while having assymetric grid (to avoid double of boundaries)
        #         #liek, [-2, -1, 0, 1, +2] -> [-2, -1, 0, 1]

    def point2idx(self, point):
        point = point.reshape(self.D, -1)
        lower_bounds = self.bounds[:, 0].reshape(-1, 1)
        L = self.L.reshape(-1, 1)
        h = self.h.reshape(-1, 1)
        mask = self.BCS_PERIODIC.reshape(-1, 1)
        point_per = mask * point
        point_nonper = (1 - mask) * point
        point_per = (point_per - lower_bounds) % L + lower_bounds
        point_nonper = np.clip(
            point_nonper,
            a_min=self.bounds[:, 0].reshape(-1, 1),
            a_max=self.bounds[:, 1].reshape(-1, 1),
        )
        point = point_per + point_nonper
        indices = np.asarray((point - lower_bounds) / h, dtype=int)
        return indices

    # def idx2point(self, idx):
    #     h = self.h.reshape(self.D, -1)
    #     idx = idx.reshape(self.D, -1)
    #     bounds = self.bounds[:, 0].reshape(self.D, -1)
    #     point = bounds + idx * h
    #     return point.reshape(-1, self.D)

    def idx2point(self, idx, reshape=False):
        if reshape:
            h = self.h.reshape(self.D, -1)
            idx = idx.reshape(self.D, -1)
            bounds = self.bounds[:, 0].reshape(self.D, -1)
            point = bounds + idx * h
            return point.reshape(-1, self.D)
        else:
            h = self.h.reshape(self.D, -1)
            # idx = idx.reshape(self.D, -1)
            bounds = self.bounds[:, 0].reshape(self.D, -1)
            point = bounds + idx * h
            return point