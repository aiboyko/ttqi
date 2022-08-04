"""this file stores functions related to the original Gorodetsky's style TT-VI algorithm."""
# self here means instance of Solver

# standard
import numpy as np
import warnings

# non-standard
import tt

# local
import mytt


def local_Q(self, x):
    # print('local_Q: x=%s'%x)
    Ds = self.dynsys.S.D
    Da = self.dynsys.A.D
    Vx = mytt.fast_slice_cross(x, self.V).reshape([-1] + [1] * Da)
    # print('local_Q: Vx = %s'%Vx)
    Q_ans = 0
    Qh = 0
    Bx_plus = [None] * Ds
    Bx_minus = [None] * Ds
    for i_d in range(Ds):
        Bx = mytt.fast_slice_cross(
            x, self.Bi[i_d]
        )  # in the current implementation B is taken from the tensor!
        Bx_plus[i_d] = mytt.pos(Bx) / self.dynsys.S.h[i_d]
        Bx_minus[i_d] = mytt.neg(Bx) / self.dynsys.S.h[i_d]
        if self.fastmixing:
            Qh += Bx_plus[i_d] + Bx_minus[i_d] + 1e-10
    if self.fastmixing:
        dts = 1.0 / Qh  # dts is computed locally
    del Qh
    Px_stay = np.ones_like(Bx)
    del Bx
    for i_d in range(Ds):

        # (+)
        if self.fastmixing:
            Px_plus = Bx_plus[i_d] * dts
        else:
            Px_plus = Bx_plus[i_d] * self.dt
        # applying BCS
        if not self.dynsys.S.BCS_PERIODIC[i_d]:
            Px_plus[x[:, i_d] == self.dynsys.S.N_cores.T[0][i_d] - 1] = 0
        # locally extracting shifted Value function (in numpy format)
        Vx_plus = mytt.fast_slice_cross(
            x, mytt.roll(self.V, axis=i_d, shift=-1)
        ).reshape([-1] + [1] * Da)
        Q_ans += Vx_plus * Px_plus
        Px_stay -= Px_plus
        del Vx_plus
        del Px_plus

        # (-)
        if self.fastmixing:
            Px_minus = Bx_minus[i_d] * dts
        else:
            Px_minus = Bx_minus[i_d] * self.dt
        # applying BCS
        if not self.dynsys.S.BCS_PERIODIC[i_d]:
            Px_minus[x[:, i_d] == 0] = 0
        # locally extracting shifted Value function (in numpy format)
        Vx_minus = mytt.fast_slice_cross(
            x, mytt.roll(self.V, axis=i_d, shift=1)
        ).reshape([-1] + [1] * Da)
        Q_ans += Vx_minus * Px_minus
        Px_stay -= Px_minus
        del Vx_minus
        del Px_minus
    # (stay)
    Q_ans += Vx * Px_stay
    Q_ans *= self.dynsys.gamma
    if self.fastmixing:
        # ttR is always actually Rdts
        Q_ans += mytt.fast_slice_cross(x, self.R)
    else:
        if self.fastmixing:
            # homunculus hack made for rectifying the fact that self.R in slowmix is done with constant dt
            Q_ans += mytt.fast_slice_cross(x, self.R) / self.dt * dts
        else:
            Q_ans += mytt.fast_slice_cross(x, self.R)
    return Q_ans


def fast_slice_V2V(self, x, V=None):
    Da = self.dynsys.A.D
    Q_ans = self.local_Q(x=x)
    V_ans = np.max(Q_ans, axis=tuple(range(-Da, 0)))
    return V_ans


def _VfromV(self, eps=1e-3, rounding=False, verbose=False):
    # this is alternative to the pipeline VfromQ-QfromV
    V = tt.multifuncrs2(
        self.ttMESH_Sidx,
        self.fast_slice_V2V,
        nswp=3,
        kickrank=3,
        eps=eps,
        y0=self.V,
        verb=verbose,
        do_qr=False,
    )

    if self.dynsys.absorber is not None:
        V *= self.ANTI_ABSORB_MASK_S

    if rounding:
        V = V.round(eps)
    return V


def _updateVfromV(self, eps=1e-3, rounding=False, verbose=False):
    self.V = self._VfromV(eps=eps, rounding=rounding, verbose=verbose)


def fast_slice_V2Pi(self, x, V=None):
    warnings.warn("Extracting policy by argmax is dangerous")
    Ds = self.dynsys.S.D
    Da = self.dynsys.A.D
    s_size = x.shape[0]
    Q_ans = self.local_Q(x=x)
    # here the argmax should be
    Q_ans = Q_ans.reshape(s_size, -1)
    Pi_ans = np.argmax(Q_ans, axis=1)

    return Pi_ans
