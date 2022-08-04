"""this file stores functions related to TT-QI algorithm."""

import numpy as np
import mytt
import tt
import warnings
from tt.optimize import tt_min

SMOOTH_CONST = 1e-3


def make_B_SIGMA2_tensors(self, verbose=False):
    # For slowmixing
    print("-- Making B and SIGMA2 TT-vectors")
    if self.ttMESH_S is None:
        print("-- TT meshgrid is not found, making it now:")
        self.make_grid()
    Da = self.dynsys.A.D
    Ds = self.dynsys.S.D
    b = self.dynsys.b_fun
    sigma2 = self.dynsys.sigma2_fun
    self.Bi = []
    self.SIGMA2i = []
    for i in range(Ds):
        self.Bi.append(
            tt.multifuncrs2(
                self.ttMESH_SA,
                lambda x: b(x[:, 0:Ds].T, x[:, Ds : Ds + Da].T)[i],
                eps=self.tol,
                verb=verbose,
            ).round(eps=self.tol)
        )
        self.SIGMA2i.append(
            tt.multifuncrs2(
                self.ttMESH_SA,
                lambda x: sigma2(x[:, 0:Ds].T, x[:, Ds : Ds + Da].T)[i],
                eps=self.tol,
                verb=verbose,
            ).round(eps=self.tol)
        )


def _make_Bplus_minus(self, alpha=1e4, verbose=True):
    # For slowmixing
    Ds = self.dynsys.S.D
    if verbose:
        print("Making B+-")
    self.B_plus = []
    self.B_minus = []
    for i in range(Ds):
        self.B_plus.append(
            tt.multifuncrs2(
                [self.Bi[i]],
                lambda x: mytt.softsign(alpha * x) * x,
                eps=self.tol,
                verb=verbose,
            ).round(eps=self.tol)
        )
        self.B_minus.append(
            -tt.multifuncrs2(
                [self.Bi[i]],
                lambda x: (1 - mytt.softsign(alpha * x)) * x,
                eps=self.tol,
                verb=verbose,
            ).round(eps=self.tol)
        )


def _make_absorber(self, ABSORBER_IS_SMOOTH=False):
    Da = self.dynsys.A.D
    Na_cores = self.dynsys.A.N_cores
    if self.dynsys.absorber is not None:
        if ABSORBER_IS_SMOOTH:
            for i in range(
                10
            ):  # gradually tightening the exponential tail of the indicator (ot lowering the Fermi temperature)
                inverse_temperature = 10 * 2 ** i

                if i == 0:
                    self.ABSORB_MASK_S = tt.multifuncrs2(
                        self.ttMESH_S,
                        lambda x: self.dynsys.absorber_fun(x.T, inverse_temperature),
                        verb=False,
                    ).round(self.tol)
                else:
                    self.ABSORB_MASK_S = tt.multifuncrs2(
                        self.ttMESH_S,
                        lambda x: self.dynsys.absorber_fun(x.T, inverse_temperature),
                        verb=False,
                        y0=self.ABSORB_MASK_S,
                    ).round(self.tol)
            l_of_ones = []
            for i in range(Da):
                l_of_ones.append(tt.ones(Na_cores[i]))

            self.ABSORB_MASK_S = self.ABSORB_MASK_S * (
                -1 / tt.optimize.tt_min.min_tens(-self.ABSORB_MASK_S, verb=False)[0]
            )

            self.ABSORB_MASK_SA = tt.mkron([self.ABSORB_MASK_S] + l_of_ones)
            self.ABSORB_MASK_SA = self.ABSORB_MASK_SA * (
                -1 / tt.optimize.tt_min.min_tens(-self.ABSORB_MASK_SA, verb=False)[0]
            )

            self.ANTI_ABSORB_MASK_S = (self.tt_ones_S - self.ABSORB_MASK_S).round(1e-10)
            self.ANTI_ABSORB_MASK_SA = (self.tt_ones_SA - self.ABSORB_MASK_SA).round(
                1e-10
            )
        else:
            # raise NotImplementedError(
            #     'Absorber as an analytic binary TT tensor is not implemented. See TT-Q ND.pynb')

            absorb_region_bounds = (
                self.dynsys.absorber
            )  # np.array([[np.pi-.1, np.pi+.1],[-.1,.1]], dtype=np.float)
            dynsys = self.dynsys

            Na_cores = dynsys.A.N_cores
            Ns_cores = dynsys.S.N_cores

            l_of_masks = []
            for i in range(dynsys.S.D):
                l_of_masks.append(
                    tt.tensor(
                        (
                            self.ttSi[i].full()
                            >= -dynsys.S.h[i] / 2 + absorb_region_bounds[i][0]
                        )
                        * (
                            self.ttSi[i].full()
                            <= dynsys.S.h[i] / 2 + absorb_region_bounds[i][1]
                        )
                    )
                )

            l_of_ones = []
            for i in range(dynsys.A.D):
                l_of_ones.append(tt.ones(Na_cores[i]))

            lst = list(Ns_cores.T[0]) + list(Na_cores.T[0])
            uno_sa = tt.ones(lst)

            self.ANTI_ABSORB_MASK_S = tt.ones(list(Ns_cores.T[0])) - tt.mkron(
                l_of_masks
            ).round(1e-10)
            self.ANTI_ABSORB_MASK_SA = tt.ones(
                list(Ns_cores.T[0]) + list(Na_cores.T[0])
            ) - tt.mkron(l_of_masks + l_of_ones).round(1e-10)
            # ABSORB_MASK = (uno_sa-ABSORB_MASK).round(1e-10)


def _make_Qh_dt(self, verbose=True):
    # For slowmixing
    Ds = self.dynsys.S.D
    hs = self.dynsys.S.h
    if verbose:
        print("-Making Qh")
    self.Qh = 0 * tt.ones(self.Bi[0].n)
    for i in range(Ds):
        self.Qh += tt.multifuncrs2(
            [self.Bi[i]], lambda x: np.abs(x) / hs[i], verb=verbose, eps=self.tol
        )

    if verbose:
        print("--Finding max Qh")
    self.Qh = -tt_min.min_tens(-self.Qh, verb=verbose)[0]
    if verbose:
        print("--Qh=%s", self.Qh)
    self.dt = 1 / self.Qh


def _applyBCS__(self, verbose=False):  # applies BCS by multiplying P tensor by BCS mask
    if verbose:
        print("Applying BCS")
    Ns_cores = self.dynsys.S.N_cores
    Na_cores = self.dynsys.A.N_cores
    BCS_PERIODIC = self.dynsys.S.BCS_PERIODIC
    Ds = self.dynsys.S.D
    hs = self.dynsys.S.h
    self.BCSMASK = [[], []]
    for iside in [0, 1]:  # sweep sides
        for i_external in range(self.dynsys.S.D):  # sweep dimensions
            # forming a patch
            l = []
            for i in range(self.dynsys.S.D):
                if i_external == i:
                    if iside == 0:
                        delta_center = 0
                    else:
                        delta_center = Ns_cores.T[0][i] - 1

                    # each side, dimension has exactly 1 tt-mask
                    tt_uno = tt.ones([Ns_cores.T[0][i]])
                    if not BCS_PERIODIC[i_external]:
                        l.append(
                            (
                                tt_uno
                                - tt.delta([Ns_cores.T[0][i]], center=delta_center)
                            ).round(eps=self.tol)
                        )
                    else:
                        l.append(tt_uno)

                else:
                    l.append(tt.ones([Ns_cores.T[0][i]]))
            # adding a patch
            self.BCSMASK[iside].append(tt.mkron(l + [tt.ones(list(Na_cores.T[0]))]))

    for i in range(Ds):
        self.P_plus[i] = (self.P_plus[i] * self.BCSMASK[1][i]).round(eps=self.tol)
        self.P_minus[i] = (self.P_minus[i] * self.BCSMASK[0][i]).round(eps=self.tol)


def create_local_dts_Bxpm(self, x):
    Ds = self.dynsys.S.D
    Qh = 0
    Bx_plus = [None] * Ds
    Bx_minus = [None] * Ds
    for i_d in range(Ds):
        Bx = mytt.fast_slice_cross(x, self.Bi[i_d])
        Bx_plus[i_d] = mytt.pos(Bx) / self.dynsys.S.h[i_d]
        Bx_minus[i_d] = mytt.neg(Bx) / self.dynsys.S.h[i_d]
        Qh += Bx_plus[i_d] + Bx_minus[i_d] + 1e-0 * self.tol + SMOOTH_CONST
    dts = 1.0 / Qh
    return dts, Bx_plus, Bx_minus


def make_P_tensors(self, absorber=None, verbose=False, alpha=1e4, fastmixing=False):
    Ds = self.dynsys.S.D
    hs = self.dynsys.S.h
    if verbose:
        print("- Making P TT-vectors")
    # 0. Making B
    if self.Bi is None:
        self.make_B_SIGMA2_tensors()
        if verbose:
            print("-- TT B is not found, making it now:")

    if not fastmixing:
        # 1. Separating B+ and B-
        self._make_Bplus_minus(alpha=alpha, verbose=verbose)
        # 2. Calculating normalization constant (or function)
        self._make_Qh_dt(verbose=verbose)
        # 3. Creating tt P_plus and P_minus
        self.P_plus = []
        self.P_minus = []
        for i in range(Ds):
            self.P_plus.append(
                ((self.B_plus[i] * (1 / hs[i])) * self.dt).round(eps=self.tol)
            )
            self.P_minus.append(
                ((self.B_minus[i] * (1 / hs[i])) * self.dt).round(eps=self.tol)
            )
        # 4. Calculating and applying BCS #as function
        self._applyBCS__(verbose=verbose)
    else:
        print("+++++++++++++FASTMIXING+++++++++++++")
        ### идея:
        # Вероятности пусть будут в тензорах, а домножение уже будет как бог даст
        # Вероятности такие же, какие были внутри ттви

        ### Оригинальный ttvi-fastmix код работает так
        # Берется тензор(!!) B
        # из него крестом извлекается орган (сосиска по S и полностью по А) в np.array
        # орган разделяется на + и -
        # По всем B+-_i органам делается Qh и dts
        # в органы BCS запихивается простым занулением, без тензоров
        # Вероятности делаются из органов B+- с обычным домножением на dts
        # Правильно домножается на V, выдается ответ (как np.array)

        ### А ttqi-fastmix код должен работать так:
        # Берется тензор B
        # В цикле берется крест по каждому отдельному P+-_i
        # Из B крестом извлекается орган (сосиска по S и полностью по А) в np.array
        # орган разделяется на + и -
        # По всем B+-_i органам делается Qh и dts
        # в органы BCS запихивается простым занулением, без тензоров
        # Вероятности делаются из органов B+- с обычным домножением на dts
        # Из абсорбера извлекается орган и домножается на вероятность
        # Выдается ответ (как np.array)

        self.P_plus = []
        self.P_minus = []
        self.P_stay = None

        def create_local_Pxpm(self, x, i_d, sign):
            # x is sa_idx
            dts, Bx_plus, Bx_minus = create_local_dts_Bxpm(self, x)
            if sign == "+":
                Px_proper_local = Bx_plus[i_d] * dts
                if not self.dynsys.S.BCS_PERIODIC[i_d]:  # applying BCS
                    Px_proper_local[
                        x[:, i_d] == self.dynsys.S.N_cores.T[0][i_d] - 1
                    ] = 0
            else:
                Px_proper_local = Bx_minus[i_d] * dts
                if not self.dynsys.S.BCS_PERIODIC[i_d]:  # applying BCS
                    Px_proper_local[x[:, i_d] == 0] = 0
            return Px_proper_local

        for i_d in range(Ds):
            print("Fastslice - Forming P_pm_i with i_d=%i" % i_d)
            self.P_plus.append(
                tt.multifuncrs2(
                    self.ttMESH_SAidx, lambda x: create_local_Pxpm(self, x, i_d, "+")
                )
            )
            self.P_minus.append(
                tt.multifuncrs2(
                    self.ttMESH_SAidx, lambda x: create_local_Pxpm(self, x, i_d, "-")
                )
            )
    # ----------------------------------------------------------------------------------------

    # 5. Calculating and applying ABSORBERS #as function
    # tt_sa_uno = tt.ones(self.ttMESH_SA[0].n)
    if self.dynsys.absorber is not None:
        self._make_absorber()
        for i in range(Ds):
            self.P_plus[i] = (
                self.P_plus[i] * self.ANTI_ABSORB_MASK_SA
            )  # .round(eps=self.tol)
            self.P_minus[i] = (
                self.P_minus[i] * self.ANTI_ABSORB_MASK_SA
            )  # .round(eps=self.tol)
    # 6. Calculating P_stay
    # tt_sa_uno = tt.ones(self.ttMESH_SA[0].n)
    self.P_stay = self.tt_ones_SA
    for i in range(self.dynsys.S.D):
        self.P_stay -= self.P_minus[i]
        self.P_stay = self.P_stay.round(eps=self.tol)
        self.P_stay -= self.P_plus[i]
        self.P_stay = self.P_stay.round(eps=self.tol)


def _QfromV(self, V, eps=1e-5, gamma=1):
    Na_cores = self.dynsys.A.N_cores
    R = self.R
    V = self.V
    Ds = self.dynsys.S.D
    gamma = self.dynsys.gamma
    RMAX_Q = 2 * self.RMAX

    def ro(ttv):
        return ttv.round(rmax=RMAX_Q, eps=eps)

    # uses the access to P_plus, P_minus, P_stay
    # print('QfromV')
    # print('R.full()')
    # print(R.full())
    # print('V.full()')
    # print(V.full())
    Q = R + gamma * (tt.mkron(V, tt.ones(Na_cores.T[0])) * self.P_stay)
    Q = ro(Q)
    for i_d in range(Ds):
        Q += gamma * (
            tt.mkron(mytt.roll(V, -1, axis=i_d), tt.ones(Na_cores.T[0]))
            * self.P_plus[i_d]
        )
        Q = ro(Q)
        Q += gamma * (
            tt.mkron(mytt.roll(V, +1, axis=i_d), tt.ones(Na_cores.T[0]))
            * self.P_minus[i_d]
        )
        Q = ro(Q)

    # print('Q.full()')
    # print(Q.full())
    return Q


def _updateQfromV(self, eps=1e-5, gamma=1):
    self.Q = self._QfromV(self.V, eps=eps, gamma=gamma)


def _VfromQ(self, eps=1e-5, Q=None, verbose=False):
    if Q is None:
        Q = self.Q
    V = tt.multifuncrs2(
        self.ttMESH_Sidx,
        lambda x: mytt.fast_slice_cross(x, Q, fun="max"),
        verb=verbose,
        eps=eps,
        nswp=5,
        kickrank=5,
        y0=self.V,
    ).round(eps=eps, rmax=self.RMAX)
    return V


def _updateVfromQ(self, Q=None, eps=1e-5):
    if Q is None:
        Q = self.Q
    self.V = self._VfromQ(eps=eps, Q=Q)


def make_Policy_from_Q_argmax_full(self, Q=None):
    warnings.warn("Unpacking Q, will cause crash if Q is large")
    if Q is None:
        Q = self.Q
    # DIRTYHACK
    # sooo we need argmax on all dimensions of action space
    Ns = self.dynsys.S.N
    Na = self.dynsys.A.N
    self.Policy = self.dynsys.A.idx2point(
        np.argmax(Q.full().reshape(Ns, Na), axis=-1).reshape(self.dynsys.S.N_cores.T[0])
    )
