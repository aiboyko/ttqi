import warnings
import numpy as np
import mytt
import tt
import time
import numbers
import scipy as sp
import copy
import scipy.sparse
import sdeint
from collections import Counter
import matplotlib.pyplot as plt
from tt.optimize import tt_min
from IPython.display import display, clear_output


class Solver:
    from ._ttqi import (
        make_B_SIGMA2_tensors,
        _make_Bplus_minus,
        _make_absorber,
        _make_Qh_dt,
        _applyBCS__,
        make_P_tensors,
        _QfromV,
        _updateQfromV,
        _VfromQ,
        _updateVfromQ,
        make_Policy_from_Q_argmax_full,
        create_local_dts_Bxpm,
    )
    from ._ttvi import local_Q, fast_slice_V2V, _VfromV, _updateVfromV, fast_slice_V2Pi

    def __init__(self, dynsys, tol=1e-5, RMAX=75, fastmixing=False, adaptive_coef=0.2):
        self.dynsys = dynsys
        self.Bi = None
        self.R = None
        self.tol = tol
        self.RMAX = RMAX
        self.ttMESH_S = None
        self.P_plus = None  #  used as a flag that we haven't initialized grid yet
        self.fastmixing = fastmixing
        # probably needed to be done properly
        self.policyfun_zero = self.dynsys.policyfun_zero
        self.adaptive_coef = adaptive_coef

    def make_grid(self):
        print("--- Making TT meshgrids")
        self.ttSi = []
        # self.ttSidx = []
        for i in range(self.dynsys.S.D):
            self.ttSi.append(
                tt.linspace(
                    self.dynsys.S.N_cores[i],
                    a=self.dynsys.S.bounds[i, 0],
                    b=self.dynsys.S.bounds[i, 1],
                )
            )
        self.ttAi = []
        # self.ttAidx = []
        for i in range(self.dynsys.A.D):
            self.ttAi.append(
                tt.linspace(
                    self.dynsys.A.N_cores[i],
                    a=self.dynsys.A.bounds[i, 0],
                    b=self.dynsys.A.bounds[i, 1],
                )
            )
        self.ttMESH_S = mytt.meshgrid(self.ttSi)
        self.ttMESH_SA = mytt.meshgrid(self.ttSi + self.ttAi)
        self.ttMESH_Sidx = mytt.meshgrid_from_N(self.dynsys.S.N_cores.T[0])

        lst_s = list(self.dynsys.S.N_cores.T[0])
        self.tt_ones_S = tt.ones(lst_s)
        lst_sa = list(self.dynsys.S.N_cores.T[0]) + list(self.dynsys.A.N_cores.T[0])
        self.tt_ones_SA = tt.ones(lst_sa)
        self.ttMESH_SAidx = mytt.meshgrid_from_N(lst_sa)

    def _Rdts(self, idx_sa):
        dts, _, _ = self.create_local_dts_Bxpm(idx_sa)
        # print('local dts:')
        # print(dts)
        s_idx = idx_sa.T[0 : self.dynsys.S.D]
        a_idx = idx_sa.T[self.dynsys.S.D : self.dynsys.S.D + self.dynsys.A.D]
        s = self.dynsys.S.idx2point(s_idx)
        a = self.dynsys.A.idx2point(a_idx)
        Rbuf = self.dynsys.R_fun(s, a, dynsys=self.dynsys)
        ans = dts * Rbuf
        return ans

    def make_R_tensor(self):
        print("-- Making R TT-vector")
        print("make_R_tensor(Solver): fastmixing = %s" % self.fastmixing)
        if not self.fastmixing:
            # slowmixing,
            R = tt.multifuncrs2(
                self.ttMESH_SA,
                lambda sa: self.dt
                * self.dynsys.R_fun(
                    sa.T[0 : self.dynsys.S.D],
                    sa.T[self.dynsys.S.D : self.dynsys.S.D + self.dynsys.A.D],
                ),
                eps=self.tol,
                rmax=self.RMAX,
            ).round(eps=self.tol)
        else:
            # fastmixing
            R = tt.multifuncrs2(
                self.ttMESH_SAidx, self._Rdts, eps=self.tol, rmax=self.RMAX
            ).round(eps=self.tol)
            print("Rdts tensor is:")
            print(R.full())

        # tt_sa_uno = tt.ones(self.ttMESH_SA[0].n)
        if self.dynsys.absorber is not None:
            print("Applying absorber mask")
            R *= self.ANTI_ABSORB_MASK_SA

        self.R = R.round(eps=self.tol)

    def policyfun_fromPolicy(self, s):
        # DIRTYHACK
        sidxs = self.dynsys.S.point2idx(s)
        return self.Policy[tuple(sidxs)]

    def VI(
        self,
        maxit=100,
        typeOfIteration="QI",
        relerr_goal=1e-3,
        rounding_in_VI=False,
        verbose=True,
    ):
        # Value iteration with honest tt-elwise products and tons of round()'s
        if self.P_plus is None and typeOfIteration == "QI":
            # this scenario means it is the first start
            print("- TT P's are not found, making it now:")
            self.make_P_tensors(fastmixing=self.fastmixing, verbose=verbose)

        if self.R is None:
            print("- TT R is not found, making it now:")
            self.make_R_tensor()
        tol_current = 1
        try:
            self.it
        except BaseException:
            self.V = tt.ones(self.dynsys.S.N_cores.T[0]) * 0.0  # comment
            self.err = 1  # this
            self.it = 0  # out
            self.its = []
            self.errs = []
            self.abserrs = []
            self.times = []
            self._Vnew = tt.ones(self.dynsys.S.N_cores.T[0]) * 0

            if self.dynsys.absorber is not None:  #
                self.V *= self.ANTI_ABSORB_MASK_S
                self._Vnew *= self.ANTI_ABSORB_MASK_S

            print("Starting Value Iteration in Saved-TT format")
        while True:  # Main loop of iteration
            self.it += 1
            if verbose:
                print("it = %s" % self.it)
            timeOfStart = time.time()
            if self.it >= 100:
                tol_current = self.err * self.adaptive_coef
            elif self.it >= 2:
                tol_current = 1e-3
            if verbose:
                print("tol_current=%s" % tol_current)

            if typeOfIteration == "QI":
                timeOfQfun = time.time()
                self._updateQfromV(eps=tol_current, gamma=self.dynsys.gamma)
                timeOfQfun = time.time() - timeOfQfun
                if verbose:
                    print("tQfun = %s" % timeOfQfun)
                Vold = copy.copy(self.V)
                timeOfVfromQ = time.time()
                self._updateVfromQ(eps=tol_current)
                if self.dynsys.absorber is not None:
                    self.V *= self.ANTI_ABSORB_MASK_S
                timeOfVfromQ = time.time() - timeOfVfromQ
                if verbose:
                    print("tVfromQfun = %s" % timeOfVfromQ)
            else:
                Vold = copy.copy(self.V)
                timeOfVcross = time.time()
                self._updateVfromV(
                    eps=tol_current, rounding=rounding_in_VI, verbose=verbose
                )
                timeOfVcross = time.time() - timeOfVcross
                if verbose:
                    print("tVfromVcross = %s" % timeOfVcross)

            self.abserr = mytt.abserr(self.V, Vold)
            self.err = self.abserr / (Vold.norm() + 1e-16)
            if verbose:
                print("abserr = %s" % self.abserr)
                print("relerr = %s" % self.err)
                print("V ranks = %s" % self.V.r)
            #     print(V)
            if (self.err < relerr_goal) | (self.it > maxit):
                print(self.err)
                print(self.it)
                break
            self.its.append(self.it)
            self.errs.append(self.err)
            self.abserrs.append(self.abserr)
            curtime = time.time() - timeOfStart
            self.times.append(curtime)
            if verbose:
                print("time spent per iteration (s) = %s" % curtime)
                print("__________________")
                print(" ")
