import tt
import numpy as np
import copy
import numbers

from collections import Counter


def sum(ttv, axislist):
    if isinstance(axis, numbers.Number):
        return tt.sum(ttv, axis=axis)
    elif isinstance(axis, np.ndarray):
        axislist = np.array(axislist, dtype=np.int)
        if axislist.size == 1:
            return tt.sum(ttv, axis=axislist.flatten()[0])
        else:
            axislist = axislist.sort()[::-1]
            for ax in axislist:
                ttv = tt.sum(ttv, axis=ax)
            return ttv


def softsign(x, alpha=1):
    # print('softsign: x = %s'%x )
    return 1.0 / (1 + np.exp(-alpha * x))


def pos(f, alpha=1e5):
    return softsign(f, alpha) * f


def neg(f, alpha=1e5):
    return -(1 - softsign(f, alpha)) * f


def ones_like(ttv):
    return tt.ones(ttv.n)


def naive_slice_cross(x, ttv):  # , shift=[None,0]):
    # d_shift, where2shift = shift

    N = ttv.n
    D_others = len(N) - x.shape[1]
    Na_cores = list(ttv.n[-D_others:])
    # print('Na_cores=%s'%Na_cores)
    buf = np.zeros([x.shape[0], 1] + Na_cores, dtype=float)

    for i in range(x.shape[0]):
        address = list(np.array(x[i, :], dtype=int)) + [slice(None)] * D_others
        # print('address=%s'%address)
        buf[i, :] = ttv[address].full()
        # if (d_shift != None):
        #     address[d_shift] = (address[d_shift] + where2shift) % N[d_shift]

    return buf


def fast_slice_cross(idx, ttv, fun=None):
    """x is a pack of indices"""

    # essentially this is a modified fastslice
    # it does fastslice-alike pre-processing for all given indices
    # then it fills remaining indices with ':''
    # then it does fastslice
    # then it maximizes over spurious (action) dimensions and returns the answer
    # the algorithm may be manually re-written such that instead of O(Ns_i * Na) it will require O(Na) memory

    """фастслайс делает так:
     в приходящей пачке сосисок адресов, в пределах каждой сосиски
     один адрес бегает, а остальные стоят.
     Надо вычислить бегающий адрес, заменить его на операцию слайса,
     а остальные адреса взять как ни в чем ни бывало.
     Далее, надо подставить этот адрес в слайсинг тт и получаить
     сразу сосиску ответов. Так как сосисок несколько, мы их подшиваем
     в один массив, а потом решейпим таким образом,
     чтобы восстановить исходную форму """
    # print('mytt.fast_slice_cross: idx=%s'%idx)
    x = np.array(idx, dtype=int)
    N = ttv.n
    batchsize = x.shape[0]
    Ds = x.shape[1]
    D_others = len(N) - Ds
    Na_cores = []
    if D_others > 0:
        Na_cores += list(ttv.n[-D_others:])
    ans = np.zeros([batchsize] + Na_cores)
    if fun == "max":
        fun = lambda y: np.max(y, axis=tuple(range(-D_others, 0)))
    elif fun == None:
        fun = lambda y: y

    if batchsize == 1:
        if D_others > 0:
            ans = ttv[list(x[0]) + [slice(None)] * D_others].full()
        else:
            ans = ttv[list(x[0])]
    else:
        Nmicrobatch = 0
        iline = 0
        lastval = 0
        for d_saw in range(
            Ds
        ):  # детектим, по какой координате индекс бегает. В тт-кресте такой индекс всегда один на сосиску
            cc = Counter(x[:, d_saw])
            if len(cc.keys()) == N[d_saw]:
                #             print('Saw structure is in d=%i' % d_saw)
                break
        tt_full_list = []
        while iline < batchsize:
            if (
                x[iline, d_saw] == 0
            ):  # пила начинается с нуля!! (ну, если мы не делаем фокусов с адресами)
                Nmicrobatch += 1  # и еще одна колбаса в нашу пачку колбас
                address = list(
                    np.asarray(x[iline], dtype=int)
                )  # наш адрес это типа как индес самого первого в колбасе
                address[d_saw] = slice(
                    None
                )  # ну кроме индекса, где пила. Его мы заменяем на slice

                if D_others > 0:
                    address += [
                        slice(None)
                    ] * D_others  # this line adds additional slicing to the end

                y = ttv[address].full()
                # print('fastslice y.shape:=%s' ,  y.shape)
                # print('fastslice y:=%s'% y )
                tt_full_list.append(
                    y
                )  # <- here! is the function of ttv (which gere is just ttv[address]). The output should be a scalar array
            elif (lastval == 0) * (x[iline, d_saw] == 1.0):

                """Здесь осуществляется подшивка.
                Суть-то какая:
                эта функция должна по идее работать и с общим случаем, а именно с фастслайс_Да.
                Что в частности должно означать, что по одному адресу возвращается прямоугольник.
                А по пачке колбас адресов, ориентированных вдоль возвращается длинный прямоугольник.
                Что у нас есть на данном этапе:
                tt_full_list хранит лист колбас
                Что мы делали раньше:
                1) Брали лист диапазонов адресов колбас,
                2) превращали их в массив колбас
                3) решейпили в ордеринге Ф, чтобы автовыпрямить
                4) имплантировали в ответ.
                (хотя вместо этого всего можно было просто делать hstack)

                # print('Nmicrobatch = %s' % Nmicrobatch)
                # print('N[saw] = %s' % N[d_saw])

                на какой вопрос надо ответить?
                является ли то, что выводит фастслайс_да реально слайсами вглубь от крестового запроса x
                для этого надо проверить на модельном примере.
                если да, то пока кладется болт на до-верификацию и идём дальше
    """
                batch2rectify = np.array(tt_full_list)
                # print(batch2rectify.shape)
                # print(Nmicrobatch * N[d_saw])
                # print(np.array(ttv.n[-D_others:], dtype=int))
                shape = np.array([Nmicrobatch * N[d_saw]] + Na_cores)
                # print('shape=%s'%shape)
                batch2rectify = batch2rectify.reshape(shape, order="F")
                # print('batch2rectify')
                # print(batch2rectify)
                ans[
                    iline - Nmicrobatch : iline - Nmicrobatch + Nmicrobatch * N[d_saw]
                ] = batch2rectify
                Nmicrobatch = 0
                tt_full_list = []
                # batch2rectify = np.array(tt_full_list)
                # print(batch2rectify.shape)
            lastval = x[iline, d_saw]
            iline += 1
            # print('ansshape')
            # print(ans.shape)
    return fun(ans).reshape(batchsize, -1)


def meshgrid(ttlist):
    ans = []
    oneslist = []
    for X in ttlist:
        oneslist.append(tt.ones(X.n))

    for i in range(len(ttlist)):
        buf = copy.copy(oneslist)
        buf[i] = ttlist[i]
        ans.append(tt.mkron(buf))

    return ans


def meshgrid_from_N(N_cores):
    meshes = []
    D = len(N_cores)
    for d in range(D):
        #         print(N[d])
        meshes.append(tt.xfun([N_cores[d]]))
    TTMESHi = meshgrid(meshes)
    return TTMESHi


def roll(ttv, shift, axis=0):  # aka circshift in MATLAB
    cores_list = tt.vector.to_list(ttv)
    cores_list[axis] = np.roll(cores_list[axis], shift, axis=1)
    ans = tt.vector.from_list(cores_list)
    return ans


def abserr(ttv1, ttv2):
    return (ttv1 - ttv2).norm()


def relerr(ttv1, ttv2):
    err = abserr(ttv1, ttv2) / (ttv1.norm() + 1e-16)
    return err


def shift_m(N, periodic=0, shift=1):  # 1d shifted numpy matrix
    print("shift in shift_m = %s" % shift)
    if periodic:
        M = np.roll(np.eye(N), shift, axis=0)
    else:
        M = np.diag(np.ones(N), -shift)[:N, :N]
    return M


def shift_ttm(N_cores, dim, BCS_PERIODIC, shift=1):
    # shift here is SAME as np.roll shift
    Nc = N_cores.T[0]
    l = []
    D = len(Nc)

    for i in range(D):
        N = Nc[i]
        if i == dim:
            l.append(
                shift_m(N, periodic=BCS_PERIODIC[i], shift=shift).reshape(1, N, N, 1)
            )
        else:
            l.append(np.eye(N).reshape(1, N, N, 1))

    return tt.matrix.from_list(l)
