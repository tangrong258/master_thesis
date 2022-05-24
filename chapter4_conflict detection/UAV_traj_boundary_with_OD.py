import numpy as np
import math
import scipy.integrate as si
from scipy.optimize import fsolve
# only used to solve the eq
from matplotlib import pyplot as plt

from sympy import *
import sympy as sy
import multiprocessing as mp  # mp.pool是多进程池， 用cpu多个核心

from tools_function import cs_09_2_cs_xy, cs_xy_2_cs_09, v_average


def cal_prob_in_ras_OD(space_ras, ras_size, wp_ij, v_m, t):
    def cal_xmt(wp_ij, v_m, t):
        v_avg = v_average(wp_ij)
        ti = wp_ij[0, -1]
        tj = wp_ij[1, -1]
        xmt = (v_avg ** 2 - v_m ** 2) * (ti + tj - 2 * t) / (2 * v_avg)
        arr_low = (v_avg - v_m) * (tj - t)
        dpt_up = (v_avg - v_m) * (ti - t)

        if xmt <= arr_low:
            kind = -1
        elif xmt >= dpt_up:
            kind = 1
        else:
            kind = 0
        return xmt, kind

    def cal_alpha(wp_ij):
        vector = [wp_ij[1, 0] - wp_ij[0, 0], wp_ij[1, 1] - wp_ij[0, 1]]
        alpha = np.arctan((vector[1]) / (vector[0]))
        if vector[0] < 0:
            alpha += np.pi
        return alpha

    def cal_uxy(wp_ij, t):
        ux = ((t - wp_ij[0, -1]) * wp_ij[1, 0] + (wp_ij[1, -1] - t) * wp_ij[0, 0]) / (wp_ij[1, -1] - wp_ij[0, -1])
        uy = ((t - wp_ij[0, -1]) * wp_ij[1, 1] + (wp_ij[1, -1] - t) * wp_ij[0, 1]) / (wp_ij[1, -1] - wp_ij[0, -1])
        return np.array([ux, uy])

    def generate_initial_ras_set(xy_ij_09, kind, dpt_r, arr_r, alpha, uxy):
        def cal_its_2circle(c1, c2, r1, r2):
            x, y = symbols('x, y', real=True)

            # (x-x0)^2 + (y-y0)^2 = R^2
            dpt_area_globe_expr = (x - c1[0]) ** 2 + (y - c1[1]) ** 2 - r1 ** 2

            arr_area_globe_expr = (x - c2[0]) ** 2 + (y - c2[1]) ** 2 - r2 ** 2

            eq = [dpt_area_globe_expr, arr_area_globe_expr]
            ist = sy.solve(eq, [x, y])  # intersection
            return ist, eq

        xy_ij_xy = cs_09_2_cs_xy(xy_ij_09.T, alpha, np.expand_dims(uxy, axis=1)).T
        if kind == 0:
            x, y = sy.symbols('x, y', real=True)
            solves, eqs = cal_its_2circle(xy_ij_xy[0], xy_ij_xy[1], dpt_r, arr_r)
            ist_set = []
            for i in solves:
                s_i = []
                for j in range(len(i)):
                    try:
                        j_t = float(i[j])
                    except:
                        j_t = np.nan
                    s_i.append(j_t)
                if not np.isnan(s_i).any():
                    ist_set.append(s_i)

            eqs_0 = eqs[0].subs(y, 0)
            tgt_set = sy.solve(eqs_0, x)
            tg_x_right = np.max([float(tgt) for tgt in tgt_set])

            eqs_1 = eqs[1].subs(y, 0)
            tgt_set = sy.solve(eqs_1, x)
            tg_x_left = np.min([float(tgt) for tgt in tgt_set])

            tg_y_up = np.max([ist[1] for ist in ist_set])
            tg_y_low = np.min([ist[1] for ist in ist_set])

            vertices_xy = np.array([[tg_x_left, tg_y_low],
                                    [tg_x_left, tg_y_up],
                                    [tg_x_right, tg_y_up],
                                    [tg_x_right, tg_y_low]])

            vertices_09 = cs_xy_2_cs_09(vertices_xy.T, alpha, np.expand_dims(uxy, axis=1)).T
            ve_x_left = np.min(vertices_09[:, 0])
            ve_x_right = np.max(vertices_09[:, 0])

            ve_y_low = np.min(vertices_09[:, 1])
            ve_y_up = np.max(vertices_09[:, 1])

            init_area_09 = np.array([[ve_x_left, ve_y_low],
                                     [ve_x_left, ve_y_up],
                                     [ve_x_right, ve_y_up],
                                     [ve_x_right, ve_y_low]])

        elif kind == -1:
            ve_x_left = xy_ij_09[0, 0] - dpt_r
            ve_x_right = xy_ij_09[0, 0] + dpt_r

            ve_y_low = xy_ij_09[0, 1] - dpt_r
            ve_y_up = xy_ij_09[0, 1] + dpt_r

            init_area_09 = np.array([[ve_x_left, ve_y_low],
                                     [ve_x_left, ve_y_up],
                                     [ve_x_right, ve_y_up],
                                     [ve_x_right, ve_y_low]])

        else:
            ve_x_left = xy_ij_09[1, 0] - arr_r
            ve_x_right = xy_ij_09[1, 0] + arr_r

            ve_y_low = xy_ij_09[1, 1] - arr_r
            ve_y_up = xy_ij_09[1, 1] + arr_r

            init_area_09 = np.array([[ve_x_left, ve_y_low],
                                     [ve_x_left, ve_y_up],
                                     [ve_x_right, ve_y_up],
                                     [ve_x_right, ve_y_low]])

        return init_area_09

    def distinguishing_xy_in_srd(space_ras, ras_size, init_area_09, xy_ij_09, dpt_r, arr_r):
        ras_i_left = int(np.floor(init_area_09[0, 0] / ras_size[0]))
        ras_i_right = int(np.floor(init_area_09[3, 0] / ras_size[0]))

        ras_j_low = int(np.floor(init_area_09[0, 1] / ras_size[1]))
        ras_j_up = int(np.floor(init_area_09[1, 1] / ras_size[1]))

        ras_in_its = []
        for i in range(ras_i_left, ras_i_right + 1, 1):
            for j in range(ras_j_low, ras_j_up + 1, 1):
                ras_ij = space_ras[:, i, j, 0]
                dst_dpt = np.linalg.norm((xy_ij_09[0] - ras_ij), ord=2)
                if dst_dpt <= dpt_r:
                    dst_arr = np.linalg.norm((xy_ij_09[1] - ras_ij), ord=2)
                    if dst_arr <= arr_r:
                        ras_in_its.append([i, j])

        return np.array((ras_in_its))

    def generate_approximate_real_UL(wp_ij, v_m, t, xmt, less_xm, xt):
        v_avg = v_average(wp_ij)
        ti = wp_ij[0, -1]
        tj = wp_ij[1, -1]
        arr_low = (v_avg - v_m) * (tj - t)
        dpt_up = (v_avg - v_m) * (ti - t)

        if xmt <= arr_low:
            Uxt = (v_avg - v_m) * (ti - t)
            Lxt = Uxt - 2 * v_m * (t - ti)
            Uyt = math.sqrt((v_m * (t - ti)) ** 2 - (xt - v_avg * (ti - t)) ** 2)
            Lyt = -Uyt
        elif xmt >= dpt_up:
            Lxt = (v_avg - v_m) * (tj - t)
            Uxt = Lxt + 2 * v_m * (tj - t)
            Uyt = math.sqrt((v_m * (t - tj)) ** 2 - (xt - v_avg * (t - tj)) ** 2)
            Lyt = -Uyt
        else:
            Uxt = (v_avg - v_m) * (ti - t)
            Lxt = (v_avg - v_m) * (tj - t)
            if less_xm:
                Uyt = math.sqrt((v_m * (tj - t)) ** 2 - (v_avg * (tj - t) - xt) ** 2)
            else:
                Uyt = math.sqrt((v_m * (t - ti)) ** 2 - (v_avg * (ti - t) - xt) ** 2)
            Lyt = -Uyt

        return Uxt, Lxt, Uyt, Lyt

    def prob_ras_k(wp_ij, t, Uxt, Lxt, Uyt, Lyt):

        sigma_x2 = sigma_y2 = (t - wp_ij[0, -1]) * (wp_ij[1, -1] - t) / (wp_ij[1, -1] - wp_ij[0, -1])

        fx = lambda xt: 1 / math.sqrt(2 * math.pi * sigma_x2) * math.e ** (-(xt ** 2) / (2 * sigma_x2))
        [Fxab, err] = si.quad(fx, Lxt, Uxt)
        if Fxab >= 1:
            Fxab = Fxab - err

        fy = lambda yt: 1 / math.sqrt(2 * math.pi * sigma_y2) * math.e ** (-(yt ** 2) / (2 * sigma_y2))

        [Fyab, err] = si.quad(fy, Lyt, Uyt)
        if Fyab >= 1:
            Fyab = Fyab - err

        prob_xt_yt_t = lambda xt, yt: (1 / math.sqrt(2 * math.pi * sigma_x2) * math.e ** (-(xt ** 2) / (2 * sigma_x2)) / Fxab) * \
                                      (1 / math.sqrt(2 * math.pi * sigma_y2) * math.e ** (-(yt ** 2) / (2 * sigma_y2)) / Fyab)
        return prob_xt_yt_t

    def cal_double_quad(prob_fxy, xl, xu, yl, yu):
        [prob_k, err] = si.dblquad(prob_fxy, xl, xu, yl, yu)
        if prob_k >= 1:
            prob_k = prob_k - err

        return prob_k

    def func_re(k, space_its_xy, wp_ij, v_m, t, xmt):
        space_k_xy = space_its_xy[:, k,:]
        k_xl = np.min(space_k_xy[0])
        k_xu = np.max(space_k_xy[0])
        k_yl = np.min(space_k_xy[1])
        k_yu = np.max(space_k_xy[1])

        if space_k_xy[0, 0] < xmt:
            less_xm = False
        else:
            less_xm = True

        Uxt, Lxt, Uyt, Lyt = generate_approximate_real_UL(wp_ij, v_m, t, xmt, less_xm, xt=space_k_xy[0, 0])
        if k_xl < Lxt:
            k_xl = Lxt
        if k_xu > Uxt:
            k_xu = Uxt
        if k_yl < Lyt:
            k_yl = Lyt
        if k_yu > Uyt:
            k_yu = Uyt

        prob_fxy = prob_ras_k(wp_ij, t, Uxt, Lxt, Uyt, Lyt)
        prob_k_xy = cal_double_quad(prob_fxy, k_xl, k_xu, k_yl, k_yu)

        prob_k_hollow = cal_double_quad(prob_fxy,
                                        space_k_xy[0, 0] - abs(
                                            ras_size[0] * abs(np.cos(alpha)) - ras_size[1] * abs(
                                                np.sin(alpha))) / 2,
                                        space_k_xy[0, 0] + abs(
                                            ras_size[0] * abs(np.cos(alpha)) - ras_size[1] * abs(
                                                np.sin(alpha))) / 2,
                                        space_k_xy[1, 0] - abs(
                                            ras_size[1] * abs(np.cos(alpha)) - ras_size[0] * abs(
                                                np.sin(alpha))) / 2,
                                        space_k_xy[1, 0] + abs(
                                            ras_size[1] * abs(np.cos(alpha)) - ras_size[0] * abs(
                                                np.sin(alpha))) / 2)
        return (prob_k_xy + prob_k_hollow) / 2


    tgt_dpt_r = (t - wp_ij[0, -1]) * v_m
    tgt_arr_r = (wp_ij[1, -1] - t) * v_m
    xmt, kind = cal_xmt(wp_ij, v_m, t)
    alpha = cal_alpha(wp_ij)
    uxy = cal_uxy(wp_ij, t)
    init_area_09 = generate_initial_ras_set(wp_ij[:, 0:-1], kind, tgt_dpt_r, tgt_arr_r, alpha, uxy)
    ras_in_its = distinguishing_xy_in_srd(space_ras, ras_size, init_area_09, wp_ij[:, 0:-1], tgt_dpt_r, tgt_arr_r)

    space_its_09 = np.concatenate([np.expand_dims(space_ras[:, ras_in_its[k, 0], ras_in_its[k, 1], :], axis=1)
                                   for k in range(len(ras_in_its))],
                                   axis=1)
    space_its_xy = cs_09_2_cs_xy(space_its_09, alpha, np.expand_dims(uxy, axis=1))

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores - 2)

    process_list, prob_its = [], []
    for k, _ in enumerate(ras_in_its):
        space_k_xy = space_its_xy[:, k, :]
        k_xl = np.min(space_k_xy[0])
        k_xu = np.max(space_k_xy[0])
        k_yl = np.min(space_k_xy[1])
        k_yu = np.max(space_k_xy[1])

        if space_k_xy[0, 0] < xmt:
            less_xm = False
        else:
            less_xm = True

        Uxt, Lxt, Uyt, Lyt = generate_approximate_real_UL(wp_ij, v_m, t, xmt, less_xm, xt=space_k_xy[0, 0])
        if k_xl < Lxt:
            k_xl = Lxt
        if k_xu > Uxt:
            k_xu = Uxt
        if k_yl < Lyt:
            k_yl = Lyt
        if k_yu > Uyt:
            k_yu = Uyt

        prob_fxy = prob_ras_k(wp_ij, t, Uxt, Lxt, Uyt, Lyt)
        prob_k_xy = cal_double_quad(prob_fxy, k_xl, k_xu, k_yl, k_yu)

        prob_k_hollow = cal_double_quad(prob_fxy,
                                        space_k_xy[0, 0] - abs(
                                            ras_size[0] * abs(np.cos(alpha)) - ras_size[1] * abs(
                                                np.sin(alpha))) / 2,
                                        space_k_xy[0, 0] + abs(
                                            ras_size[0] * abs(np.cos(alpha)) - ras_size[1] * abs(
                                                np.sin(alpha))) / 2,
                                        space_k_xy[1, 0] - abs(
                                            ras_size[1] * abs(np.cos(alpha)) - ras_size[0] * abs(
                                                np.sin(alpha))) / 2,
                                        space_k_xy[1, 0] + abs(
                                            ras_size[1] * abs(np.cos(alpha)) - ras_size[0] * abs(
                                                np.sin(alpha))) / 2)

        prob_its.append((prob_k_xy + prob_k_hollow) / 2)

    #     process = pool.apply_async(func_re, args=(k, space_its_xy, wp_ij, v_m, t, xmt))
    #     process_list.append(process)
    #
    # for k in process_list:
    #     k.wait()
    #
    # for k in process_list:
    #     if k.ready():
    #         if k.successful():
    #             prob_its.append(k.get())
    #             print(len(prob_its))
    #
    # pool.close()
    # pool.join()

    return ras_in_its, np.array(prob_its)


if __name__ == '__main__':

    # define the airspace of management
    print()














