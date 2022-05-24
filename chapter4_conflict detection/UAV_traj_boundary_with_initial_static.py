# the initial state of the UAV is static(v0 = 0), which is equal to kind=1 one of the O-D flying situation
import numpy as np
import math
import scipy.integrate as si
from scipy.optimize import fsolve
# only used to solve the eq
from matplotlib import pyplot as plt

from sympy import *
import sympy as sy

from conflict_detection.tools_function import cs_09_2_cs_xy, cs_xy_2_cs_09


def cal_r_t(v_m, delta_t, acc_m):
    t_acc = v_m / acc_m
    if delta_t <= t_acc:
        r_t = 1 / 2 * acc_m * (delta_t ** 2)
    else:
        r_t1 = 1 / 2 * acc_m * (t_acc ** 2)
        r_t2 = v_m * (delta_t - t_acc)
        r_t = r_t1 + r_t2
    return r_t


def generate_initial_ras_set(xy_ij_09, dpt_r):
    ve_x_left = xy_ij_09[0, 0] - dpt_r
    ve_x_right = xy_ij_09[0, 0] + dpt_r

    ve_y_low = xy_ij_09[0, 1] - dpt_r
    ve_y_up = xy_ij_09[0, 1] + dpt_r

    init_area_09 = np.array([[ve_x_left, ve_y_low],
                             [ve_x_left, ve_y_up],
                             [ve_x_right, ve_y_up],
                             [ve_x_right, ve_y_low]])

    return init_area_09


def distinguishing_xy_in_srd(space_ras, ras_size, init_area_09, xy_ij_09, dpt_r):
    ras_i_left = int(np.floor(init_area_09[0, 0] / ras_size[0]))
    ras_i_right = int(np.ceil(init_area_09[3, 0] / ras_size[0]))

    ras_j_low = int(np.floor(init_area_09[0, 1] / ras_size[1]))
    ras_j_up = int(np.ceil(init_area_09[1, 1] / ras_size[1]))

    ras_in_its = []
    for i in range(ras_i_left, ras_i_right+1, 1):
        for j in range(ras_j_low, ras_j_up+1, 1):
            ras_ij = space_ras[:, i, j, 0]
            dst_initial = np.linalg.norm((xy_ij_09[0] - ras_ij), ord=2)
            if dst_initial <= dpt_r:
                ras_in_its.append([i, j])

    return np.array((ras_in_its))


def cal_prob_in_ras(space_ras, ras_size, alpha, uxy, wp_ij, v_m, acc_m):
    """
    Args:
    :param space_ras:
    :param ras_size:
    :param ras_in_its:
    :param alpha:
    :param uxy:
    :param wp_ij: .shape=[2, f], not O and D, but O and expected location
    :param v_m:
    :param t:
    :param xmt:
    :return:
    """
    def generate_Upper_Lower(wp_ij, r, xt):

        # O_E = np.linalg.norm((wp_ij[0, 0:-1] - wp_ij[1, 0:-1]), ord=2)
        # Uxt = r - O_E
        # Lxt = r + O_E
        # Uyt = math.sqrt(r ** 2 - (xt - (-O_E)) ** 2)
        # Lyt = - Uyt
        # undirected Brownian motion in xy coordinates
        Uxt = r
        Lxt = -r
        Uyt = math.sqrt(r ** 2 - xt ** 2)
        Lyt = - Uyt
        return Uxt, Lxt, Uyt, Lyt

    def prob_ras_k(wp_ij, Uxt, Lxt, Uyt, Lyt):
        # due to there is only origin, but the flying is directed, so we should build the movement model by considering
        # the expectation and variance. the expectation of variable x, y should be zero in rotational/ translational coordinate,
        # however, there is not destination, the tj is not defined, so,the variances of x,y are defined as follows,
        # which is equal to undirected movement model.

        def variance_xy(t, ti):
            return t - ti
        miu_x = 0
        miu_y = 0

        sigma_x2 = sigma_y2 = variance_xy(wp_ij[1, -1], wp_ij[0, -1])

        fx = lambda xt: 1 / math.sqrt(2 * math.pi * sigma_x2) * math.e ** (-((xt - miu_x) ** 2) / (2 * sigma_x2))
        [Fxab, err] = si.quad(fx, Lxt, Uxt)
        if Fxab >= 1:
            Fxab = Fxab - err

        fy = lambda yt: 1 / math.sqrt(2 * math.pi * sigma_y2) * math.e ** (-((yt - miu_y) ** 2) / (2 * sigma_y2))

        [Fyab, err] = si.quad(fy, Lyt, Uyt)
        if Fyab >= 1:
            Fyab = Fyab - err

        prob_xt_yt_t = lambda xt, yt: (1 / math.sqrt(2 * math.pi * sigma_x2) * math.e ** (-((xt - miu_x) ** 2) / (2 * sigma_x2)) / Fxab) * \
                                      (1 / math.sqrt(2 * math.pi * sigma_y2) * math.e ** (-((yt - miu_y) ** 2) / (2 * sigma_y2)) / Fyab)
        return prob_xt_yt_t

    def cal_double_quad(prob_fxy, xl, xu, yl, yu):
        [prob_k, err] = si.dblquad(prob_fxy, xl, xu, yl, yu)
        if prob_k >= 1:
            prob_k = prob_k - err

        return prob_k

    r_t = cal_r_t(v_m, wp_ij[1, -1] - wp_ij[0, -1], acc_m)
    init_area_09 = generate_initial_ras_set(wp_ij[:, 0:-1], r_t)
    ras_its = distinguishing_xy_in_srd(space_ras, ras_size, init_area_09, wp_ij[:, 0:-1], r_t)
    space_its_09 = np.concatenate([np.expand_dims(space_ras[:, ras_its[k, 0], ras_its[k, 1], :], axis=1)
                                   for k in range(len(ras_its))],
                                  axis=1)
    space_its_xy = cs_09_2_cs_xy(space_its_09, alpha, np.expand_dims(uxy, axis=1))

    prob_its = []
    for k in range(len(ras_its)):
        space_k_xy = space_its_xy[:, k, :]
        k_xl = np.min(space_k_xy[0])
        k_xu = np.max(space_k_xy[0])
        k_yl = np.min(space_k_xy[1])
        k_yu = np.max(space_k_xy[1])

        Uxt, Lxt, Uyt, Lyt= generate_Upper_Lower(wp_ij, r_t, xt=space_k_xy[0, 0])

        prob_fxy = prob_ras_k(wp_ij, Uxt, Lxt, Uyt, Lyt)

        prob_k_xy = cal_double_quad(prob_fxy, k_xl, k_xu, k_yl, k_yu)

        prob_k_hollow = cal_double_quad(prob_fxy,
                                        space_k_xy[0, 0] - abs(ras_size[0] * math.cos(alpha) - ras_size[1] * math.sin(alpha)) / 2,
                                        space_k_xy[0, 0] + abs(ras_size[0] * math.cos(alpha) - ras_size[1] * math.sin(alpha)) / 2,
                                        space_k_xy[1, 0] - abs(ras_size[1] * math.cos(alpha) - ras_size[0] * math.sin(alpha)) / 2,
                                        space_k_xy[1, 0] + abs(ras_size[1] * math.cos(alpha) - ras_size[0] * math.sin(alpha)) / 2)
        prob_its.append((prob_k_xy + prob_k_hollow) / 2)

    return np.array(prob_its)


if __name__ == '__main__':
    # the calculation of probablity of each ras belong to the position reachable domain for flying target is the first step,
    # and the defination and processing of conflict detection between target and other objects will be similar,
    # which will be calculate in one file

    print('the information processing of the target, which initial state is static')

