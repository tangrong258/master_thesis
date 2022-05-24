# the initial state of the UAV is non-static(v0 != 0), which is a more complex situation

import numpy as np
import math
import scipy.integrate as si
from scipy.optimize import fsolve
from scipy.optimize import leastsq
from scipy import optimize
# only used to solve the eq
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

from sympy import *
import sympy as sy

from tools_function import cs_09_2_cs_xy, cs_xy_2_cs_09


def calculation_of_first_point_of_intersection(v0, vm, am):
    """
    Args:
    :param s_acc: the distance of acceleration flying, if v0 and vm is unchanged
    :param r: the minimum radius for UAV
    :return: the beta with the first point of intersection is generated,  which should be 180 < beta <= 270
    """
    nn = 3.5
    g = 9.8
    r = v0 ** 2 / (np.sqrt(nn ** 2 - 1) * g)
    unit_t = v0 / (np.sqrt(nn ** 2 - 1) * g) * np.pi / 180
    t_acc = (vm - v0) / am
    s_acc = v0 * t_acc + am * t_acc ** 2 / 2
    return 180 - math.asin(-2 * s_acc * r / (s_acc ** 2 + r ** 2)) / math.pi * 180, r, unit_t


def generate_the_traj_all_possible(v0, vm, am, t, ti, angle_delta, is_plot=True):
    # first of all, there is a setting about time,
    # which defines that the total predicted time is much larger than the turn and acceleration time for UAV,
    # also means that three flying procedure will be done,
    # including turn, acceleration, and uniform motion in a straight line
    # assuming the velocity in turn is constant
    """
    args:
    :param v0: the ground speed for UAV in monitoring moment, vector
    :param vm: the maximum gs for UAV
    :param am: the maximum acceleration for UAV
    :param t: the predicted time
    :param ti: right now
    :param angle_delta: the sampling angle difference
    :param max_beta: the absolute beta value when the point of intersection is generated between turn right and turn left
    :param nn: the maximum normal acceleration
    :param g: gravitational acceleration
    :return: the discrete boundary point
    """
    v0 = np.linalg.norm(v0, ord=2)
    a_d = angle_delta
    max_beta, r, unit_t = calculation_of_first_point_of_intersection(v0, vm, am)
    beta_set = np.arange(-max_beta, max_beta, a_d)

    last_positions = np.zeros(shape=(len(beta_set), 2))

    for i, beta in enumerate(beta_set):
        # turn
        if beta < 0:
            plot_angle = np.linspace(beta + a_d - 1, beta, a_d)
            [a, b] = [-r, 0]
            x_turn = a + r * np.cos(-plot_angle * np.pi / 180)
            y_turn = b + r * np.sin(-plot_angle * np.pi / 180)
        elif beta > 0:
            plot_angle = np.linspace(beta - a_d + 1, beta, a_d)
            [a, b] = [r, 0]
            x_turn = a + r * np.cos((180 - plot_angle) * np.pi / 180)
            y_turn = b + r * np.sin((180 - plot_angle) * np.pi / 180)
        else:
            plot_angle = np.linspace(0, beta, abs(beta) + 1)
            [a, b] = [r, 0]
            x_turn = a + r * np.cos((180 - plot_angle) * np.pi / 180)
            y_turn = b + r * np.sin((180 - plot_angle) * np.pi / 180)

        t_turn = unit_t * abs(beta)

        # accelerate
        t_acc = (vm - v0) / am
        plot_acc = np.linspace(0, t_acc, 2)
        x_acc = x_turn[-1] + (v0 * plot_acc + am * plot_acc ** 2 / 2) * np.sin(beta * np.pi / 180)
        y_acc = y_turn[-1] + (v0 * plot_acc + am * plot_acc ** 2 / 2) * np.cos(beta * np.pi / 180)

        # uniform motion in a straight line
        t_cs = t - ti - t_turn - t_acc
        plot_cs = np.linspace(0, t_cs, 2)
        x_cs = x_acc[-1] + (vm * plot_cs) * np.sin(beta * np.pi / 180)
        y_cs = y_acc[-1] + (vm * plot_cs) * np.cos(beta * np.pi / 180)

        last_positions[i] = [x_cs[-1], y_cs[-1]]

    # do not forget the target is rotor uav, the flying operation that the uav deceleration to static,
    # and then acceleration flying with every heading, which should considered in the boundary calculation
    # Assuming that the norm of the maximum acceleration and deceleration is the same
    # t_dec = v0 / am
    # plot_dec = np.linspace(0, t_dec, 2)
    # y_dec = v0 * plot_dec + am * plot_dec ** 2 / 2
    # t_acc_1 = vm / am
    # t_cs = t - t_dec - t_acc_1
    # if t_cs <= 0:
    #     r_t = 1 / 2 * am * (t_acc_1 ** 2)
    # else:
    #     r_t1 = 1 / 2 * am * (t_acc_1 ** 2)
    #     r_t2 = vm * t_cs
    #     r_t = r_t1 + r_t2
    # r_m = r_t
    # circle = []
    # for alpha in range(0, 360, 1):
    #     circle.append([0 + r_m * np.cos(alpha / 180 * np.pi), y_dec[-1] + r_m * np.sin(alpha / 180 * np.pi)])

    # circle_boundary = np.array((circle))
    # circle_center = [0, y_dec[-1]]

    if is_plot:
        fig = plt.figure(figsize=(7, 7))
        plt.plot(last_positions[:, 0], last_positions[:, 1], marker='o', ms=5.0)
        # plt.plot(circle_boundary[:, 0], circle_boundary[:, 1], marker='o', ms=5.0)
        plt.show()

    return last_positions


def cal_prob_in_ras(space_ras, ras_size, last_positions, v0, uxy_092o, uxy_092e, t, to):
    """
    # there is different with other situations, which are only one time coordinate rotation, from space to O-D or O-E,
    # however, due to the initial velocity is not equal to 0, the O-E direction is not equal to heading(velocity),
    #
    args:
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

    def cal_alpha_092o(v0):
        alpha = np.arctan(v0[0] / v0[1] + 1.0e-06)
        if v0[1] < 0:
            alpha += np.pi
        return -alpha

    def cal_alpha_092e(wp_oe):
        vector = [wp_oe[1, 0] - wp_oe[0, 0], wp_oe[1, 1] - wp_oe[0, 1]]
        alpha = np.arctan((vector[1]) / (vector[0]) + 1.0e-06)
        if vector[0] < 0:
            alpha += np.pi
        return alpha

    def generate_initial_ras_set(last_positions_09):
        # ve_x_left = np.min(np.min(last_positions_09[:, 0]), np.min(circle_boundary_09[:, 0]))
        # ve_x_right = np.max(np.max(last_positions_09[:, 0]), np.max(circle_boundary_09[:, 0]))
        #
        # ve_y_low = np.min(np.min(last_positions_09[:, 1]), np.min(circle_boundary_09[:, 1]))
        # ve_y_up = np.max(np.max(last_positions_09[:, 1]), np.max(circle_boundary_09[:, 1]))

        ve_x_left = np.min(last_positions_09[:, 0])
        ve_x_right = np.max(last_positions_09[:, 0])

        ve_y_low = np.min(last_positions_09[:, 1])
        ve_y_up = np.max(last_positions_09[:, 1])

        init_area_09 = np.array([[ve_x_left, ve_y_low],
                                 [ve_x_left, ve_y_up],
                                 [ve_x_right, ve_y_up],
                                 [ve_x_right, ve_y_low]])

        return init_area_09

    def distinguishing_xy_in_srd(space_ras, ras_size, init_area_09, alpha_092o, uxy_092o, last_positions):
        def local_polynomial_fitting(last_positions_o, ras_ij_o, epsilon=5):
            def cal_ri(xc, yc):
                return np.sqrt((x_val - xc) ** 2 + (y_val - yc) ** 2)

            def cal_r_diff(c):
                ri = cal_ri(*c)
                return ri - np.mean(ri)

            def jb(c):
                xc, yc = c
                jb_mat = np.empty((len(c), len(x_val)))
                ri = cal_ri(xc, yc)
                jb_mat[0] = (xc - x_val) / ri
                jb_mat[1] = (yc - y_val) / ri
                jb_mat = jb_mat - np.mean(jb_mat, axis=1, keepdims=True)
                return jb_mat

            beta_ras_o = np.arctan((ras_ij_o[0] - 0) / (ras_ij_o[1] - 0) + 1.0e-06)
            # due to the return of arctan function is belong to (-pi / 2, pi / 2), but the beta_ras_o should be (-pi, pi]
            if ras_ij_o[1] < 0:
                if beta_ras_o > 0:
                    beta_ras_o -= np.pi
                if beta_ras_o <= 0:
                    beta_ras_o += np.pi

            beta_ras_o = beta_ras_o / np.pi * 180
            local_beta_range = [int(beta_ras_o) - epsilon, int(beta_ras_o) + epsilon + 1]
            beta_0 = int(len(last_positions_o) / 2 + 1)

            local_last_positions = last_positions_o[local_beta_range[0] + beta_0:local_beta_range[1] + beta_0]

            x_val = local_last_positions[:, 0]
            y_val = local_last_positions[:, 1]

            c_est = 0, 0
            center_2b, ier = leastsq(cal_r_diff, c_est, Dfun=jb, col_deriv=True)

            x0, y0 = center_2b
            ri_0 = cal_ri(*center_2b)
            r0 = np.mean(ri_0)
            # residu_ri2 = np.sum((ri_0 - r0) ** 2)

            return x0, y0, r0

        ras_i_left = int(np.floor(init_area_09[0, 0] / ras_size[0]))
        ras_i_right = int(np.ceil(init_area_09[3, 0] / ras_size[0])) - 1

        ras_j_low = int(np.floor(init_area_09[0, 1] / ras_size[1]))
        ras_j_up = int(np.ceil(init_area_09[1, 1] / ras_size[1])) - 1

        ras_in_its = []

        if ras_i_right >= space_ras.shape[1]:
            ras_i_right = space_ras.shape[1] -1
        if ras_j_up >= space_ras.shape[2]:
            ras_j_up = space_ras.shape[2] - 1

        for i in range(ras_i_left, ras_i_right + 1, 1):
            for j in range(ras_j_low, ras_j_up + 1, 1):
                ras_ij = space_ras[:, i, j, 0]
                ras_ij_o = cs_09_2_cs_xy(np.expand_dims(ras_ij, axis=1), alpha_092o, np.expand_dims(uxy_092o, axis=1)).T.squeeze()
                x0, y0, r0 = local_polynomial_fitting(last_positions, ras_ij_o)
                dst_xy0 = np.linalg.norm((np.array([x0, y0]) - ras_ij_o), ord=2)
                if dst_xy0 <= r0:
                    ras_in_its.append([i, j])
        return np.array((ras_in_its))

    def generate_y_Upper_Lower(last_positions_e, xt):

        Lyt, Uyt = -1e-06, 1e-06
        diff_xt = np.abs(last_positions_e[:, 0] - xt)
        min_diff = np.argmin(diff_xt)
        y = last_positions_e[:, 1]
        one_ULyt = y[min_diff]
        if one_ULyt >= 0:
            Uyt = one_ULyt
            diff_xt[min_diff] = np.inf
            for i in range(len(diff_xt)):
                min_diff = np.argmin(diff_xt)
                one_ULyt = y[min_diff]
                if one_ULyt < 0:
                    Lyt = one_ULyt
                    break
                else:
                    diff_xt[min_diff] = np.inf
        else:
            Lyt = one_ULyt
            diff_xt[min_diff] = np.inf
            for i in range(len(diff_xt)):
                min_diff = np.argmin(diff_xt)
                one_ULyt = y[min_diff]
                if one_ULyt >= 0:
                    Uyt = one_ULyt
                    break
                else:
                    diff_xt[min_diff] = np.inf

        return Uyt, Lyt

    def prob_ras_k(t, to, Uxt, Lxt, Uyt, Lyt):
        # due to there is only origin, but the flying is directed, so we should build the movement model by considering
        # the expectation and variance. the expectation of variable x, y should be zero in rotational coordinate,
        # however, there is not destination, the tj is not defined, so,the variances of x,y are defined as follows,
        # which is equal to undirected movement model.

        def variance_xy(t, ti):
            return t - ti
        miu_x = 0
        miu_y = 0

        sigma_x2 = sigma_y2 = variance_xy(t, to)

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

    alpha_o209 = cal_alpha_092o(v0)
    last_positions_09 = cs_xy_2_cs_09(last_positions.T, alpha_o209, np.expand_dims(uxy_092o, axis=1)).T
    # circle_boundary_09 = cs_xy_2_cs_09(circle_boundary.T, alpha_092o, uxy_092o).T
    init_area_09 = generate_initial_ras_set(last_positions_09)
    ras_its = distinguishing_xy_in_srd(space_ras, ras_size, init_area_09, alpha_o209, uxy_092o, last_positions)

    if len(ras_its) < 1:
        return [], []
    else:
        space_its_09 = np.concatenate([np.expand_dims(space_ras[:, ras_its[k, 0], ras_its[k, 1], :], axis=1)
                                       for k in range(len(ras_its))], axis=1)

        wp_oe = np.array([uxy_092o, uxy_092e])
        alpha_092e = cal_alpha_092e(wp_oe)
        space_its_xy = cs_09_2_cs_xy(space_its_09, alpha_092e, np.expand_dims(uxy_092e, axis=1))
        last_positions_e = cs_09_2_cs_xy(last_positions_09.T, alpha_092e, np.expand_dims(uxy_092e, axis=1)).T
        # circle_boundary_e = cs_09_2_cs_xy(circle_boundary_09.T, alpha_092e, uxy_092e).T
        Lxt = np.min(last_positions_e[:, 0])
        Uxt = np.max(last_positions_e[:, 0])
        prob_its = []
        for k in range(len(ras_its)):
            space_k_xy = space_its_xy[:, k, :]
            k_xl = np.min(space_k_xy[0])
            k_xu = np.max(space_k_xy[0])
            k_yl = np.min(space_k_xy[1])
            k_yu = np.max(space_k_xy[1])

            Uyt, Lyt = generate_y_Upper_Lower(last_positions_e, xt=space_k_xy[0, 0])

            if k_xl < Lxt:
                k_xl = Lxt
            if k_xu > Uxt:
                k_xu = Uxt
            if k_yl < Lyt:
                k_yl = Lyt
            if k_yu > Uyt:
                k_yu = Uyt

            prob_fxy = prob_ras_k(t, to, Uxt, Lxt, Uyt, Lyt)

            prob_k_xy = cal_double_quad(prob_fxy, k_xl, k_xu, k_yl, k_yu)

            prob_k_hollow = cal_double_quad(prob_fxy,
                                            space_k_xy[0, 0] - abs(
                                                ras_size[0] * abs(np.cos(alpha_092e)) - ras_size[1] * abs(np.sin(alpha_092e))) / 2,
                                            space_k_xy[0, 0] + abs(
                                                ras_size[0] * abs(np.cos(alpha_092e)) - ras_size[1] * abs(np.sin(alpha_092e))) / 2,
                                            space_k_xy[1, 0] - abs(
                                                ras_size[1] * abs(np.cos(alpha_092e)) - ras_size[0] * abs(np.sin(alpha_092e))) / 2,
                                            space_k_xy[1, 0] + abs(
                                                ras_size[1] * abs(np.cos(alpha_092e)) - ras_size[0] * abs(np.sin(alpha_092e))) / 2)
            prob_its.append((prob_k_xy + prob_k_hollow) / 2)

        return ras_its, np.array(prob_its)


if __name__ == '__main__':

    print()
