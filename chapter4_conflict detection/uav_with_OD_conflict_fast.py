
import numpy as np
import pandas as pd
import pickle
import math
import scipy.integrate as si
from scipy.optimize import fsolve
# only used to solve the eq
from matplotlib import pyplot as plt

from sympy import *
import sympy as sy

from space_rasterization import space_rasterizing
from UAV_traj_boundary_with_initial_velocity import generate_the_traj_all_possible, cal_prob_in_ras
from UAV_traj_boundary_with_OD import cal_prob_in_ras_OD
from tools_function import coors2degree, coors2meters, pos2vel, cal_its_among_mul_its, cal_cft_prob_of_AB
from UAV_config import rotor_drone_info
from trajectory_pred_models import mean_vel_pred, cnn_model_pred, gru_model_pred, kpf_pred

import tensorflow as tf
import gc
import operator
import multiprocessing as mp  # mp.pool是多进程池， 用cpu多个核心
from multiprocessing.dummy import Pool  # 这是多线程进程，意思就是用一个核心和多线程
from shapely.geometry import Point, Polygon
import time


def generate_od_flying_trajectory(pos_o, pos_d, to, td, vm, N):
    trajectory_od = []
    vector = [pos_d[0] - pos_o[0], pos_d[1] - pos_o[1]]
    alpha = np.arctan((vector[1]) / (vector[0]) + 1.0e-06)
    if vector[0] < 0:
        alpha += np.pi
    for i in range(N):
        traj_i = np.zeros(shape=(td-to + 1, 3))
        alpha_1 = alpha + np.random.randn() * (5/180 * np.pi / 3)
        v1 = (vm / 2) + np.random.randn() * (vm / (2 * 3))
        if alpha_1 > alpha + 5:
            alpha_1 = alpha + 5
        if alpha_1 < alpha_1 - 5:
            alpha_1 = alpha_1 - 5

        if v1 > vm:
            v1 = vm
        if v1 < 0:
            v1 = 0

        for t in range(to, int((td - to) / 2)+1, 1):
            traj_i[t-to, 0] = v1 * (t - to) * np.cos(alpha_1) + pos_o[0]
            traj_i[t-to, 1] = v1 * (t - to) * np.sin(alpha_1) + pos_o[1]
            traj_i[t - to, 2] = t

        v2 = (np.linalg.norm((pos_d - pos_o), ord=2) - int((td - to) / 2) * v1) / (td - int((td - to) / 2))
        vector_2 = [pos_d[0] - traj_i[int((td - to) / 2), 0], pos_d[1] - traj_i[int((td - to) / 2), 1]]
        alpha_2 = np.arctan((vector[1]) / (vector[0]) + 1.0e-06)
        if vector_2[0] < 0:
            alpha_2 += np.pi

        for t in range(int((td - to) / 2)+1, td + 1, 1):
            traj_i[t-to, 0] = v2 * (t - int((td - to) / 2)) * np.cos(alpha_2) + traj_i[int((td - to) / 2), 0]
            traj_i[t-to, 1] = v2 * (t - int((td - to) / 2)) * np.sin(alpha_2) + traj_i[int((td - to) / 2), 1]
            traj_i[t - to, 2] = t

        # plt.plot(traj_i[:, 0], traj_i[:, 1])
        # plt.show()

        trajectory_od.append(traj_i)
    return trajectory_od


def label_conflict_between_uav_od(pos_can_use, ts, tl, traj_od, to, td, D):

    tp_od, B, labels_AB = [], [], []
    for j in range(len(traj_od)):
        y_p = (traj_od[j][-1, 1] - traj_od[j][0, 1]) / 2 + np.random.randn() * ((traj_od[j][-1, 1] - traj_od[j][0, 1]) / (4 * 6))
        diff_y = traj_od[j][:, 1] - (y_p + traj_od[j][0, 1])
        id_min = np.argmin(abs(diff_y))
        tp = int(traj_od[j][id_min, -1])
        if tp > td:
            tp = td - tl
        if tp < to:
            tp = to
        tp_od.append(tp)
        for i in range(len(pos_can_use)):
            dst = np.linalg.norm((pos_can_use[i][ts + tl - 1] - traj_od[j][tp]), ord=2)
            if dst <= D:
                ab_label = 1
            else:
                ab_label = 0
            B.append(j)
            labels_AB.append(ab_label)
    return tp_od, B, labels_AB


def cal_all_od_flight_spd(space_rased, ras_size, traj_od, t_pred, vm):
    ras_index_od, prob_ras_od = [], []
    for j in range(len(t_pred)):
        ras_index, prob_ras = cal_prob_in_ras_OD(space_rased, ras_size, traj_od[j][[0, -1]], vm, t_pred[j])
        ras_index_od.append(ras_index)
        prob_ras_od.append(prob_ras)
    return ras_index_od, prob_ras_od


def prob_conflict_between_uavs(space_rased, ras_itsA, prob_rasA, ras_itsB, prob_rasB, D):
    """
    args: idea, judge all ras in protected area, then sum which prob_ras, obtain the prob uav in protected area.
    :param space_ras_uav: shape=[2, len(ras_uav), 5]
    :param prob_ras: shape=[len(ras_uav)]
    :return:
    """
    cflt_ras, cflt_ras_whereA, cflt_who, cflt_where = cal_its_among_mul_its(ras_tgt=ras_itsA, ras_in_its_set=[ras_itsB],
                                                                            prob_tgt=prob_rasA, prob_set=[prob_rasB],
                                                                            n=1)
    if not cflt_ras:
        cft_prob_AB = 0
    else:
        cft_prob_AB = cal_cft_prob_of_AB(space_rased, cflt_ras, cflt_ras_whereA, [i[0] for i in cflt_where],
                                         prob_rasA, prob_rasB, D)

    return cft_prob_AB


def is_used_to_recurrent(space_rased, ras_size, pred_tA, last_positionsA, v0A, uxy_o209A,
                         ras_od, prob_od, ts_in_out, ts, D):
    uxy_092eA = pred_tA[0:2]
    rasA_index, prob_rasA = cal_prob_in_ras(space_rased, ras_size, last_positionsA, v0A, uxy_o209A, uxy_092eA, ts_in_out, ts)

    if len(rasA_index) < 1 or len(ras_od) < 1:
        cft_prob_AB = 0
    else:
        cft_prob_AB = prob_conflict_between_uavs(space_rased, rasA_index, prob_rasA, ras_od, prob_od, D)

    return cft_prob_AB


def func(k, space_rased, ras_size, preds_tA, last_positionsA, v0A, uxy_o209A, ras_od, prob_od, ts_in_out, ts, D):
    a_pool_result = np.zeros(shape=[len(preds_tA)])
    for m in range(len(preds_tA)):
        a_pool_result[m] = is_used_to_recurrent(space_rased, ras_size, preds_tA[m], last_positionsA,
                                                v0A, uxy_o209A, ras_od, prob_od, ts_in_out, ts, D)
    return a_pool_result


if __name__ == '__main__':

    ts = 40
    tl = 20
    ts_in_out = ts + tl

    # the setting of uav parameters
    rotor_info = rotor_drone_info()

    # read coordinates of peaks of rasterized air space, and set the number of raster
    root = r'.\conflict_detection\home'
    name = 'coors_info.xlsx'

    coors_as = pd.read_excel(root + '\\' + name, sheet_name='air_space', index_col=None, header=0)
    ll_as_abs = coors2degree(coors_as[['longitude', 'latitude']].values.T)
    origin = np.expand_dims(ll_as_abs[:, 0], axis=1)
    ll_as = coors2meters(ll_as_abs - origin, origin_lat=origin[1])
    grain_num = [50, 50]
    space_rased, ras_size = space_rasterizing(ll_as, grain_num)
    D = np.linalg.norm((np.array(ras_size) - np.array([0, 0])), ord=2) * grain_num[0] / 25

    coors_od2 = pd.read_excel(root + '\\' + name, sheet_name='od', index_col=None, header=0)
    ll_od2_abs = coors2degree(coors_od2[['longitude', 'latitude']].values.T)
    ll_od2 = coors2meters(ll_od2_abs - origin, origin[1])

    # generate the o-d flights
    ti = 0
    tj = 200

    traj_od = generate_od_flying_trajectory(pos_o=ll_od2.T[0], pos_d=ll_od2.T[1], to=ti, td=tj, vm=10, N=1)

    with open(r'.\conflict_detection\results' + '\\' + 'uav_bet_predictions4_' + str(tl) + '.pkl', 'rb')as f:
        pos_can_use, v_in_s, predictions = pickle.load(f)

    t_pred, OD_idx, labels_AB = label_conflict_between_uav_od(pos_can_use, ts, tl, traj_od, ti, tj, D)

    idx_cft = np.argwhere(np.array(labels_AB) > 0).squeeze()
    idx_all = [i for i in range(len(labels_AB))]
    idx_nc_all = list(set(idx_all) - set(list(idx_cft)))
    idx_nc = np.random.choice(np.array(idx_nc_all), len(idx_cft), replace=False)

    idx_used = list(idx_cft) + list(idx_nc)
    OD_used = np.array(OD_idx)[idx_used]
    labels_used = np.array(labels_AB)[idx_used]

    tt1 = time.time()
    ras_index_od, prob_ras_od = cal_all_od_flight_spd(space_rased, ras_size, traj_od, t_pred, vm=10)
    print(time.time() - tt1)

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(int(num_cores / 2))

    process_list, true_pred_labels = [], []

    time_avg = []

    for k, i in enumerate(idx_used[0:20]):
        v10 = v_in_s[i][ts - 1, 0:2]
        xyo1 = pos_can_use[i][ts - 1, 0:2]
        uxy1_o209 = xyo1

        t_start = time.time()
        last_positions1 = generate_the_traj_all_possible(v10, rotor_info['vm'], rotor_info['am'], ts_in_out, ts, angle_delta=1,
                                                         is_plot=False)

        ras_od, prob_od = ras_index_od[OD_used[k]], prob_ras_od[OD_used[k]]

        a_pool_result = np.zeros(shape=[len(predictions[k])])
        for m in range(len(predictions[k])):
            if m == 0:
                a_pool_result[m] = is_used_to_recurrent(space_rased, ras_size, predictions[i, m], last_positions1,
                                                        v10, uxy1_o209, ras_od, prob_od, ts_in_out, ts, D)
                time_avg.append(time.time() - t_start)
        true_pred_labels.append(a_pool_result)

    print(np.mean(time_avg))

    #     process = pool.apply_async(func, args=(k, space_rased, ras_size, predictions[i],
    #                                            last_positions1, v10, uxy1_o209, ras_od, prob_od,
    #                                            ts_in_out, ts, D))
    #     process_list.append(process)
    #
    # for k in process_list:
    #     k.wait()
    #
    # for k in process_list:
    #     if k.ready():
    #         if k.successful():
    #             true_pred_labels.append(k.get())
    #             print(len(true_pred_labels))
    #
    # pool.close()
    # pool.join()
    #
    # with open(r'.\conflict_detection\results' + '\\' + '2uav_od4_' + str(tl) + '_gn' + str(grain_num[0]) + '.pkl', 'wb')as f:
    #     pickle.dump(np.concatenate([np.expand_dims(labels_used, axis=1), np.array(true_pred_labels)], axis=1), f)
