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
from tools_function import coors2degree, coors2meters, pos2vel, cal_its_among_mul_its, cal_cft_prob_of_AB
from UAV_config import rotor_drone_info
from trajectory_pred_models import mean_vel_pred, cnn_model_pred, gru_model_pred, dgru_model_pred, kpf_pred

import tensorflow as tf
import gc
import operator
import multiprocessing as mp  # mp.pool是多进程池， 用cpu多个核心
from multiprocessing.dummy import Pool  # 这是多线程进程，意思就是用一个核心和多线程
from shapely.geometry import Point, Polygon

import time


def label_conflict_between_uavs(pos_can_use, t, D):
    """
    Args:
    :param posA_t: the position of A, B, in time t
    :param posB_t:
    :param D: the minimum distance allowed
    :return:
    """
    couple_idx, labels_AB = [], []
    for i in range(len(pos_can_use) - 1):
        for j in range(i, len(pos_can_use)):
            dst = np.linalg.norm((pos_can_use[i][t - 1] - pos_can_use[j][t - 1]), ord=2)
            if dst <= D:
                ab_label = 1
            else:
                ab_label = 0
            couple_idx.append([i, j])
            labels_AB.append(ab_label)

    return couple_idx, labels_AB


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


def is_used_to_recurrent(space_rased, ras_size, pred_tA, pred_tB, last_positionsA, last_positionsB, v0A, uxy_o209A, v0B,
                         uxy_o209B, t, to, D):
    uxy_092eA = pred_tA[0:2]
    rasA_index, prob_rasA = cal_prob_in_ras(space_rased, ras_size, last_positionsA, v0A, uxy_o209A, uxy_092eA, t, to)

    uxy_092eB = pred_tB[0:2]
    rasB_index, prob_rasB = cal_prob_in_ras(space_rased, ras_size, last_positionsB, v0B, uxy_o209B, uxy_092eB, t, to)

    if len(rasA_index) < 1 or len(rasB_index) < 1:
        cft_prob_AB = 0
    else:
        cft_prob_AB = prob_conflict_between_uavs(space_rased, rasA_index, prob_rasA, rasB_index, prob_rasB, D)

    return cft_prob_AB


def func1(i, pos, to, t):
    pred_t_mean = mean_vel_pred(pos[0:to], t_to=t - to)
    _, pred_t_pf = kpf_pred(pos[0:to + 1], space=1, out_len=t - to)
    a_pool_result = [pred_t_mean, pred_t_pf]
    return np.array(a_pool_result)


def func(k, space_rased, ras_size, preds_tA, preds_tB, last_positionsA, last_positionsB, v0A, uxy_o209A, v0B, uxy_o209B,
         t, to, D):
    a_pool_result = np.zeros(shape=[len(preds_tA)])
    for m in range(len(preds_tA)):
        a_pool_result[m] = is_used_to_recurrent(space_rased, ras_size, preds_tA[m], preds_tB[m], last_positionsA,
                                                last_positionsB, v0A, uxy_o209A, v0B, uxy_o209B, t, to, D)
    return a_pool_result


if __name__ == '__main__':

    to = 40
    t = 50

    # the setting of uav parameters
    rotor_info = rotor_drone_info()

    # read coordinates of peaks of rasterized air space, and set the number of raster
    root = r'.\conflict_detection\home'
    name = 'coors_info.xlsx'

    coors_as = pd.read_excel(root + '\\' + name, sheet_name='air_space', index_col=None, header=0)
    ll_as_abs = coors2degree(coors_as[['longitude', 'latitude']].values.T)
    origin = np.expand_dims(ll_as_abs[:, 0], axis=1)
    ll_as = coors2meters(ll_as_abs - origin, origin_lat=origin[1])
    grain_num = [100, 100]
    space_rased, ras_size = space_rasterizing(ll_as, grain_num)
    D = np.linalg.norm((np.array(ras_size) - np.array([0, 0])), ord=2) * grain_num[0] / 25

    # # read the target dataset used to predict the position in t, and determine if it intrudes into protected areas
    # with open(r'.\conflict_detection\origin_tracks\\segments_ts60.pkl', 'rb')as f:
    #     uav_segs = pickle.load(f)  # shape=[n, 40 + tmax, 3], and the coordinates is absolutely in global
    #
    # seg_iter = iter([i for i in uav_segs])
    # pos_can_use = []
    # for i, seg in enumerate(seg_iter):
    #     pos = coors2meters(seg.T - np.concatenate([origin, np.array([[0]])], axis=0), origin[1]).T
    #     if np.max(pos) > np.max(ll_as) or np.min(pos[:, 0:2]) < 0:
    #         break
    #     pos_can_use.append(pos)
    #
    # # predict the position firstly
    # num_cores = int(mp.cpu_count())
    # print("本地计算机有: " + str(num_cores) + " 核心")
    # pool = mp.Pool(num_cores - 2)
    #
    # pos_iter = iter(pos_can_use)
    # predictions = np.zeros(shape=(len(pos_can_use), 5, 3))
    # v_in_s = []
    # process_list, predictions_km = [], []
    # for k, pos in enumerate(pos_iter):
    #     v_in_s.append(pos2vel(pos))
    #     process = pool.apply_async(func1, args=(k, pos, to, t))
    #     process_list.append(process)
    #
    # for k in process_list:
    #     k.wait()
    #
    # for k in process_list:
    #     if k.ready():
    #         if k.successful():
    #             predictions_km.append(k.get())
    #
    # pool.close()
    # pool.join()
    #
    # pred_t_gru = gru_model_pred(np.array(v_in_s)[:, 0:to, :],
    #                             np.array(pos_can_use)[:, t - 1, :] - np.array(pos_can_use)[:, to - 1, :],
    #                             delta_t=t - to) + \
    #              np.array(pos_can_use)[:, to - 1]
    # pred_t_cnn = cnn_model_pred(np.array(v_in_s)[:, 0:to, :],
    #                             np.array(pos_can_use)[:, t - 1, :] - np.array(pos_can_use)[:, to - 1, :],
    #                             delta_t=t - to) + \
    #              np.array(pos_can_use)[:, to - 1]
    # pred_t_dgru = dgru_model_pred(np.array(v_in_s)[:, 0:to, :],
    #                             np.array(pos_can_use)[:, t - 1, :] - np.array(pos_can_use)[:, to - 1, :],
    #                               delta_t=t - to) + \
    #              np.array(pos_can_use)[:, to - 1]
    #
    # predictions[:, 0:2, :] = np.array(predictions_km)
    # predictions[:, 2, :] = pred_t_cnn
    # predictions[:, 3, :] = pred_t_gru
    # predictions[:, 4, :] = pred_t_dgru
    #
    # with open(r'.\conflict_detection\results' + '\\' + 'uav_bet_predictions4_' + str(t - to) + '.pkl', 'wb')as f:
    #     pickle.dump([pos_can_use, v_in_s, predictions], f)

    with open(r'.\conflict_detection\results' + '\\' + 'uav_bet_predictions4_' + str(t - to) + '.pkl', 'rb')as f:
        pos_can_use, v_in_s, predictions = pickle.load(f)

    # generate the couple flight segments from pos_can_use
    couple_idxes, labels_AB = label_conflict_between_uavs(pos_can_use, t, D)

    idx_cft = np.argwhere(np.array(labels_AB) > 0).squeeze()
    idx_all = [i for i in range(len(labels_AB))]
    idx_nc_all = list(set(idx_all) - set(list(idx_cft)))
    idx_nc = np.random.choice(np.array(idx_nc_all), len(idx_cft), replace=False)

    sam_idx1 = np.random.choice(np.arange(0, len(idx_cft)), 500, replace=False)
    sam_idx2 = np.random.choice(np.arange(0, len(idx_nc)), 500, replace=False)

    idx_used = list(idx_cft[sam_idx1]) + list(idx_nc[sam_idx2])
    labels_used = np.array(labels_AB)[idx_used]
    couple_iter = np.array((couple_idxes))[idx_used]

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores - 4)

    process_list, true_pred_labels = [], []
    time_avg = []

    for k, idx12 in enumerate(couple_iter[0:20]):
        [i, j] = idx12
        v10, v20 = v_in_s[i][to - 1, 0:2], v_in_s[j][to - 1, 0:2]
        xyo1, xyo2 = pos_can_use[i][to - 1, 0:2], pos_can_use[j][to - 1, 0:2]
        uxy1_o209, uxy2_o209 = xyo1, xyo2

        t_start = time.time()
        last_positions1 = generate_the_traj_all_possible(v10, rotor_info['vm'], rotor_info['am'], t, to, angle_delta=1,
                                                         is_plot=False)
        last_positions2 = generate_the_traj_all_possible(v20, rotor_info['vm'], rotor_info['am'], t, to, angle_delta=1,
                                                         is_plot=False)

        a_pool_result = np.zeros(shape=[len(predictions[i])])
        for m in range(len(predictions[i])):
            if m == 0:
                a_pool_result[m] = is_used_to_recurrent(space_rased, ras_size, predictions[i, m], predictions[j, m],
                                                        last_positions1,
                                                        last_positions2, v10, uxy1_o209, v20, uxy2_o209, t, to, D)
                time_avg.append(time.time() - t_start)

        true_pred_labels.append(a_pool_result)

    print(np.mean(time_avg))


    #     process = pool.apply_async(func, args=(k, space_rased, ras_size, predictions[i], predictions[j],
    #                                            last_positions1, last_positions2, v10, uxy1_o209, v20, uxy2_o209,
    #                                            t, to, D))
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
    # with open(r'.\conflict_detection\results' + '\\' + '2uav_bet4_' + str(t - to) + '_gn' + str(grain_num[0]) + '.pkl',
    #           'wb')as f:
    #     pickle.dump(np.concatenate([np.expand_dims(labels_used, axis=1), np.array(true_pred_labels)], axis=1), f)
