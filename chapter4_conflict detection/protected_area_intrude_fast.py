# when the ras of uav and probobility of each are built, the probobility of uav flying into protected area will calculated
# by sum the probobility of ras, which center located in the protected area
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
from tools_function import coors2degree, coors2meters, pos2vel
from UAV_config import rotor_drone_info
from trajectory_pred_models import mean_vel_pred, cnn_model_pred, gru_model_pred, dgru_model_pred, kpf_pred

import tensorflow as tf
import gc
import operator
import multiprocessing as mp  # mp.pool是多进程池， 用cpu多个核心
from multiprocessing.dummy import Pool  # 这是多线程进程，意思就是用一个核心和多线程
from shapely.geometry import Point, Polygon
import time

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_virtual_device_configuration(gpu,
                                                            [tf.config.experimental.VirtualDeviceConfiguration
                                                             (memory_limit=3036)])


def label_uav_in_protected_area(pos_t, ptd_area):
    """
    args:
    :param uav_segments: shape=[m, 3], the coordinates of uav at t moment
    :param ptd_area: shape=[2, n], n is the pink number of protected area, and the sequence is followed counterclockwise
    :return:
    """
    # ptd_area_circle = np.concatenate([ptd_area, np.expand_dims(ptd_area[:, 0], axis=1)], axis=1)  # [2, n+1]
    # center = pos_t
    #
    # ejk_set = []
    # for j in range(ptd_area.shape[1]):
    #     ejk = (center[0] - ptd_area_circle[0, j]) * (ptd_area_circle[1, j + 1] - ptd_area_circle[1, j]) - \
    #           (center[1] - ptd_area_circle[1, j]) * (ptd_area_circle[0, j + 1] - ptd_area_circle[0, j])
    #     ejk_set.append(ejk)
    # if np.min(ejk_set) >= 0:
    #     seg_label = 1
    # else:
    #     seg_label = 0

    p1 = Point(pos_t[0], pos_t[1])
    poly = Polygon([(i[0], i[1]) for i in ptd_area.T])
    is_in = p1.within(poly)
    if is_in:
        seg_label = 1
    else:
        seg_label = 0

    return seg_label


def prob_uav_in_protected_area(space_ras, ras_its, pro_ras, ptd_area):
    """
    args: idea, judge all ras in protected area, then sum which prob_ras, obtain the prob uav in protected area.
    :param space_ras_uav: shape=[2, len(ras_uav), 5]
    :param pro_ras: shape=[len(ras_uav)]
    :param ptd_area: shape=[2, n], n is the pink number of protected area, and the sequence is followed counterclockwise
    :return:
    """
    space_ras_uav = np.concatenate([np.expand_dims(space_ras[:, ras_its[k, 0], ras_its[k, 1], :], axis=1)
                                   for k in range(len(ras_its))], axis=1)
    # ptd_area_circle = np.concatenate([ptd_area, np.expand_dims(ptd_area[:, 0], axis=1)], axis=1)  # [2, n+1]
    #
    # ras_kept = []
    # for i in range(space_ras_uav.shape[1]):
    #     center = space_ras_uav[:, i, 0]
    #
    #     ejk_set = []
    #     for j in range(ptd_area.shape[1]):
    #         ejk = (center[0] - ptd_area_circle[0, j]) * (ptd_area_circle[1, j + 1] - ptd_area_circle[1, j]) - \
    #               (center[1] - ptd_area_circle[1, j]) * (ptd_area_circle[0, j + 1] - ptd_area_circle[0, j])
    #         ejk_set.append(ejk)
    #     if np.min(ejk_set) >= 0:
    #         ras_kept.append(i)

    ras_kept = []
    for i in range(space_ras_uav.shape[1]):
        center = space_ras_uav[:, i, 0]
        p1 = Point(center[0], center[1])
        poly = Polygon([(i[0], i[1]) for i in ptd_area.T])
        is_in = p1.within(poly)
        if is_in:
            ras_kept.append(i)

    return np.sum(pro_ras[ras_kept])


def is_used_to_recurrent(pred_t, space_rased, ras_size, last_positions, v0, uxy_o209, t, to, ll_pa2):

    uxy_092e = pred_t[0:2]
    ras_index, prob_ras = cal_prob_in_ras(space_rased, ras_size, last_positions, v0, uxy_o209, uxy_092e, t, to)
    if len(ras_index) < 1:
        prob_intrude2 = 0
    else:
        # prob_intrude1 = prob_uav_in_protected_area(space_rased, ras_index, prob_ras, ll_pa1)
        prob_intrude2 = prob_uav_in_protected_area(space_rased, ras_index, prob_ras, ll_pa2)

    del ras_index, prob_ras
    gc.collect()

    return prob_intrude2


def func(i, preds, space_rased, ras_size, last_positions, v0, uxy_o209, t, to, ll_pa2):
    a_pool_result = np.zeros(shape=[len(preds)])
    for j in range(len(preds)):
        a_pool_result[j] = is_used_to_recurrent(preds[j], space_rased, ras_size, last_positions, v0, uxy_o209, t, to, ll_pa2)
    return a_pool_result


if __name__ == '__main__':

    to = 40
    t = 60
    # the setting of uav parameters
    rotor_info = rotor_drone_info()

    # read coordinates of peaks of rasterized air space, and set the number of raster
    root = r'.\conflict_detection\home'
    name = 'coors_info.xlsx'

    coors_as = pd.read_excel(root + '\\' + name, sheet_name='air_space', index_col=None, header=0)
    ll_as_abs = coors2degree(coors_as[['longitude', 'latitude']].values.T)
    origin = np.expand_dims(ll_as_abs[:, 0], axis=1)
    ll_as = coors2meters(ll_as_abs - origin, origin_lat=origin[1])
    grain_num = [200, 200]
    space_rased, ras_size = space_rasterizing(ll_as, grain_num)

    coors_pa2 = pd.read_excel(root + '\\' + name, sheet_name='fixed_area2', index_col=None, header=0)
    ll_pa2_abs = coors2degree(coors_pa2[['longitude', 'latitude']].values.T)
    ll_pa2 = coors2meters(ll_pa2_abs - origin, origin[1])


    # # read the target dataset used to predict the position in t, and determine if it intrudes into protected areas
    # with open(r'.\conflict_detection\origin_tracks\\segments_ts60.pkl', 'rb')as f:
    #     uav_segs = pickle.load(f)  # shape=[n, 40 + tmax, 3], and the coordinates is absolutely in global
    #
    # true_labels2 = []
    # pos_can_use = []
    # seg_iter = iter([i for i in uav_segs])
    #
    # for i, seg in enumerate(seg_iter):
    #     pos = coors2meters(seg.T - np.concatenate([origin, np.array([[0]])], axis=0), origin[1]).T
    #     label2 = label_uav_in_protected_area(pos[t - 1, 0:2], ll_pa2)
    #
    #     if np.max(pos) > np.max(ll_as) or np.min(pos[:, 0:2]) < 0:
    #         break
    #
    #     pos_can_use.append(pos)
    #     true_labels2.append(label2)
    #
    # idx_p2 = np.argwhere(np.array(true_labels2) > 0).squeeze()
    # idx_2 = [i for i in range(len(true_labels2))]
    # idx_n2_all = list(set(idx_2) - set(list(idx_p2)))
    # idx_n2 = np.random.choice(np.array(idx_n2_all), len(idx_p2), replace=False)
    #
    # idx_used2 = list(idx_p2) + list(idx_n2)
    # labels_2 = np.array(true_labels2)[idx_used2]
    # pos_iter = iter([i for i in np.array(pos_can_use)[idx_used2]])
    # predictions = np.zeros(shape=(len(idx_used2), 5, 3))
    # v_in_s = []
    #
    # for i, pos in enumerate(pos_iter):
    #     v_in = pos2vel(pos)
    #     v_in_s.append(v_in)
    #     pred_t_mean = mean_vel_pred(pos[0:to], t_to=t - to)
    #     _, pred_t_pf = kpf_pred(pos[0:to+1], space=1, out_len=t-to)
    #     predictions[i, 0, :] = pred_t_mean
    #     predictions[i, 1, :] = pred_t_pf
    #
    # pred_t_cnn = cnn_model_pred(np.array(v_in_s)[:, 0:to, :],
    #                             np.array(pos_can_use)[idx_used2][:, t - 1, :] - np.array(pos_can_use)[idx_used2][:, to - 1, :],
    #                             delta_t=t - to) + \
    #              np.array(pos_can_use)[idx_used2, to-1]
    # pred_t_gru = gru_model_pred(np.array(v_in_s)[:, 0:to, :],
    #                             np.array(pos_can_use)[idx_used2][:, t - 1, :] - np.array(pos_can_use)[idx_used2][:, to - 1, :],
    #                             delta_t=t - to) + \
    #              np.array(pos_can_use)[idx_used2, to-1]
    # pred_t_dgru = dgru_model_pred(np.array(v_in_s)[:, 0:to, :],
    #                             np.array(pos_can_use)[idx_used2][:, t - 1, :] - np.array(pos_can_use)[idx_used2][:,
    #                                                                             to - 1, :], delta_t=t - to) + \
    #              np.array(pos_can_use)[idx_used2, to - 1]
    #
    # predictions[:, 2, :] = pred_t_cnn
    # predictions[:, 3, :] = pred_t_gru
    # predictions[:, 4, :] = pred_t_dgru
    #
    # with open(r'.\conflict_detection\results' + '\\' + 'area2_predictions4_' + str(t - to) + '.pkl', 'wb')as f:
    #     pickle.dump([np.array(pos_can_use)[idx_used2], np.array(v_in_s), np.array((predictions)), labels_2], f)
    #

    with open(r'.\conflict_detection\results' + '\\' + 'area2_predictions4_' + str(t - to) + '.pkl', 'rb')as f:
        [poss, v_ins, preds, labels] = pickle.load(f)

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")

    pool = mp.Pool(int(num_cores / 2))
    process_list, true_pred_labels = [], []

    for i, pos in enumerate(poss):

        v0 = v_ins[i][to - 1, 0:2]
        xyo = pos[to - 1, 0:2]
        uxy_o209 = xyo

        last_positions = generate_the_traj_all_possible(v0, rotor_info['vm'], rotor_info['am'], t, to, angle_delta=1,
                                                        is_plot=False)

        process = pool.apply_async(func, args=(i, preds[i], space_rased, ras_size, last_positions, v0, uxy_o209, t, to, ll_pa2))
        process_list.append(process)

    for i in process_list:
        i.wait()

    for i in process_list:
        if i.ready():
            if i.successful():
                true_pred_labels.append(i.get())
                print(len(true_pred_labels))

    pool.close()
    pool.join()

    with open(r'.\conflict_detection\results' + '\\' + '2fixed4_area2_' + str(t - to) + '_gn' + str(grain_num[0]) + '.pkl', 'wb')as f:
        pickle.dump(np.concatenate([np.expand_dims(labels, axis=1), np.array(true_pred_labels)], axis=1), f)

    true_pred_labels = []
    time_avg = []
    for i, pos in enumerate(poss[0:20]):
        v_in = v_ins[i]
        v0 = v_in[to-1, 0:2]
        xyo = pos[to-1, 0:2]
        uxy_o209 = xyo

        t_start = time.time()

        last_positions = generate_the_traj_all_possible(v0, rotor_info['vm'], rotor_info['am'], t, to, angle_delta=1, is_plot=False)

        prob2_mean = is_used_to_recurrent(preds[i, 0], space_rased, ras_size, last_positions, v0, uxy_o209, t, to, ll_pa2)
        time_avg.append(time.time() - t_start)


    print(np.mean(time_avg))


