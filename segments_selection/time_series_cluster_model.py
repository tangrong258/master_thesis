from cluster_method import k_means_cluster, generate_rand_center, generate_sample_center, \
    update_mean_center, fuzzy_c_means_cluster, k_medoid_clara, OPTICS_cluster, extract_label
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def optics_2d_series(cluster_dataset, dst, mpts, is_precomputed=False, is_featuring=False, featuring_method=None):

    if isinstance(cluster_dataset, list):
        cluster_dataset = np.array(cluster_dataset)

    scaler = None

    if is_featuring:
        train_all_dis = []
        for t in range(cluster_dataset.shape[0]):
            feature = featuring_method(cluster_dataset[t])
            train_all_dis.append(feature)
        train_all_dis = np.array(train_all_dis)
        scaler = MinMaxScaler(feature_range=(0, 1))
        cluster_dataset = scaler.fit_transform(train_all_dis)

    orders, reach_dst = OPTICS_cluster(dst_matrix=None, data=cluster_dataset, dst_func=dst,
                                       min_pts=mpts, is_precomputed=is_precomputed)

    return orders, reach_dst, scaler


def k_clara_2d_series(cluster_dataset, cluster_num, dst, step_pam, steps,
                      generate_center=generate_sample_center, is_featuring=False, featuring_method=None):

    if isinstance(cluster_dataset, list):
        cluster_dataset = np.array(cluster_dataset)

    scaler = None

    if is_featuring:
        train_all_dis = []
        for t in range(cluster_dataset.shape[0]):
            feature = featuring_method(cluster_dataset[t])
            train_all_dis.append(feature)
        train_all_dis = np.array(train_all_dis)
        scaler = MinMaxScaler(feature_range=(0, 1))
        cluster_dataset = scaler.fit_transform(train_all_dis)

    index, center, sil_score = k_medoid_clara(dataset=cluster_dataset, k=cluster_num, dis_func=dst,
                                              generate_center=generate_center, step_pam=step_pam, steps=steps)
    return index, center, sil_score, scaler


def k_means_2d_series(cluster_dataset, cluster_num, dst, stop_rate,
                      generate_center=generate_rand_center, update_center=update_mean_center,
                      is_featuring=False,
                      featuring_method=None):
    if isinstance(cluster_dataset, list):
        cluster_dataset = np.array(cluster_dataset)

    scaler = None

    if is_featuring:
        train_all_dis = []
        for t in range(cluster_dataset.shape[0]):
            feature = featuring_method(cluster_dataset[t])
            train_all_dis.append(feature)
        train_all_dis = np.array(train_all_dis)
        scaler = MinMaxScaler(feature_range=(0, 1))
        cluster_dataset = scaler.fit_transform(train_all_dis)

    index, center, sil_score = k_means_cluster(dataset=cluster_dataset, k=cluster_num, dis_func=dst,
                                               generate_center=generate_center,
                                               update_center=update_center,
                                               stop_rate=stop_rate)
    return index, center, sil_score, scaler


def inverse_k_means_2d_series(cluster_dataset, cluster_num, dst, stop_rate,
                              generate_center=generate_rand_center, update_center=update_mean_center,
                              is_featuring=False,
                              featuring_method=None):
    if isinstance(cluster_dataset, list):
        cluster_dataset = np.array(cluster_dataset)

    scaler = None

    if is_featuring:
        train_all_dis = []
        for t in range(cluster_dataset.shape[0]):
            feature = featuring_method(cluster_dataset[t])
            train_all_dis.append(feature)
        train_all_dis = np.array(train_all_dis)
        scaler = MinMaxScaler(feature_range=(0, 1))
        cluster_dataset = scaler.fit_transform(train_all_dis)

    index, center, sil_score = k_means_cluster(dataset=cluster_dataset, k=cluster_num, dis_func=dst,
                                               generate_center=generate_center,
                                               update_center=update_center,
                                               stop_rate=stop_rate)
    # 对于center[i], index为[index[j] for j in range(len(index)), except j=i]
    c_num = len(center)
    index_non_with_center = []
    for i in range(c_num):
        index_non_i = []
        for j in range(c_num):
            if i != j:
                index_non_i += list(index[j])
        index_non_with_center.append(index_non_i)

    return index_non_with_center, center, sil_score, scaler


def fuzzy_c_means_2d_series(cluster_dataset, cluster_num, dst, stop_rate, p,
                            generate_center=generate_rand_center,
                            is_featuring=False,
                            featuring_method=None):
    if isinstance(cluster_dataset, list):
        cluster_dataset = np.array(cluster_dataset)

    scaler = None

    if is_featuring:
        train_all_dis = []
        for t in range(cluster_dataset.shape[0]):
            feature = featuring_method(cluster_dataset[t])
            train_all_dis.append(feature)
        train_all_dis = np.array(train_all_dis)
        scaler = MinMaxScaler(feature_range=(0, 1))
        cluster_dataset = scaler.fit_transform(train_all_dis)

    index, center, sil_score = fuzzy_c_means_cluster(dataset=cluster_dataset, k=cluster_num, dis_func=dst,
                                                     generate_center=generate_center,
                                                     stop_rate=stop_rate, p=p)
    return index, center, sil_score, scaler


def out_is_weighted_series(center, series, is_weighted, dst_func, scaler, is_featuring=False, featuring_method=None):
    if is_featuring:
        series = featuring_method(series)
        series = scaler.transform([series])[0]

    if is_weighted:
        weights = []
        for i in range(len(center)):
            dis = dst_func(center[i], series)
            weights.append(1 / dis)
        return np.array(weights / np.sum(weights))
    else:
        weights = np.zeros(shape=(len(center),))
        dis_center = []
        for i in range(len(center)):
            dis = dst_func(center[i], series)
            dis_center.append(dis)
        min_idx = np.argmin(dis_center)
        weights[min_idx] = 1
        return np.array(weights)


def out_is_weighted_inverse_series(center, series, is_weighted, dst_func, scaler, is_featuring=False, featuring_method=None):
    if is_featuring:
        series = featuring_method(series)
        series = scaler.transform([series])[0]

    weights = np.zeros(shape=(len(center),))
    dis_center = []
    for i in range(len(center)):
        dis = dst_func(center[i], series)
        dis_center.append(dis)
    min_idx = np.argmax(dis_center)
    weights[min_idx] = 1
    return np.array(weights)