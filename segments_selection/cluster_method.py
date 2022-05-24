from distance_function_s2s import dst_euc, dtw
import numpy as np
import operator
import copy
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def hopkins_statistic(dataset):
    if len(dataset.shape) == 1:
        dataset = np.expand_dims(dataset, axis=1)
    sample_size = int(dataset.shape[0] * 0.1)

    selected_indices = random.sample(range(0, dataset.shape[0], 1), sample_size)
    dataset_sample = dataset[selected_indices]

    neigh = NearestNeighbors(n_neighbors=2)  # 由于dataset_sample是dataset中的点，它的最近阶会是自己
    nbrs = neigh.fit(dataset)  # 构建匹配搜索域，即在dataset里面寻找最近邻接

    dis_100 = []
    for i in range(100):
        set_uniform_random_sample = np.random.uniform(np.min(dataset, axis=0),
                                                      np.max(dataset, axis=0),
                                                      (sample_size, dataset.shape[1]))

        u_dis, u_idx = nbrs.kneighbors(set_uniform_random_sample, n_neighbors=2)
        d_dis, d_idx = nbrs.kneighbors(dataset_sample, n_neighbors=2)

        u_dis_sum = np.sum(u_dis[:, 1])
        d_dis_sum = np.sum(d_dis[:, 1])

        dis_100.append(u_dis_sum / (u_dis_sum + d_dis_sum))

    return np.mean(dis_100)


def silhouette_score_each(samples, dst_each_sam, index_each_center, centers, dst_func, is_each):
    score = [[] for _ in range(len(centers))]
    for i in range(len(centers)):
        center_excepted = np.delete(np.copy(centers), obj=i, axis=0)
        a = np.mean(dst_each_sam[index_each_center[i]])
        for k in range(len(index_each_center[i])):
            dst_sam_center = []
            for j in range(len(center_excepted)):
                dst_sam_center.append(dst_func(samples[index_each_center[i][k]], center_excepted[j]))
            b = np.min(dst_sam_center)
            score_k_in_i = (b - a) / np.max([a, b])
            score[i].append(score_k_in_i)
    if is_each:
        return score
    else:
        score_list = []
        [score_list.extend(i) for i in score]
        return np.mean(score_list)


def out_is_weighted(center, series, is_weighted, dst_func):
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


def out_is_weighted_inverse(center, series, is_weighted, dst_func):

    weights = np.zeros(shape=(len(center),))
    dis_center = []
    for i in range(len(center)):
        dis = dst_func(center[i], series)
        dis_center.append(dis)
    min_idx = np.argmax(dis_center)
    weights[min_idx] = 1
    return np.array(weights)


def generate_rand_center(dataset, k):
    one_data = dataset[0]
    if len(one_data.shape) < 2:
        center_coor = [np.zeros(shape=(1, len(one_data))) for _ in range(k)]
        center_coor = np.concatenate(center_coor, axis=0)
        for j in range(len(one_data)):
            min_f = np.min(dataset[:, j])
            max_f = np.max(dataset[:, j])
            range_f = float(max_f - min_f)
            center_coor[:, j] = min_f + range_f * np.random.rand(k)

    else:
        center_coor = [np.zeros_like(one_data) for _ in range(k)]
        center_coor = np.array(center_coor)
        for j in range(one_data.shape[0]):
            for i in range(one_data.shape[1]):
                min_f = np.min(dataset[:, j, i])
                max_f = np.max(dataset[:, j, i])
                range_f = float(max_f - min_f)
                center_coor[:, j, i] = min_f + range_f * np.random.rand(k)

    return center_coor


def generate_sample_center(dataset, k):
    center_index = random.sample(range(0, len(dataset), 1), k)
    return dataset[center_index]


def update_mean_center(center_set):
    return np.mean(center_set, axis=0)


def dst_center_data(center, data, dis_func):
    cluster_ass = np.zeros(shape=[len(data), 2])
    for i in range(len(data)):
        dst = []
        for j in range(len(center)):
            dst.append(dis_func(center[j], data[i]))
        cluster_ass[i, 0] = np.argmin(dst)
        cluster_ass[i, 1] = np.min(dst)
    return cluster_ass


def k_medoid_pam(dataset, k, dis_func, generate_center, step_pam, is_pam):
    sample_num = dataset.shape[0]
    center_coor = generate_center(dataset, k)
    cluster_ass = dst_center_data(center_coor, dataset, dis_func)

    loss_all = np.ones(shape=(sample_num, k))

    for i in range(step_pam):
        for m in range(k):
            for l in range(sample_num):
                center_now = np.copy(center_coor)
                center_now[m] = dataset[l]
                cluster_ass_now = dst_center_data(center_now, dataset, dis_func)
                loss_all[l, m] = np.sum(cluster_ass_now[:, 1] - cluster_ass[:, 1])

        if np.min(loss_all) < 0:
            index_alter = np.argwhere(loss_all == np.min(loss_all))
            index_l = index_alter[0, 0]
            index_m = index_alter[0, 1]
            center_coor[index_m] = dataset[index_l]
            cluster_ass = dst_center_data(center_coor, dataset, dis_func)
        else:
            break

    if is_pam:
        index_each_cluster = []
        for i in range(k):
            index_each_cluster.append(np.argwhere(cluster_ass[:, 0] == i).squeeze())

        silhouette_score = silhouette_score_each(dataset, cluster_ass[:, 1], index_each_cluster, center_coor, dis_func,
                                                 is_each=False)
        return index_each_cluster, center_coor, silhouette_score
    else:
        return center_coor


def k_medoid_clara(dataset, k, dis_func, generate_center, steps, step_pam):
    sample_num = dataset.shape[0]
    cluster_ass = np.ones(shape=[sample_num, 2]) * float('inf')
    center_coor = generate_center(dataset, k)
    for i in range(steps):
        index_sub = random.sample(range(0, len(dataset)), 40 + 2*k)
        center_pam = k_medoid_pam(dataset[index_sub], k, dis_func, generate_center=generate_center, step_pam=step_pam, is_pam=False)
        cluster_ass_now = dst_center_data(center_pam, dataset, dis_func)
        if np.mean(cluster_ass_now[:, 1]) < np.mean(cluster_ass[:, 1]):
            cluster_ass = cluster_ass_now
            center_coor = center_pam

    index_each_cluster = []
    for i in range(k):
        index_each_cluster.append(np.argwhere(cluster_ass[:, 0] == i).squeeze())

    silhouette_score = silhouette_score_each(dataset, cluster_ass[:, 1], index_each_cluster, center_coor, dis_func,
                                             is_each=False)
    return index_each_cluster, center_coor, silhouette_score


def k_means_cluster(dataset, k, dis_func, generate_center, update_center, stop_rate):
    sample_num = dataset.shape[0]
    cluster_ass = np.zeros(shape=[sample_num, 2])
    center_coor = generate_center(dataset, k)
    cluster_chg = True
    while cluster_chg:
        index_chg_num = 0
        for i in range(sample_num):
            min_idx = -1
            min_dst = float('inf')
            for j in range(k):
                dst_i2j = dis_func(dataset[i], center_coor[j])
                if dst_i2j < min_dst:
                    min_dst = dst_i2j
                    min_idx = j
            if cluster_ass[i, 0] != min_idx:
                index_chg_num += 1
            cluster_ass[i] = [min_idx, min_dst]

        if index_chg_num / sample_num <= stop_rate:
            cluster_chg = False

        for cent in range(k):
            cent_set = dataset[np.nonzero(cluster_ass[:, 0] == cent)[0]]
            center_coor[cent, :] = update_center(cent_set)

    index_each_cluster = []
    for i in range(k):
        index_each_cluster.append(np.argwhere(cluster_ass[:, 0] == i).squeeze())

    silhouette_score = silhouette_score_each(dataset, cluster_ass[:, 1], index_each_cluster, center_coor, dis_func,
                                             is_each=False)
    return index_each_cluster, center_coor, silhouette_score


def fuzzy_c_means_cluster(dataset, k, dis_func, generate_center, stop_rate, p=2.0):
    sample_num = dataset.shape[0]
    cluster_ass = np.zeros(shape=[sample_num, 3])
    center_coor = generate_center(dataset, k)
    weights = np.random.rand(sample_num, k)
    cluster_chg = True

    while cluster_chg:
        index_chg_num = 0

        for i in range(sample_num):
            sum_dist_iq = 0
            for q in range(k):
                dist_iq = dis_func(dataset[i], center_coor[q])
                sum_dist_iq += pow((1 / pow(dist_iq, 2)), (1 / (p - 1)))

            for j in range(k):
                dist_ij = dis_func(dataset[i], center_coor[j])
                weights[i, j] = pow((1/pow(dist_ij, 2)), (1/(p-1))) / sum_dist_iq

        for i in range(sample_num):
            max_idx = -1
            max_weight = 0
            for j in range(k):
                cur_weight = weights[i, j]
                if cur_weight > max_weight:
                    max_weight = cur_weight
                    max_idx = j
            if cluster_ass[i, 0] != max_idx:
                index_chg_num += 1

            max_dst = dis_func(dataset[i], center_coor[max_idx])
            cluster_ass[i] = [max_idx, max_dst, max_weight]

        if index_chg_num / sample_num <= stop_rate:
            cluster_chg = False

        for cent in range(k):
            numerator = np.sum([np.power(weights[:, cent], p)[m] * dataset[m] for m in range(len(dataset))], axis=0)
            denominator = np.sum(np.power(weights[:, cent], p), axis=0)
            center_coor[cent, :] = numerator / denominator

    index_each_cluster = []
    for i in range(k):
        index_each_cluster.append(np.argwhere(cluster_ass[:, 0] == i).squeeze())

    silhouette_score = silhouette_score_each(dataset, cluster_ass[:, 1], index_each_cluster, center_coor, dis_func,
                                             is_each=False)

    return index_each_cluster, center_coor, silhouette_score


def find_neighbor(one, dataset, eps, dst_func):
    idx_fine = []
    for i in range(dataset.shape[0]):
        distance = dst_func(one, dataset[i])
        if distance <= eps:
            idx_fine.append(i)
    return set(idx_fine)


def DBSCAN_cluster(dataset, eps, min_pts, dst_func):
    k = -1
    neighbor_each = []
    omega_set = []
    gama_set = set(list(np.arange(0, len(dataset), 1)))
    cluster_idx_each = list(np.ones(shape=(len(dataset), )))
    
    for i in range(len(dataset)):
        neighbor_each.append(find_neighbor(dataset[i], dataset, eps, dst_func))
        if len(neighbor_each[i]) >= min_pts:
            omega_set.append(i)
    omega_set = set(omega_set)

    while len(omega_set) > 0:
        gama_old = copy.deepcopy(gama_set)
        j = random.choice(list(omega_set))
        k = k + 1
        Q = list()
        Q.append(j)
        gama_set.remove(j)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(neighbor_each[q]) >= min_pts:
                delta = neighbor_each[q] & gama_set
                delta_list = list(delta)
                for i in range(len(delta_list)):
                    Q.append(delta_list[i])
                gama_set = gama_set - delta
        ck = gama_old - gama_set
        ck_list = list(ck)
        for i in range(len(ck)):
            cluster_idx_each[ck_list[i]] = k
        omega_set = omega_set - ck

    return cluster_idx_each


def extract_label(data, orders, reach_dst, eps):
    # 可达距离按orders排序
    reach_dst_order = reach_dst[orders]
    # 找到排序中可达距离不高于eps的index，这里的index不再是样本的index，而是orders的index
    rdo_less_pes_id = np.where(reach_dst_order <= eps)[0]
    # 检查id是否时连续的，不连续的位置即为outlier, 而且不能忘记前后的位置检查, 找断点不是很好处理，判断语句麻烦
    labels = np.full((len(data)), -1)
    cluster = 0
    previous_id = rdo_less_pes_id[0] - 1
    for now in rdo_less_pes_id:
        # 第一个点肯定满足这个判断语句, 因为previous_id就是第一个元素减一定义的
        if now - previous_id != 1:
            cluster += 1
        labels[orders[now]] = cluster
        previous_id = now
    return labels


def OPTICS_cluster(dst_matrix, data=None, dst_func=None, eps=np.inf, min_pts=10, is_precomputed=True):
    def data2matrix(data_list, dst_func):
        up_tri_matrix = np.zeros(shape=[len(data_list), len(data_list)])
        for i in range(len(data_list) - 1):
            for j in range(i + 1, len(data_list)):
                up_tri_matrix[i, j] = dst_func(data_list[i], data_list[j])
        matrix = up_tri_matrix + up_tri_matrix.T
        return matrix

    def update(seeds: dict, reach_dst, p, core_dst, N_eighbors, dst_m, is_processed):
        # core_p的核心距离
        core_dst_p = core_dst[p]
        # 遍历未分类对象
        for n in N_eighbors:
            if is_processed[n] == -1:
                # 计算由p到n的可达距离
                new_reach_dst = max(core_dst_p, dst_m[p, n])
                # 判断是否以前算过到n的可达距离
                if np.isnan(reach_dst[n]) or new_reach_dst < reach_dst[n]:
                    reach_dst[n] = new_reach_dst
                    seeds[n] = new_reach_dst
        return seeds, reach_dst

    if is_precomputed:
        dst_matrix = dst_matrix
    else:
        dst_matrix = data2matrix(data, dst_func)

    r, c = dst_matrix.shape
    # 先假装每一个都是核心对象，求解满足min_pts条件时的最远距离
    temp_core_dst = dst_matrix[np.arange(0, r), np.argsort(dst_matrix, axis=1)[:, min_pts - 1]]
    # 将满足min_pts条件时距离超过eps的点核心距离设为-1
    core_dst = np.where(temp_core_dst <= eps, temp_core_dst, -1)
    # 初始化每个点的可达距离
    reach_dst = np.full((r,), np.nan)
    # 定位core的位置，但是有的程序在这里还会再算一次core, 难道不能直接通过core_dst得到吗？
    core_point_index = np.where(core_dst != -1)[0]
    # 标识点是否被处理，未处理为-1
    is_processed = np.full((r,), -1)
    # 存放序列点
    orders = []
    # 遍历核心点
    for p_id in core_point_index:
        if is_processed[p_id] == -1:
            is_processed[p_id] = 1
            orders.append(p_id)
            # 寻找core_p的邻居中(不含自己)未分类的点，放入种子集合, np.where 返回的是行和列的array, 条件分开写
            neighbors = np.where((0 < dst_matrix[p_id, :]) & (dst_matrix[p_id, :] <= eps) & (is_processed == -1))[0]
            seeds = dict()
            seeds, reach_dst = update(seeds, reach_dst, p_id, core_dst, neighbors, dst_matrix, is_processed)
            while len(seeds) > 0:
                p_next_id = sorted(seeds.items(), key=operator.itemgetter(1))[0][0]
                del seeds[p_next_id]
                is_processed[p_next_id] = 1
                orders.append(p_next_id)
                # 判断next点还是core吗，如果是执行update, 更新seeds, 不是下一次循环，
                # 注意的是这里判断core， 如果是core, 但是is_processed = 1， 好像是不可能的，这个担心多余?
                if core_dst[p_next_id] != -1:
                    neighbors = np.where((0 < dst_matrix[p_id, :]) & (dst_matrix[p_id, :] <= eps))[0]
                    seeds, reach_dst = update(seeds, reach_dst, p_next_id, core_dst, neighbors, dst_matrix,
                                              is_processed)
    return orders, reach_dst


def sil_score(samples, labels):
    return silhouette_score(X=samples, labels=labels, metric='euclidean')


def dbscan_cluster(dataset, eps, min_samples, dst_func):

    if len(dataset.shape) == 1:
        dataset = np.expand_dims(dataset, axis=1)
    estimator = DBSCAN(eps=eps, min_samples=min_samples, metric=dst_func, n_jobs=-1)
    estimator.fit(dataset)
    label_pred = estimator.labels_
    core_index = estimator.core_sample_indices_
    core = estimator.components_

    return label_pred, core_index, core


def agglomerative_cluster(dataset, k, dst_func):
    estimator = AgglomerativeClustering(n_clusters=k, affinity=dst_func)
    estimator.fit(dataset)
    label_pred = estimator.labels_
    return label_pred, None, None

