import numpy as np
import math
from similarity_of_NN.similarity_of_rpt_of_layers import compute_cca_dst


def dst_euc(p1, p2):
    return np.linalg.norm((p1 - p2), ord=2)


def dst_cos(p1, p2):
    numerator = np.cross(p1, p2)
    denominator = np.linalg.norm(p1, ord=2) * np.linalg.norm(p2, ord=2)
    return numerator / denominator


def cos_one_fea(sg):
    cos_series = []
    for i in range(1, len(sg) - 1):
        cos_series.append(dst_cos(p1=sg[i] - sg[i - 1],
                                  p2=sg[i + 1] - sg[i]))
    return cos_series


def cos_one_fea_euc(sg1, sg2):
    cos_series_1 = cos_one_fea(sg1)
    cos_series_2 = cos_one_fea(sg2)
    dis_series = []
    for i in range(0, len(cos_series_1), 1):
        dis_series.append(np.linalg.norm(([cos_series_1[i] - cos_series_2[i]]), ord=2))
    return np.sum(dis_series)


def euc_s2s(sg1, sg2):
    dis_series = []
    for i in range(0, len(sg1), 1):
        dis_series.append(dst_euc(sg1[i], sg2[i]))
    return np.sum(dis_series)


def cos_s2s(sg1, sg2):
    dis_series = []
    for i in range(len(sg1)):
        dis_series.append(dst_cos(sg1[i], sg2[i]))
    return np.sum(dis_series)


def euc_cos(sg1, sg2):
    dis_euc = euc_s2s(sg1, sg2)
    dis_cos = cos_s2s(sg1, sg2)
    c1 = dis_euc / (dis_euc + dis_cos)
    c2 = dis_cos / (dis_euc + dis_cos)
    return c1 * dis_euc + c2 * dis_cos


def dst_hdf(sg1, sg2):
    array_dst = np.zeros(shape=(len(sg1), len(sg2)))
    for i in range(array_dst.shape[0]):
        for j in range(array_dst.shape[1]):
            array_dst[i, j] = dst_euc(sg1[i], sg2[j])
    hdf_1_2 = np.max(np.min(array_dst, axis=1))
    hdf_2_1 = np.max(np.min(array_dst, axis=0))
    return np.max([hdf_1_2, hdf_2_1])


def dtw(sg1, sg2, dst=None):
    S = len(sg1)
    T = len(sg2)
    index = []
    D_mm = np.zeros((S, T))
    if dst is None:
        dst = dst_euc
    D_mm[0, 0] = dst(sg1[0], sg2[0])
    for i in range(1, S):
        D_mm[i, 0] = D_mm[i - 1, 0] + dst(sg1[i], sg2[0])

    for j in range(1, T):
        D_mm[0, j] = D_mm[0, j - 1] + dst(sg1[0], sg2[j])

    for i in range(1, S):
        for j in range(1, T):
            target = D_mm[[i - 1, i, i - 1], [j - 1, j - 1, j]]
            tar_idx = [(i - 1, j - 1), (i, j - 1), (i - 1, j)]
            min_idx = np.argmin(target)
            index.append(tar_idx[int(min_idx)])
            D_mm[i, j] = np.min(target) + dst(sg1[i], sg2[j])
    return D_mm[S - 1, T - 1]


def dst_mahalanobis(sg1, sg2):
    sigma_1 = np.linalg.pinv(np.cov(sg1.T))
    x_y = sg1 - sg2
    return np.sqrt(np.dot(x_y, np.dot(sigma_1, x_y.T)))


def pearson(series_1, series_2):
    return np.corrcoef(series_1, series_2)[0, 1]


def dst_cca(sg1, sg2):
    cca_dst = compute_cca_dst(sg1, sg2, epsilon=1.0e-06, is_remove_small_var=False,
                              verbose=False, dst_name='pw', is_sv=False)
    return cca_dst


def over_lap_volumn():
    a = 1
