import numpy as np
import math
from sklearn.manifold import MDS

import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.decomposition import PCA
import time

def similarity_rectilinear_motion(segment):
    """
    Args:
    :param segment: shape=[time_steps, input_size]
    :return:
    """
    start = np.array([0, 0, 0])
    end = np.sum(segment, axis=0)
    distance = []
    if (start == end).all():
        for i in range(segment.shape[0]):
            distance.append(np.linalg.norm(segment[i, :]))
    else:
        vec_start_end = end - start
        for i in range(segment.shape[0]):
            vec_point_start = start - segment[i, :]
            distance.append(np.linalg.norm(np.cross(vec_point_start, vec_start_end), ord=2) \
                            / np.linalg.norm(vec_start_end, ord=2))

    return np.array([np.max(distance), np.mean(distance)])


def similarity_velocity_var_directory_sum(segment):
    '''
    Args:
    :param serie: segment.shape[time_len, input_size]
    :return:
    '''
    speed = np.zeros(shape=(segment.shape[0],))
    for i in range(segment.shape[0]):
        speed[i] = np.linalg.norm(segment[i], ord=2)
    speed_var = np.var(speed)

    angle_change_sum = 0

    for j in range(0, segment.shape[0]):
        if j != 0:
            angle_1 = math.atan(segment[j - 1, 1] / (segment[j - 1, 0] + 0.001))
            angle_2 = math.atan(segment[j, 1] / (segment[j, 0] + 0.001))
            angle_c = np.abs(angle_2 - angle_1) * 180 / math.pi
            angle_change_sum += angle_c

    dis = similarity_rectilinear_motion(segment)
    return np.array([speed_var, angle_change_sum, dis])


def series_mean_var(segment):
    """
    Args:
    :param segment: shape = [time_len, input_size]
    :return:
    """
    return np.concatenate([np.mean(segment, axis=0), np.var(segment, axis=0)], axis=0)


def acceleration_vector_sum(segment):
    """
    Args:
    :param segment: shape = [time_len, input_size]
    :return:
    """
    acc_series = []
    for i in range(1, len(segment)):
        acc_series.append(segment[i] - segment[i - 1])
    acc_series = np.array(acc_series)  # shape=[time_len-1, input_size]

    return np.sum(acc_series, axis=0)


def get_layer_out(model: tf.keras.models, inputs, layer_name):
    functor = K.function(inputs=model.input, outputs=model.get_layer(name=layer_name).output)
    return functor([inputs, 1])


def mds(dataset, n_dims):
    def cal_pairwise_dist(x):
        """计算pairwise 距离, x是matrix
        (a-b)^2 = a^2 + b^2 - 2*a*b
        """
        sum_x = np.sum(np.square(x), 1)
        dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
        # 返回任意两个点之间距离的平方
        return dist

    n, d = dataset.shape
    dist = cal_pairwise_dist(dataset)
    T1 = np.ones((n, n)) * np.sum(dist) / n ** 2
    T2 = np.sum(dist, axis=1, keepdims=True) / n
    T3 = np.sum(dist, axis=0, keepdims=True) / n

    B = -(T1 - T2 - T3 + dist) / 2

    eig_val, eig_vector = np.linalg.eig(B)
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]
    # print(picked_eig_vector.shape, picked_eig_val.shape)

    return np.real(picked_eig_vector * picked_eig_val ** 0.5)


def mds_reduction(encode_model, dataset, layer_name, mds_dim):
    latent_rep = get_layer_out(model=encode_model, inputs=dataset, layer_name=layer_name)
    latent_mds = mds(latent_rep, mds_dim)
    return latent_mds


def pca_weighted_1fea(time_series, k=2):
    pca = PCA(n_components=k)
    pca.fit(time_series)
    ekj = pca.components_
    wk = pca.explained_variance_ratio_
    wj = np.sum(np.multiply(np.expand_dims(wk, axis=1), ekj), axis=0)
    return np.matmul(time_series, np.expand_dims(wj, axis=1)).squeeze()


def pca_to_k(time_series, k=1):

    pca = PCA(n_components=k)
    pca.fit(time_series)
    return pca.fit_transform(time_series).squeeze()

