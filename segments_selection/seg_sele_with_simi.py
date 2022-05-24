import pickle
import numpy as np
from distance_function_s2s import dtw


def load_pkl(file_root):
    with open(file_root, 'rb') as f:
        data = pickle.load(f)
    return data


def class_10_sele(tar_seg, segments, dst_func=None):
    if dst_func is None:
        dst_func = dtw
    dst_s = []
    for i in range(segments.shape[0]):
        dst = dst_func(tar_seg, segments[i])
        dst_s.append(dst)
    index_sort = np.argsort(dst_s)
    return index_sort


# if __name__ == '__main__':
