import numpy as np


def cmds(dst_mat, k):
    m, n = dst_mat.shape
    delta = dst_mat * dst_mat
    U = np.ones(shape=(n, n))
    E = np.eye(n)
    S = -((E - 1/n * U) @ delta @ (E - 1/n * U)) / 2
    eigenvalues, V = np.linalg.eigh(S)
    # 特征向量是按照特征值升序，按列排列的， 反转为降序
    V = np.flip(V, axis=1)
    eigenvalues = np.flip(eigenvalues)

    sigma = np.zeros((k, k))
    for i in range(0, k):
        sigma[i, i] = np.sqrt(eigenvalues[i])

    return V[:, :k] @ sigma
