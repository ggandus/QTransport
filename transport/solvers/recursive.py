import numpy as np
from scipy import linalg as la

#Cutoff for cutting block matrices
from .tridiagonal import cutoff

def left_div(a, b):
    # Solve ax=b
    res, resid, rank, s = la.lstsq(a, b ,cond=-1)

    return res

def recursive_gf(mat_list_ii, mat_list_ij, mat_list_ji, dos=False):

    N = len(mat_list_ii)
    mat_shapes = [mat.shape for mat in mat_list_ii]

    # np.matrix alias
    m_qii = mat_list_ii
    m_qij = mat_list_ij
    m_qji = mat_list_ji

    # Left connected green's function
    grL_qii = [None for _ in range(N)]
    # Initalize
    grL_qii[0] = la.inv(m_qii[0]) # ([eS-H]_11-Sigma_L)^-1
    # First row green's function
    gr_1i = grL_qii[0].copy()

    # Left connected recursion
    for q in range(1, N):
        # Left
        grL_qii[q] = la.inv(m_qii[q] - m_qji[q - 1] @ grL_qii[q - 1] @ m_qij[q - 1])
        # 1st row
        gr_1i = gr_1i @ m_qij[q - 1] @ grL_qii[q]

    if not dos:
        # Return g1N
        return gr_1i

    # Full green's function
    Gr_qii = [None for _ in range(N)]
    Gr_qji = [None for _ in range(N-1)]
    Gr_qij = [None for _ in range(N-1)]
    # Initialize
    Gr_qii[-1] = grL_qii[-1] # G_NN = gL_NN = ([eS-H]_NN - [eS-H]_NN-1 * grL_N-1N-1 * [eS-H]_N-1N - Sigma_R)^-1

    # Full recursion
    for q in range(N-2, -1, -1):
        Gr_qji[q] = - Gr_qii[q + 1] @ m_qji[q] @ grL_qii[q]
        Gr_qij[q] = - grL_qii[q] @ m_qij[q] @ Gr_qii[q + 1]
        Gr_qii[q] = grL_qii[q] - grL_qii[q] @ m_qij[q] @ Gr_qji[q]

    # DOS
    return Gr_qii, Gr_qij, Gr_qji


def get_mat_lists(z, hs_list_ii, hs_list_ij, sigma_L=None, sigma_R=None):

    mat_list_ii = []
    mat_list_ij = []
    mat_list_ji = []

    h_list_ij, s_list_ij = hs_list_ij
    for h_ij, s_ij in zip(h_list_ij,
                          s_list_ij):

        mat_list_ij.append(z * s_ij - h_ij)
        mat_list_ji.append(z * s_ij.T.conj() - h_ij.T.conj())

    h_list_ii, s_list_ii = hs_list_ii
    for h_ii, s_ii in zip(h_list_ii,
                          s_list_ii):
        mat_list_ii.append(z * s_ii - h_ii)

    if sigma_L is not None:
        mat_list_ii[0]  -= sigma_L
    if sigma_R is not None:
        mat_list_ii[-1] -= sigma_R

    return mat_list_ii, mat_list_ij, mat_list_ji
