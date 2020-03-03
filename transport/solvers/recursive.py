import numpy as np
from scipy import linalg as la

#Cutoff for cutting block matrices
from .tridiagonal import cutoff

def left_div(a, b):
    # Solve ax=b
    res, resid, rank, s = la.lstsq(a, b ,cond=-1)

    return res

def recursive_gf(mat_list_ii, mat_list_ij, mat_list_ji):

    N = len(mat_list_ii)
    mat_shapes = [mat.shape for mat in mat_list_ii]

    # Allocate empty list
    gr_list_ii = [None for _ in range(N)]
    gr_list_1i = [None for _ in range(N)]

    # Initialize retarded ginv_11 = (eS-H-Sigma)_11
    gr_list_ii[0] = left_div(mat_list_ii[0], np.eye(mat_shapes[0][0]))
    gr_list_1i[0] = gr_list_ii[0]

    # Downward recursion
    for q in range(1, N):
        gr_ii = gr_list_ii[q - 1]
        tau_ij = mat_list_ij[q - 1]
        tau_ji = mat_list_ji[q - 1]
        a_jj = tau_ji.dot(gr_ii.dot(tau_ij))
        # Diagonal
        gr_list_ii[q] = left_div(mat_list_ii[q] - a_jj,
                                 np.eye(mat_shapes[q][0]))
        # First row
        gr_list_1i[q] = gr_list_1i[q - 1].dot(tau_ij.dot(gr_list_ii[q]))

    return gr_list_1i[-1]

def get_mat_lists(z, hs_list_ii, hs_list_ij, sigma_L=None, sigma_R=None):

    mat_list_ii = []
    mat_list_ij = []
    mat_list_ji = []

    h_list_ij, s_list_ij = self.hs_list_ij
    for h_ij, s_ij in zip(h_list_ij,
                          s_list_ij):

        mat_list_ij.append(z * s_ij - h_ij)
        mat_list_ji.append(z * s_ij.T.conj() - h_ij.T.conj())

    h_list_ii, s_list_ii = self.hs_list_ii
    for h_ii, s_ii in zip(h_list_ii,
                          s_list_ii):
        mat_list_ii.append(z * s_ii - h_ii)

    if sigma_L and sigma_R:
        mat_list_ii[0]  -= sigma_L.retarded(energy)
        mat_list_ii[-1] -= sigma_R.retarded(energy)

    return mat_list_ii, mat_list_ij, mat_list_ji
