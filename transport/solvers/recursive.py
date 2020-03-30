import numpy as np
from scipy import linalg as la

#Cutoff for cutting block matrices
from .tridiagonal import cutoff
from transport.tools import dagger

def left_div(a, b):
    # Solve ax=b
    res, resid, rank, s = la.lstsq(a, b ,cond=-1)

    return res

def recursive_gf(mat_list_ii, mat_list_ij, mat_list_ji, s_in=None, dos=False):

    N = len(mat_list_ii)
    mat_shapes = [mat.shape[0] for mat in mat_list_ii]

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

    # Only transport
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

    # Return retarded
    if s_in is None:
        # DOS
        return Gr_qii, Gr_qij, Gr_qji
        
    # Electron correlation function
    if isinstance(s_in, list):

        gnL_qii = [None for _ in range(N)]
        # Initalize
        gnL_qii = grL_qii[0] @ s_in[0] @ dagger(grL_qii[0])

        # Left connected recursion
        for q in range(1, N):
            sl = m_qji[q - 1] @ gnL_qii[q - 1] @ m_qij[q - 1]
            gnL_qii[q] = grL_qii[q] @ (s_in[q] + sl) @ dagger(grL_N[q])

        Gn_qii = [None for _ in range(N)]
        Gn_qij = [None for _ in range(N-1)]
        Gn_qji = [None for _ in range(N-1)]
        # Initialize
        Gn_qii[-1] = gnL_qii[-1]

        # Full recursion
        for q in range(N-2, -1, -1):
            a = Gn_qii[q + 1] @ dagger(m_qji[q]) @ dagger(grL_qii[q])
            Gn_qij[q] = - Gr_qii[q + 1] @ m_qji[q] @ gnL_qii[q] - a
            Gn_qii[q] = gnL_qii[q] + \
                        grL_qii[q] @ m_qij[q] @  a - \
                        gnL_qii[q] @ dagger(m_qij[q]) @ dagger(Gr_qji[q]) - \
                        Gr_qij[q] @ m_qji[q] @ gnL_qii[q]
            Gn_qji = dagger(Gn_qij[q])

    # Return electron correlation
    return Gn_qii, Gn_qij, Gn_qji

    # # Right connected green's function
    # hgh_qii = [None for _ in range(N-1)]
    # # Initalize
    # grR_inv = m_qii[-1] # ([eS-H]_NN-Sigma_R)^-1
    #
    # # Right connected recursion
    # for q in range(N-2, -1, -1):
    #     # Left
    #     hgh_qii[q] = m_qij[q] @ la.solve(grR_inv, m_qji[q])
    #     grR_inv = m_qii[q] - hgh_qii[q]
    #
    # # Full green's function
    # Gr_qii = [None for _ in range(N)]
    # Gr_qii[-1] = grL_qii[-1]       #Actual gNN
    # Gr_qii[0]  = la.inv(grR_inv)   #Actual g11
    # for q in range(1, N-1):
    #     Gr_qii[q] = grL_qii[q] @ la.inv(
    #     np.eye(mat_shapes[q]) - grL_qii[q] @ hgh_qii[q])
    #
    # return Gr_qii, None, None
    #

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
