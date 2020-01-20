import itertools
import numpy as np
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset

def reduce_vecs(R):
    '''This function reduces symmetries along the last
    two directions'''
    R_rc = []
    for r in R:
        try:
            if r[np.nonzero(r)[0][0]] > 0:
                R_rc.append(r)
            else:
                continue
        # zero case
        except IndexError:
            R_rc.append(r)
    return R_rc


def get_realspace_R(kpts_grid):
    '''This function returns the real space hopping vectors,
    in crystal coordinates, for the input monkhorst pack size.'''
    # R_rc = list(itertools.product(range(-kpts_grid[0],kpts_grid[0]+1),
    #                               range(-kpts_grid[1],kpts_grid[1]+1),
    #                               range(-kpts_grid[2],kpts_grid[2]+1)))
    #
    ndim = len(kpts_grid) # number of dimensions
    R_rc = list(itertools.product(*[range(-kpts_grid[i]+1,kpts_grid[i])
                                   for i in range(len(kpts_grid))]))
    return reduce_vecs(R_rc)

def reduce_hop():
    c_k = np.exp(2.j * np.pi * np.dot(ibzk_c, R_c)) * weight_k[k]
    h_skmm[:, r] += c_k * h_skmm[:, k]
    s_kmm[r] += c_k * s_kmm[k]

def get_realspace_hs(h_skmm, s_kmm, bzk_kc, weight_k):
    kpts_grid, kpts_shift = get_monkhorst_pack_size_and_offset(bzk_kc)
    R_rc = get_realspace_R(kpts_grid // 2)
    ibzk_kc = reduce_vecs(bzk_kc.tolist())

    h_srii = np.zeros_like(h_skmm)
    s_rii  = np.zeros_like(s_kmm)
    for r, R_c in enumerate(R_rc):
        for k, ibzk_c in enumerate(ibzk_kc):
            c_k = np.exp(2.j * np.pi * np.dot(ibzk_c, R_c)) * weight_k[k]
            h_srii[:, r] += c_k * h_skmm[:, k]
            s_rii[r] += c_k * s_kmm[k]

    return h_srii, s_rii, R_rc
