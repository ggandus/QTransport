import numpy as np
# w(n) exp(-0.5 * (n / sig)^2)
from scipy.signal import gaussian
from scipy.linalg import toeplitz

from ase.geometry import wrap_positions
#
# def pbc2pbc(pbc):
#     newpbc = np.empty(3, int)
#     newpbc[:] = pbc
#     return newpbc

# def wrap_indices(I_cn, N_c, pbc=True):
#     pbc = pbc2pbc(pbc)
#     shift = np.zeros(3) - 0.5
#     # Do not shift non periodic
#     shift[np.logical_not(pbc)] = 0.
#     # Scaled indices
#     N_c = N_c[:,None]
#     sind_ck = I_cn / N_c - shift[:,None]
#     for i, periodic in enumerate(pbc):
#         if periodic:
#             sind_ck[i, :] %= 1.0
#             sind_ck[i, :] += shift[i]
#     return sind_ck * N_c


def get_weights(beta, omega=2):
    sigma = 1 / (2 * beta)
    alpha = beta**3 / np.pi**(3/2)
    alpha * toeplitz(np.exp(-(beta*omega+1)))

def get_sample_points(pos_nv, N_c, h_cv, pbc=True, omega=2):
    N_c = np.diag(N_c)
    # Position in grid
    pos_nc = pos_nv.dot(np.linalg.inv(h_cv))
    # Ceil bacuase neighbors (-omega,..,0,..,omega-1)
    ind_ng = np.ceil(pos_nc)
    # Fractional up-grid distances
    dist_nc = ind_ng - pos_nc
    # DistanceArray
    dist_nck = np.zeros(ind_ng.shape+(2*omega,))
    neighbors_ck = np.tile(np.arange(omega), (3,1))
    dist_nck[...,omega:] = dist_nc[...,None] + neighbors_ck
    dist_nck[...,:omega] = (1-dist_nc)[...,None] + neighbors_ck
    dist_nkv = np.swapaxes(dist_nck, 1, 2).dot(h_cv)

    # NeighborArray
    neighbors_ck = np.tile(np.arange(-omega,omega), (3,1))
    ind_ngk = ind_ng[...,None] + neighbors_ck
    for n, i_c in enumerate(ind_ng):
        ind_ngk[n, :] = wrap_positions((i_c[:,None] + neighbors_ck).T,
                                        N_c,
                                        pbc).T

    return ind_ngk, np.swapaxes(dist_nkv, 1, 2)
    #
    # # NieghborList
    # ind_nck = np.zeros(ind_nc.shape + (2*omega,), dtype=int)
    # # Centered neighborlist to iterate
    # nn_ck = np.array([list(range(-omega,omega))*3]).reshape(3,-1)
    # for n, i_c in enumerate(ind_nc):
    #     ind_nck[n, :] = wrap_positions((i_c[:,None] + nn_ck).T,
    #                                     N_c,
    #                                     pbc).T
    # return ind_nck

#class CCA:
    def __init__(self, atoms, rho, pbc):
        self.atoms = atoms
        self.rho = rho
        self.pbc = pbc

    def interpolate(self, N_c, h_cv):
        pass


    def interpolate(self, x, nnear=6, eps=0, p=1, weights=None):
        '''For the choice of value for p, one can consider the
        degree of smoothing desired in the interpolation. Greater
        values of p assign greater influence to values closest
        to the interpolated point (smoother interpolation).'''
        x = np.asarray(x)
        xdim = x.ndim
        if xdim == 1:
            x = np.array([x])
        wsum = np.zeros(nnear)

        distances, indices = self.tree.query(x, k=nnear, eps=eps)
        interpol = np.zeros((len(distances),) + np.shape(self.u[0]))
        jinterpol = 0
        for dist, ix in zip(distances, indices):
            if nnear == 1:
                wz = self.u[ix]
            elif dist[0] < 1e-10:
                wz = self.u[ix[0]]
            else:  # weight u s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot(w, self.u[ix])
                # if self.stat:
                #     self.wn += 1
                #     wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if xdim > 1  else interpol[0]
