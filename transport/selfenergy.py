import numpy as np
from functools import lru_cache
from .internalselfenergy import InternalSelfEnergy
from .tools import rotate_matrix, get_subspace
from scipy import linalg as la
# from gpaw.utilities.blas import gemm
# import _cppmodule as _cpp

# class LeadSelfEnergy(_cpp.LeadSelfEnergy, InternalSelfEnergy):
class LeadSelfEnergy(InternalSelfEnergy):
    conv = 1e-8 # Convergence criteria for surface Green function

    def __init__(self, hs_dii, hs_dij, hs_dim, eta=1e-4):
        # self.impl = _cpp.LeadSelfEnergy(*hs_dii, *hs_dij, *hs_dim, eta)
        # _cpp.LeadSelfEnergy.__init__(self, *hs_dii, *hs_dij, *hs_dim, eta)
        # self.h_ii, s_ii = hs_dii # onsite principal layer
        # self.h_im, s_im = hs_dim # coupling to the central region
        # self.nbf = self.h_im.shape[1] # nbf for the scattering region
        # self.eta = eta
        # self.energy = None
        # self.sigma_mm = np.empty((self.nbf, self.nbf), complex)

        self.h_ij, self.s_ij = hs_dij # coupling between principal layers
        self.bias = 0
        InternalSelfEnergy.__init__(self, hs_dii, hs_dim, eta=eta)

    # def retarded(self, energy):
    #     """Return self-energy (sigma) evaluated at specified energy."""
    #     if energy != self.energy:
    #         self.energy = energy
    #         z = energy - self.bias + self.eta * 1.j
    #         tau_im = z * self.s_im - self.h_im
    #         a_im = np.linalg.solve(self.get_sgfinv(energy), tau_im)
    #         tau_mi = z * self.s_im.T.conj() - self.h_im.T.conj()
    #         self.sigma_mm[:] = np.dot(tau_mi, a_im)
    #
    #     return self.sigma_mm

    def set_bias(self, bias):
        self.bias = bias

    def get_lambda(self, energy):
        """Return the lambda (aka Gamma) defined by i(S-S^d).

        Here S is the retarded selfenergy, and d denotes the hermitian
        conjugate.
        """
        sigma_mm = self.retarded(energy)
        return 1.j * (sigma_mm - sigma_mm.T.conj())

    # # @lru_cache(maxsize=None)
    def get_Ginv(self, energy):
        # """The inverse of the retarded surface Green function"""
        z = energy - self.bias + self.eta * 1.j

        v_00 = z * self.s_ii.T.conj() - self.h_ii.T.conj()
        v_11 = v_00.copy()
        v_10 = z * self.s_ij - self.h_ij
        v_01 = z * self.s_ij.T.conj() - self.h_ij.T.conj()

        delta = self.conv + 1
        while delta > self.conv:
            lu, piv = la.lu_factor(v_11)
            a = la.lu_solve((lu, piv), v_01)
            b = la.lu_solve((lu, piv), v_10)
            v_01_dot_b = np.dot(v_01, b)
            v_00 -= v_01_dot_b
            v_11 -= np.dot(v_10, a)
            v_11 -= v_01_dot_b
            v_01 = -np.dot(v_01, a)
            v_10 = -np.dot(v_10, b)
            delta = abs(v_01).max()

        return v_00

    def apply_rotation(self, c_mm):
        # self.get_sgfinv.cache_clear()
        self.h_ii = self.h_ii.copy() # needed because self[0].h_ii is self[1].h_ii
        self.s_ii = self.s_ii.copy() # as self[0].h_ii -> H1[:nprinc,:nprinc]
        InternalSelfEnergy.apply_rotation(self, c_mm)
        self.h_ij = rotate_matrix(self.h_ij, c_mm) #np.dot(c_mm.T.conj(), self.h_ij)
        self.s_ij = rotate_matrix(self.s_ij, c_mm) #np.dot(c_mm.T.conj(), self.s_ij)

    def cutcoupling_bfs(self, bfs, apply=False):
        # self.get_sgfinv.cache_clear()
        h_pp, s_pp = InternalSelfEnergy.cutcoupling_bfs(self, bfs, apply)
        if apply:
            for m in bfs:
                self.h_ij[m, :] = 0.0
                self.s_ij[m, :] = 0.0
        return h_pp, s_pp

    def take_bfs(self, bfs, apply=False):
        # self.get_sgfinv.cache_clear()
        h_pp, s_pp, c_mm = InternalSelfEnergy.take_bfs(self, bfs, apply)
        if apply:
            self.h_ij = rotate_matrix(self.h_ij, c_mm) #np.dot(c_mm.T.conj(), self.h_ij)
            self.s_ij = rotate_matrix(self.s_ij, c_mm) #np.dot(c_mm.T.conj(), self.s_ij)
        return h_pp, s_pp


class BoxProbe:
    """Box shaped Buttinger probe.

    Kramers-kroning: real = H(imag); imag = -H(real)
    """
    def __init__(self, eta, a, b, energies, S, T=0.3):
        from Transport.Hilbert import hilbert
        se = np.empty(len(energies), complex)
        se.imag = .5 * (np.tanh(.5 * (energies - a) / T) -
                        np.tanh(.5 * (energies - b) / T))
        se.real = hilbert(se.imag)
        se.imag -= 1
        self.selfenergy_e = eta * se
        self.energies = energies
        self.S = S

    def retarded(self, energy):
        return self.selfenergy_e[self.energies.searchsorted(energy)] * self.S
