import numpy as np
from scipy import linalg as la
from .coupledhamiltonian import CoupledHamiltonian

class InternalSelfEnergy(CoupledHamiltonian):

    def __init__(self, hs_dii, hs_dim, selfenergies=[], eta=1e-5):
        # LeadSelfEnergy.__init__(self, hs_dii, (None, None), hs_dim, eta)
        self.h_ii, self.s_ii = hs_dii # onsite principal layer
        self.h_im, self.s_im = hs_dim # coupling to the central region
        self.nbf_i = self.h_im.shape[0] # nbf_m for the internal self-energy
        self.nbf_m = self.h_im.shape[1] # nbf_m for the scattering region
        self.eta = eta
        self.energy = None
        self.bias = 0
        self.sigma_mm = np.empty((self.nbf_m, self.nbf_m), complex)
        self.Ginv = np.empty((self.nbf_i, self.nbf_i), complex)
        self.selfenergies = selfenergies

        CoupledHamiltonian.__init__(self, self.h_ii, self.s_ii, self.selfenergies)

    def retarded(self, energy):
        """Return self-energy (sigma) evaluated at specified energy."""
        if energy != self.energy:
            self.energy = energy
            z = energy - self.bias + self.eta * 1.j
            tau_im = z * self.s_im - self.h_im
            a_im = np.linalg.solve(self.get_Ginv(energy), tau_im)
            tau_mi = z * self.s_im.T.conj() - self.h_im.T.conj()
            self.sigma_mm[:] = np.dot(tau_mi, a_im)

        return self.sigma_mm

    def get_lambda(self, energy):
        """Return the lambda (aka Gamma) defined by i(S-S^d).

        Here S is the retarded selfenergy, and d denotes the hermitian
        conjugate.
        """
        sigma_mm = self.retarded(energy)
        return 1.j * (sigma_mm - sigma_mm.T.conj())

    def get_Ginv(self, energy, inverse=True):

        z = energy - self.bias + self.eta * 1.j

        self.Ginv[:] = z
        self.Ginv *= self.S
        self.Ginv -= self.H

        for selfenergy in self.selfenergies:
            self.Ginv -= selfenergy.retarded(energy)

        # v_00 = z * self.s_ii - self.h_ii
        # for selfenergy in self.selfenergies:
        #     self.Ginv -= selfenergy.retarded(energy)

        if inverse:
            return self.Ginv
        else:
            return la.inv(self.Ginv)

    def get_matsubara(self, beta, n):
        w_n = np.pi/beta * (2*n + 1)
        energy = 1.j * w_n - self.eta * 1.j
        return self.retarded(energy)

    ####### CONVENIENT ALISES ########

    @property
    def H(self):
        return self.h_ii
    @property
    def S(self):
        return self.s_ii
    @H.setter
    def H(self, H):
        self.h_ii = H
    @S.setter
    def S(self, H):
        self.s_ii = H

    ######## MODIFIERS ################

    def apply_rotation(self, c_mm):
        CoupledHamiltonian.apply_rotation(self, c_mm)
        self.h_im[:] = np.dot(c_mm.T.conj(), self.h_im)
        self.s_im[:] = np.dot(c_mm.T.conj(), self.s_im)

    def cutcoupling_bfs(self, bfs, apply=False):
        h_pp, s_pp = CoupledHamiltonian.cutcoupling_bfs(self, bfs, apply)
        if apply:
            for m in bfs:
                self.h_im[m, :] = 0.0
                self.s_im[m, :] = 0.0
        return h_pp, s_pp

    def take_bfs(self, bfs, apply=False):
        h_pp, s_pp, c_mm = CoupledHamiltonian.take_bfs(self, bfs, apply)
        if apply:
            self.h_im = np.dot(c_mm.T.conj(), self.h_im)
            self.s_im = np.dot(c_mm.T.conj(), self.s_im)
            self.Ginv = np.empty(self.H.shape, complex)
        return h_pp, s_pp, c_mm

    # def orthogonalize(self, ):
    #     s_ii = self.s_ii
    #     s_mi = self.s_im.T
    #     u_im = - self.s_im.T
    #     u_mi = u_im.T
    #     st_ii = s_ii + u_im.dot(s_mi) + (u_im.dot(s_mm) + s_im).dot(u_mi)
    #     ht_ii = s_ii + u_im.dot(s_mi) + (u_im.dot(s_mm) + s_im).dot(u_mi)
    #
    #     U_DD = np.eye(len(self.s_ii))
    #     U_DR = -s_im.dot(la.inv(s_mm))
    #     O = np.zeros_like(s_im.T)
    #     I = np.eye(len(s_mm))
    #     U = np.block([[U_DD,U_DR],[O,I]])
    #     c_mm = subdiagonalize(h_mm, s_mm, bfs)
    #     if apply:
    #       self.apply_rotation(c_mm)
    #       return
