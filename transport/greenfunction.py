import numpy as np
from numpy import linalg
# from functools import lru_cache
from .coupledhamiltonian import CoupledHamiltonian
from .tools import dagger

#Recursive GreenFunction helpers
from .solvers.tridiagonal   import tridiagonalize, cutoff
from .solvers.recursive import get_mat_lists, recursive_gf, multiply


class GreenFunction(CoupledHamiltonian):
    """Equilibrium retarded Green function."""

    def __init__(self, H, S=None, selfenergies=[], **kwargs):
        self.H = H
        self.S = S
        self.selfenergies = selfenergies
        # self.eta = eta
        self.energy = None
        self.Ginv = np.empty(H.shape, complex)

        self.parameters = {'eta': 1e-5,
                           'align_bf': None}
        self.initialized = False
        self.set(**kwargs)

    def set(self, **kwargs):
        for key in kwargs:
            if key in ['eta','align_bf']:
                self.initialized = False
                break
            elif key not in self.parameters:
                raise KeyError('%r not a vaild keyword' % key)

        self.parameters.update(kwargs)

    def initialize(self):#, calc, pl1, pl2=None, cutoff=cutoff, align_bf=None):
        '''
            calc: scattering calculator
            pl1: # of bfs left principal layer
            pl2: # of bfs right principal layer (default to pl1)
            cutoff: cutoff for block neighbor interaction
        '''

        if self.initialized:
            return

        p = self.parameters
        align_bf = p['align_bf']

        #Set eta
        self.eta = p['eta']

        #Align first basis funciton for Left lead
        if align_bf is not None:
            self.align_bf(align_bf)

        self.initialized = True


    # @lru_cache(maxsize=None)
    def retarded(self, energy, inverse=False):
        """Get retarded Green function at specified energy.

        If 'inverse' is True, the inverse Green function is returned (faster).
        """
        if energy != self.energy:
            self.energy = energy
            z = energy + self.eta * 1.j

            if self.S is None:
                self.Ginv[:] = 0.0
                self.Ginv.flat[:: len(self.S) + 1] = z
            else:
                self.Ginv[:] = z
                self.Ginv *= self.S
            self.Ginv -= self.H

            for selfenergy in self.selfenergies:
                self.Ginv -= selfenergy.retarded(energy)

        if inverse:
            return self.Ginv
        else:
            return np.linalg.inv(self.Ginv)

    def calculate(self, energy, sigma):
        """XXX is this really needed"""
        ginv = energy * self.S - self.H - sigma
        return np.linalg.inv(ginv)

    def apply_retarded(self, energy, X):
        """Apply retarded Green function to X.

        Returns the matrix product G^r(e) . X
        """
        return np.linalg.solve(self.retarded(energy, inverse=True), X)

    def apply_overlap(self, energy, trace=False):
        """Apply retarded Green function to S."""
        GS = self.apply_retarded(energy, self.S)
        if trace:
            return np.trace(GS)
        return GS

    def dos(self, energy):
        """Total density of states -1/pi Im(Tr(GS))"""
        if self.S is None:
            return -self.retarded(energy).imag.trace() / np.pi
        else:
            GS = self.apply_retarded(energy, self.S)
            return -GS.imag.trace() / np.pi

    def pdos(self, energy):
        """Projected density of states -1/pi Im(SGS/S)"""
        if self.S is None:
            return -self.retarded(energy).imag.diagonal() / np.pi
        else:
            S = self.S
            SGS = np.dot(S, self.apply_retarded(energy, S))
            return -(SGS.diagonal() / S.diagonal()).imag / np.pi

    def take_bfs(self, bfs, apply):
        h_pp, s_pp, c_mm = CoupledHamiltonian.take_bfs(self, bfs, apply)
        if apply:
            self.Ginv = np.empty(self.H.shape, complex)
        return h_pp, s_pp, c_mm


    def get_transmission(self, energies, T_e):

        if T_e is None:
            T_e = np.zeros(len(energies))

        for e, energy in enumerate(energies):
            Ginv_mm = self.retarded(energy, inverse=True)
            lambda1_mm = self.selfenergies[0].get_lambda(energy)
            lambda2_mm = self.selfenergies[1].get_lambda(energy)
            a_mm = linalg.solve(Ginv_mm, lambda1_mm)
            b_mm = linalg.solve(dagger(Ginv_mm), lambda2_mm)
            T_e[e] = np.trace(np.dot(a_mm, b_mm)).real

        return T_e


class RecursiveGF(CoupledHamiltonian):

    def __init__(self, H, S=None, selfenergies=[], **kwargs):

        super().__init__(H, S, selfenergies)

        self.energy = None
        self.g1N = None

        self.parameters = {'eta': 1e-5,
                           'calc': None,
                           'pl1': None,
                           'pl2': None,
                           'cutoff': cutoff,
                           'align_bf': None}

        self.initialized = False
        self.set(**kwargs)

    def set(self, **kwargs):
        for key in kwargs:
            if key in ['eta','calc', 'pl1', 'pl2',
                       'cutoff', 'align_bf']:
                self.initialized = False
                break
            elif key not in self.parameters:
                raise KeyError('%r not a vaild keyword' % key)

        self.parameters.update(kwargs)

    def initialize(self):#, calc, pl1, pl2=None, cutoff=cutoff, align_bf=None):
        '''
            calc: scattering calculator
            pl1: # of bfs left principal layer
            pl2: # of bfs right principal layer (default to pl1)
            cutoff: cutoff for block neighbor interaction
        '''
        if self.initialized:
            return

        p = self.parameters
        align_bf = p['align_bf']
        calc = p['calc']
        pl1 = p['pl1']
        pl2 = p['pl2']
        cutoff = p['cutoff']

        if pl1 is None:
            pl1 = self.selfenergies[0].natoms
        if pl2 is None:
            pl2 = pl1

        #Set eta
        self.eta = p['eta']

        #Align first basis funciton for Left lead
        if align_bf is not None:
            self.align_bf(align_bf)

        #Construct block tridiagonal lists
        self.hs_list_ii, self.hs_list_ij = tridiagonalize(
                                           calc, self.H, self.S,
                                           pl1, pl2, cutoff)

        self.hs_list_ji = [[h.T for h in self.hs_list_ij[0]],
                           [s.T for s in self.hs_list_ij[1]]]

        #Note h_im[:,:pl1]=h_ij for Left and h_im[:,-pl2:]=h_ij for Right
        for selfenergy in self.selfenergies:
            selfenergy.h_im = selfenergy.h_ij
            selfenergy.s_im = selfenergy.s_ij
            selfenergy.sigma_mm = np.empty((selfenergy.nbf_i, selfenergy.nbf_i),
                                            dtype=selfenergy.sigma_mm.dtype)

        self.initialized = True


    def _get_mat_lists(self, energy):
        z = energy + self.eta * 1.j
        if self.selfenergies:
            sigma_L = self.selfenergies[0].retarded(energy)
            sigma_R = self.selfenergies[1].retarded(energy)
        else:
            sigma_L = None
            sigma_R = None
        # mat_list_ii, mat_list_ij, mat_list_ji
        return get_mat_lists(z, self.hs_list_ii, self.hs_list_ij,
                             self.hs_list_ji, sigma_L, sigma_R)


    def get_g1N(self, energy):
        #
        if energy != self.energy:
            self.energy = energy

            mat_lists = self._get_mat_lists(energy)

            self.g1N = recursive_gf(*mat_lists)#_ii, mat_list_ij, mat_list_ji)

        return self.g1N

    def get_transmission(self, energies, T_e=None):

        if T_e is None:
            T_e = np.zeros(len(energies))

        for e, energy in enumerate(energies):

            gr_1N = self.get_g1N(energy)
            ga_1N = np.conj(gr_1N.T)
            lambda1_11 = self.selfenergies[0].get_lambda(energy)
            lambda2_NN = self.selfenergies[1].get_lambda(energy)
            T_e[e] = np.trace(lambda1_11.dot(gr_1N).dot(lambda2_NN).dot(ga_1N)).real

        return T_e

    def apply_retarded(self, energy, X_qii, X_qij, X_qji, X_1N=None):
        """Apply retarded Green function to X in tridiagonal form."""

        N = len(self.hs_list_ii[0])
        # X_qji = [x.T.conj() for x in X_qij]

        gr_1N, G_qii, G_qij, G_qji = recursive_gf(*self._get_mat_lists(energy),
                                                   dos=True)


        GX_qii = multiply(G_qii, G_qij, G_qji, X_qii, X_qij, X_qji)

        # Periodic boundary conditions
        if X_1N is not None:
            # GX[0]  += gr_1N * X_N1
            GX_qii[0][:] += gr_1N @ X_1N.T
            # GX[-1] += gr_N1 * X_1N
            GX_qii[-1][:] += gr_1N.T @ X_1N

        return GX_qii

    def apply_overlap(self, energy, trace=False, diag=False):
        S_qii = self.hs_list_ii[1]
        S_qij = self.hs_list_ij[1]
        S_qji = self.hs_list_ji[1]
        # Open boundary conditions
        if self.selfenergies:
            S_1N = None
        # Periodic boundary conditions
        else:
            # The Green function changes as well.
            raise NotImplementedError('Periodic boundary conditions not implemented')
            # S_1N = self.S_1N
        GS_qii = self.apply_retarded(energy, S_qii, S_qij, S_qji, S_1N)
        if trace:
            return sum(GS.trace() for GS in GS_qii)
        if diag:
            nao = sum(len(GS) for GS in GS_qii)
            GS_i = np.zeros(nao, GS_qii[0].dtype)
            # Loop over diagonal elements
            i0 = 0
            for GS_ii in GS_qii:
                i1 = i0 + len(GS_ii)
                GS_i[i0:i1] = GS_ii.diagonal()
                i0 = i1
            return GS_i
        return GS_qii


    def dos(self, energy):
        GS = self.apply_overlap(energy, trace=True)
        return - GS.imag / np.pi

    def pdos(self, energy):
        p = self.parameters
        calc = p['calc']
        n_a = len(calc.atoms)
        # Diagonal elements of GS product
        GS_i = self.apply_overlap(energy, diag=True).imag
        # Sum diagonal elements per atoms
        GS_a = np.zeros(n_a)
        i0 = 0
        for a0 in range(n_a):
            i1 = i0 + calc.wfs.setups[a0].nao
            GS_a[a0] = sum(GS_i[i0:i1])
            i0 = i1
        return - GS_a / np.pi
