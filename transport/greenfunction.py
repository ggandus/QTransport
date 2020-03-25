import numpy as np
from numpy import linalg
# from functools import lru_cache
from .coupledhamiltonian import CoupledHamiltonian
from .tools import dagger

#Recursive GreenFunction helpers
from .solvers.tridiagonal   import tridiagonalize, cutoff
from .solvers.recursive import get_mat_lists, recursive_gf


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

        #Note h_im[:,:pl1]=h_ij for Left and h_im[:,-pl2:]=h_ij for Right
        for selfenergy in self.selfenergies:
            selfenergy.h_im = selfenergy.h_ij
            selfenergy.s_im = selfenergy.s_ij
            selfenergy.sigma_mm = np.empty((selfenergy.nbf_i, selfenergy.nbf_i),
                                            dtype=selfenergy.sigma_mm.dtype)

        self.initialized = True


    def _get_mat_lists(self, energy):
        z = energy + self.eta * 1.j
        sigma_L = self.selfenergies[0].retarded(energy)
        sigma_R = self.selfenergies[1].retarded(energy)
        # mat_list_ii, mat_list_ij, mat_list_ji
        return get_mat_lists(z, self.hs_list_ii, self.hs_list_ij,
                             sigma_L, sigma_R)


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


    def get_dos(self, energies, dos_e=None):
        N = len(self.hs_list_ii[0])

        if dos_e is None:
            dos_e = np.zeros(len(energies))

        for e, energy in enumerate(energies):

            self.G_ii, self.G_ij, self.G_ji = recursive_gf(*self._get_mat_lists(energy),
                                                           dos=True)

            for q in range(N):
                dos_e[e] += np.trace(self.G_ii[q] @ self.hs_list_ii[1][q]).imag
                if q < N-1:
                    dos_e[e] += np.trace(self.G_ij[q] @ self.hs_list_ij[1][q].T).imag
                if q > 0:
                    dos_e[e] += np.trace(self.G_ji[q-1] @ self.hs_list_ij[1][q-1]).imag

        return - dos_e / np.pi
