#!/usr/bin/env python

import pickle
import numpy as np
from os.path import join
import pytest

from gpaw import GPAW
from ase.io import read

from transport.calculators import TransportCalculator
from transport.greenfunction import GreenFunction, RecursiveGF
from transport.lcao.principallayer import PrincipalSelfEnergy

test_filename = {'transmission': 'ET.pckl',
                 'dos': 'ED.pckl',
                 'nelectrons': 'N.pckl',
				 'density': 'rho.pckl'}
test_types = test_filename.keys()

def get_prefix(prefix):
    return join('data',prefix+'_')

def load_pickle(file):
    return pickle.load(open(file,'rb'))

@pytest.fixture(scope='module')
def get_data():
    def inner(prefix):
        prefix = get_prefix(prefix)
        atoms = read(prefix+'scatt.xyz')
        pcalc = GPAW(prefix+'leads.gpw',txt=None)
        hs_qii, hs_qij  = load_pickle(prefix+'hs_lists.pckl')
        h1_k, s1_k  = load_pickle(prefix+'hs1_k.pckl')
        return atoms, pcalc, hs_qii, hs_qij, h1_k, s1_k
    return inner

@pytest.fixture(scope='module')
def get_expected():
    def inner(prefix, test_type):
        if test_type in test_types:
            prefix = get_prefix(prefix)
            data  = load_pickle(prefix+test_filename[test_type])
        else:
            raise VauleError('Not valid test type {}'.formta(type))
        return data
    return inner


@pytest.fixture
def setup(get_data):
    # Get data
    def inner(method, prefix):
        atoms, pcalc, hs_qii, hs_qij, h1_k, s1_k = get_data(prefix)
        # Initialize selfenergies
        PS = [PrincipalSelfEnergy(pcalc, scatt=atoms, id=0), # Left
              PrincipalSelfEnergy(pcalc, scatt=atoms, id=1)] # Right
        # Initialize calculator from greenfunction
        if method == 'from_gf':
            GF = GreenFunction(h.astype(complex),
                               s.astype(complex),
                               selfenergies=PS)
            tcalc = TransportCalculator(greenfunction=GF,
                                        selfenergies=PS,
                                        align_bf=0) # Align GF
        # Initialize calculator from recursive greenfunction
        elif method == 'from_rgf':
            GF = RecursiveGF(selfenergies=PS)
            # GF.set(calc=scalc)
            tcalc = TransportCalculator(greenfunction=GF,
                                        selfenergies=PS,
                                        h1_k=h1_k,
                                        s1_k=s1_k,
                                        hs_qii=hs_qii,
                                        hs_qij=hs_qij,
                                        align_bf=0) # Align GF
        # Initialize calculator from Hamiltonian and Overlap
        elif method == 'from_hs':
            tcalc = TransportCalculator(h=h.astype(complex),
                                        s=s.astype(complex),
                                        selfenergies=PS,
                                        align_bf=0) # Align GF
        else:
            raise ValueError('Method non valid {}'.format(method))
        # Rreturn transport calculator
        return tcalc
    return inner
