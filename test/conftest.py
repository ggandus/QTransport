#!/usr/bin/env python

from gpaw import GPAW
import pickle
import numpy as np
from os.path import join

import pytest

from transport.calculators import TransportCalculator
from transport.greenfunction import GreenFunction, RecursiveGF
from transport.lcao.principallayer import PrincipalSelfEnergy

test_types = ['transmission','dos']
test_filename = {'transmission': 'ET.pckl',
                 'dos': 'ED.pckl'}

def get_prefix(prefix):
    return join('data',prefix+'_')

def load_pickle(file):
    return pickle.load(open(file,'rb'))

@pytest.fixture(scope='module')
def get_data():
    def inner(prefix):
        prefix = get_prefix(prefix)
        scalc = GPAW(prefix+'scatt.gpw',txt=None)
        pcalc = GPAW(prefix+'leads.gpw',txt=None)
        h, s  = load_pickle(prefix+'hs_scat.pckl')
        return scalc, pcalc, h, s
    return inner

@pytest.fixture(scope='module')
def get_expected():
    def inner(prefix, test_type):
        if test_type in test_types:
            prefix = get_prefix(prefix)
            E, T  = load_pickle(prefix+test_filename[test_type])
        else:
            raise VauleError('Not valid test type {}'.formta(type))
        return E, T
    return inner


@pytest.fixture
def setup(get_data):
    # Get data
    def inner(method, prefix):
        scalc, pcalc, h, s = get_data(prefix)
        # Initialize selfenergies
        PS = [PrincipalSelfEnergy(pcalc, scatt=scalc, id=0), # Left
              PrincipalSelfEnergy(pcalc, scatt=scalc, id=1)] # Right
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
            GF = RecursiveGF(h.astype(complex),
                             s.astype(complex),
                             selfenergies=PS,
                             calc=scalc)
            # GF.set(calc=scalc)
            tcalc = TransportCalculator(greenfunction=GF,
                                        selfenergies=PS,
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
