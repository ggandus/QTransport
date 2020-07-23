import numpy as np
from collections import namedtuple

fields = ['M_a','nao_a','nao','atoms']
Calc = namedtuple('calc', fields)

def initialize_calculator(calc):
    calc.atoms.set_calculator(calc)
    if calc.wfs.S_qMM is None:
        # Initialize calculator
        calc.wfs.set_positions(calc.spos_ac)

def make_calc(atoms, basis):
    M_a = np.zeros(len(atoms),dtype=int)
    nao_a = np.zeros(len(atoms),dtype=int)
    nao = 0
    # Init
    M_a[0] = 0
    nao_a[0] = basis[atoms[0].symbol]
    nao += nao_a[0]
    for a0 in range(1,len(atoms)):
        nao_a[a0] = basis[atoms[a0].symbol]
        M_a[a0] = M_a[a0-1] + nao_a[a0-1]
        nao += nao_a[a0]
    return Calc(M_a, nao_a, nao, atoms)


def get_info(calc, attributes):
    info = []
    gpaw = hasattr(calc, 'wfs')
    # From gpaw calculator
    if gpaw:
        initialize_calculator(calc)
        natoms = len(calc.atoms)
        for attr in attributes:
            if attr in ['M_a','nao']:
                info.append(getattr(calc.setups, attr))
            elif attr in ['nao_a']:
                info.append([calc.setups[a0].nao for a0 in range(natoms)])
            else:
                raise NotImplementedError('{}'.format(attr))
    else:
        # From namedtuple
        for attr in attributes:
            info.append(getattr(calc, attr))
    return info
