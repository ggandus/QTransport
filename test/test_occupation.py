#!/usr/bin/env python

import numpy as np
import pytest

from transport.continued_fraction import integrate_dos

@pytest.mark.parametrize('prefix',['poly2'])
@pytest.mark.parametrize('method',[#'from_gf',
                                   #'from_hs',
                                   'from_rgf'])
                                   # pytest.param('from_rgf',pytest.mark.recursive)])
def test_occupation(prefix, method, get_data, setup):

    tcalc = setup(method, prefix)
    scalc = get_data(prefix)[0]
    # Periodic boundary conditions
    tcalc.set(align_bf=None)
    tcalc.selfenergies = []
    tcalc.initialize()
    #
    GF = tcalc.greenfunction
    ocp = integrate_dos(GF)
    nvalence = scalc.wfs.nvalence
    assert np.allclose(ocp,nvalence)

# test_transmission(setup)
