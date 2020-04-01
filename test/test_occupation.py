#!/usr/bin/env python

import numpy as np
import pytest

from transport.continued_fraction import integrate_dos

@pytest.mark.parametrize('prefix',['poly2'])
@pytest.mark.parametrize('method',[#'from_gf',
                                   #'from_hs',
                                   'from_rgf'])
                                   # pytest.param('from_rgf',pytest.mark.recursive)])
def test_occupation(prefix, method, get_expected, setup):

    tcalc = setup(method, prefix)
    tcalc.initialize()
    N = get_expected(prefix, 'nelectrons')
    #
    GF = tcalc.greenfunction
    nele = integrate_dos(GF)
    assert np.allclose(nele, N)

# test_transmission(setup)
