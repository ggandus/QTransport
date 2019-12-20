import numpy as np

from transport import _cpp #import _cpp
from transport.greenfunction import GreenFunction
from transport.internalselfenergy import InternalSelfEnergy
from transport.selfenergy import LeadSelfEnergy

val=3
eta=1e-5
m=3
i=2

h_mm = np.diag([val]*m)
s_mm = np.eye(m)
h_ii = np.diag([val]*i)
s_ii = np.eye(i)
h_im = np.zeros((i,m))
h_im[0,0]=1.
h_im[1,1]=1.
s_im = np.zeros((i,m))
h_ij = np.eye(i)
s_ij = np.zeros((i,i))

gf = _cpp.GreenFunction(h_mm,s_mm,eta)
pygf = GreenFunction(h_mm,s_mm,eta=eta)
assert np.allclose(gf.retarded(1.),np.diag([1-val]*m))
assert np.allclose(gf.retarded(1.),pygf.retarded(1.,inverse=True))

se = _cpp.SelfEnergy(h_ii,s_ii,h_im,s_im,eta)
pyse = InternalSelfEnergy((h_ii,s_ii),(h_im,s_im),eta=eta)
assert np.allclose(se.retarded(1.),np.array([[-0.5,0,0],[0,-0.5,0],[0,0,0]]))
assert np.allclose(se.retarded(1.),pyse.retarded(1.))

lse = _cpp.LeadSelfEnergy(h_ii,s_ii,h_ij,s_ij,h_im,s_im,eta)
pylse = LeadSelfEnergy((h_ii,s_ii),(h_ij,s_ij),(h_im,s_im),eta=eta)
assert np.allclose(lse.retarded(1.),pylse.retarded(1.))
