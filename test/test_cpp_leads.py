import numpy as np

from transport import _cpp
from transport.selfenergy import LeadSelfEnergy
from ase.transport.selfenergy import LeadSelfEnergy as aseLSE
from timeit import default_timer as timer


val=3
eta=1e-5
m=3
i=100
dtype=np.complex64

h_ii = np.array(np.diag([val]*i),dtype=dtype,order='F')
s_ii = np.eye(i,dtype=dtype,order='F')
h_im = np.zeros((i,m),dtype=dtype,order='F')
h_im[0,0]=1.
h_im[1,1]=1.
s_im = np.zeros((i,m),dtype=dtype,order='F')
h_ij = np.eye(i,dtype=dtype,order='F')
s_ij = np.zeros((i,i),dtype=dtype,order='F')


#lse = _cpp.LeadSelfEnergy(h_ii,s_ii,h_ij,s_ij,h_im,s_im,eta)
pylse = LeadSelfEnergy((h_ii,s_ii),(h_ij,s_ij),(h_im,s_im),eta=eta)
aselse = aseLSE((h_ii,s_ii),(h_ij,s_ij),(h_im,s_im),eta=eta)

start = timer()
aseres = aselse.retarded(1.)
end = timer()
print('ase: ',end - start) # Time in seconds, e.g. 5.38091952400282
start = timer()
myres = pylse.retarded(1.)
end = timer()
print('my: ',end - start) # Time in seconds, e.g. 5.38091952400282
assert np.allclose(aseres,myres)
