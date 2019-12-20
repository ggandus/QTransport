from tbnegf.transport import mpi
import numpy as np
timer = mpi.MPI
t1=timer.Wtime()
vect=mpi.get_domain(np.arange(100000000))
print(np.min(vect),np.max(vect),mpi.rank)
np.square(vect, out=vect)
vect=mpi.gather(vect)
t2=timer.Wtime()
print(t2-t1)
