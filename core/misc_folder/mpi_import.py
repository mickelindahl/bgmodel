import toolbox


print 'l'

from toolbox.mpi_comm_wrap import comm
from mpi_import2 import fun
#import mpi4py.MPI as MPI #import last

#obj=MyMpi()
fun(id(comm.obj))
#p= MPI.COMM_WORLD.Get_size()

