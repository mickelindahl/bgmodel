

#import mpi4py.MPI as MPI
#import nest
from toolbox.mpi_comm_wrap import comm
def fun(_id):
    if _id==id(comm.obj):
        print 'Same'
    else:
        print 'Different'