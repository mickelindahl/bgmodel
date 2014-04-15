'''
Created on Apr 9, 2014

@author: lindahlm
'''
import nest #Need to be imported before MPI is impoorted!!!
import mpi4py.MPI as MPI

class comm(object):
    '''dependancy injection'''
    obj=MPI.COMM_WORLD
    
    @classmethod
    def is_mpi_used(cls):
        return cls.obj.Get_size()-1
     
    @classmethod
    def Get_size(cls):
        return cls.obj.Get_size()


