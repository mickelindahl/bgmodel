"""
NeuroTools.datastore
====================

The `datastore` package aims to present a consistent interface for persistent
data storage, irrespective of storage back-end.

It is intended for objects to be able to store part or all of their internal
data, and so the storage/retrieval keys are based on the object identity and
state.

We assume that an object's identity is uniquely defined by its type (which may
also depend on the source code revision number) and its parameters, while its
state is defined by its identity and by its inputs (we should possibly add some
concept of time to this).

Hence, any object (which we call a 'component' in this context) must have
the following attributes:

`parameters`
  a `neurotools` `ParameterSet` object

`input`
  another component or `None`; we assume a single
  input for now. A list of inputs should also be possible. We need to be wary
  of recurrent loops, in which two components both have each other as direct or
  indirect inputs).

`full_type`
  the object class and module

`version`
  the source-code version

Classes
-------

ShelveDataStore    - Persistent data store based on the `shelve` module and the
                     filesystem.
DjangoORMDataStore - Persistent data store using the Django ORM
                     (object-relational mapping - an object-oriented interface
                     to an SQL database) to store/retrieve keys/indices
                     with data stored using `pickle` on the filesystem.

"""

import warnings
from shelve_ds import ShelveDataStore

# other possibilities...
#   FileSystemDataStore   
#   SRBDataStore  
#   HttpDataStore 
#   HDF5DataStore


    