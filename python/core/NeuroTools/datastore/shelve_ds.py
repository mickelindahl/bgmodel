"""
Implementation of a `shelve`-based DataStore
"""

from NeuroTools.datastore.interface import AbstractDataStore
from NeuroTools.datastore.keygenerators import join_with_underscores
import os.path, shelve
import logging

class ShelveDataStore(AbstractDataStore):
    """Persistent data store based on the `shelve` module and the filesystem."""
    
    def __init__(self, root_dir, key_generator=join_with_underscores):
        """
        `root_dir` is a filesystem directory below which all shelve files
        will be saved.
        `key_generator` is a function that accepts a mapping and returns a string.
        """
        if os.path.exists(root_dir):
            if not os.path.isdir(root_dir): # should also test if the directory is readable/writeable
                raise Exception("The supplied root_dir exists but is not a directory.")
        else:
            os.mkdir(root_dir)
        self._root_dir = root_dir
        self._generate_key = key_generator
    
    def retrieve(self, component, attribute_name):
        __doc__ = AbstractDataStore.retrieve.__doc__
        storage_key = self._generate_key(component)
        path = os.path.join(self._root_dir, storage_key+".shelf")
        if os.path.exists(path):
            shelf = shelve.open(path, flag='r') # 'r' means read-only
            if attribute_name in shelf:
                data = shelf[attribute_name]
            else:
                data = None
            shelf.close()
            return data
        else:
            return None

    def store(self, component, attribute_name, data):
        __doc__ = AbstractDataStore.store.__doc__
        storage_key = self._generate_key(component)
        try:
            path = os.path.join(self._root_dir, storage_key+".shelf")
            shelf = shelve.open(path, flag='c') # 'c' means "create if doesn't exist"
        except Exception, errmsg:
            if errmsg[1] == 'File name too long':
                logging.error("shelf filename: '%s' is too long", os.path.join(self._root_dir, storage_key+".shelf"))
            raise
        shelf.update({attribute_name: data})
        shelf.close()

