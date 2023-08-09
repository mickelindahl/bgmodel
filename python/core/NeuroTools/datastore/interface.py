class AbstractDataStore(object):
    """
    Abstract base class for a persistent data store.
    """
    
    def retrieve(self, component, attribute_name):
        """
        Retrieve data belonging to a component.
        
        The component must have the following attributes:
          `parameters`: a `NeuroTools` `ParameterSet` object
          `input`: another component, or None
          `full_type`: the object class and module
          `version`: the source-code version
        """
        # construct a way to get the data if it exists
        # e.g. create a unique key, an SQL query
        # this will involve getting a key or something
        # for the input object
        # try to get the data
        # if the data exists,
        #  return it
        # else
        #  return None
        raise NotImplemented()

    def store(self, component, attribute_name, data):
        """
        Store data belonging to a component.
        
        The component must have the following attributes:
          `parameters`: a `NeuroTools` `ParameterSet` object
          `input`: another component, or None
          `full_type`: the object class and module
          `version`: the source-code version
        """
        # check we know how to handle the data type
        # construct a way to store the data, e.g. create a unique key,
        # an SQL query, etc
        # store the data
        # possibly we could check if data already exists, and raise an Exception if
        # it is different to the new data (should probably be a flag to control this,
        # because it might be heavyweight
        raise NotImplemented()