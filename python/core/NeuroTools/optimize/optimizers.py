"""
===========
optimizers.py
===========

Contains functions to search minima in parameter space. Uses parameter_search
to do the actual searching.

$Id: optimizers.py 366 2008-12-23 21:58:19Z mschmucker $
"""

__author__="Michael Schmuker"
__date__ ="$23.12.2008 11:17:16$"

from NeuroTools.optimize import parameter_search
from NeuroTools.parameters import ParameterSpace

class GridSearcher():
    """
    Simple grid search algorithm. Calls the error function func with all
    parameters in grid. Returns the first parameter combination which yielded
    the minimal value for func, together with that value in a dictionary.
    """
    def __init__(self, grid, func,
                 searcher = parameter_search.ParameterSearcher,
                 searcherargs = {}):
        """
        Initialize the grid searcher.
        Parameters:

        grid - NeuroTools.ParameterSpace scpecifying the grid.
        func - function to minimize. It should take a dictionary with its
               parameters and return a float.
        searcher - the searcher backend to use. Should be of type
                   NeuroTools.optimize.parameter_search.ParameterSearcher
                   or a child thereof. Default is to use the plain
                   ParameterSearcher.
        searcherargs  - dictionary with additional keyword arguments for the searcher.
        """
        import types
        if type(grid) != ParameterSpace:
            raise Exception("The grid must be defined as " +
                            "NeuroTools.ParameterSpace.")
        self.grid = grid
        param_iter = grid.iter_inner()
        if type(func ) != types.FunctionType:
            raise Exception("func must be a function.")
        self.searcher = searcher(dict_iterable = param_iter,
                                 func = func,
                                 **searcherargs)

    def search(self):
        """
        Do the actual searching.
        """
        min_params = None
        self.searcher.search()
        retvals = self.searcher.harvest()
        import numpy
        minindex = numpy.argmin(retvals)
        min_val = retvals[minindex]
        # retrieve the parameter combination that yielded the minimum value
        tmp_iter = self.grid.iter_inner()
        for i in range(minindex):
            tmp_iter.next()
        min_params = tmp_iter.next()
        return {'min_params': min_params, 'min_value':min_val}




        