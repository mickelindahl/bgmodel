'''
Created on May 19, 2014

@author: mikael
'''
from toolbox.network.manager import (Builder_abstract,
                                     Mixin_dopamine,
                                     Mixin_general_network,
                                     Mixin_reversal_potential_striatum)
from toolbox.network.default_params import Inhibition
from toolbox.network.default_params import Perturbation_list as pl


class Builder_network_base(Builder_abstract):    
    
    def _variable(self):
        
        l=[]
        l+=[pl(**{'name':'no_pert'})]
        l+=[pl({'nest':{'ST_GI_ampa':{'weight':2.0}}},
               '*', **{'name':'Doubke STN-GPE-TI'})]
        
        
        # Set directory where connection data is stored
        from os.path import expanduser
        home = expanduser("~")
        path_conn=(home+ '/results/papers/inhibition/network/jyotika/conn/')
        for i in range(len(l)):
            l[i]+=pl({'simu':{'path_conn':path_conn}},
               '=', **{'name':'Doubke STN-GPE-TI'})  
        
        return l
    

    def _get_dopamine_levels(self):
        return [self._dop()]
    
    
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per):
        return Inhibition(**{'perturbations':per})
    
class Builder_network(Builder_network_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass