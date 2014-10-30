'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
from toolbox import misc
import pprint
pp=pprint.pprint


c={'FS_FS':['FS_FS_gaba'],
   'FS_MS':['FS_M1_gaba',
            'FS_M2_gaba'],
   'GA_FS':['GA_FS_gaba'],
   'GA_GA':['GA_GA_gaba'],
   'GA_GI':['GA_GI_gaba'],
   'GA_M1':['GA_M1_gaba'],
   'GA_M2':['GA_M2_gaba'],
   'GI_GA':['GI_GA_gaba'],
   'GI_GI':['GI_GI_gaba'],
   'GI_SN':['GI_SN_gaba'],
   'GI_ST':['GI_ST_gaba'],
   'M1_SN':['M1_SN_gaba'],
   'M2_GI':['M2_GI_gaba'],
   'MS_MS':['M1_M1_gaba',
            'M1_M2_gaba',
            'M2_M1_gaba',
            'M2_M2_gaba'],
   'ST_GA':['ST_GA_ampa'],
   'ST_GI':['ST_SN_ampa'],
   }
mod=[0.75, 1.25,
     0.5, 1.5, 
     0.25, 1.75]


def get_perturbation_dics(c, w_rel):
    for key in c.keys():
        d = {}
        for conn in c[key]:
            u = {key:{'node':{conn:{'weight':w_rel}}}}
            d = misc.dict_update(d, u)

def get():
    
    l=[]
    
    l+=[pl(**{'name':'no_pert'})]
    for w_rel in mod:
        d=get_perturbation_dics(c, w_rel)
        for key, val in d.idems():
            l+=pl(val,'*', **{'name':key+'_pert_'+str(w_rel)})
        
    return l
 