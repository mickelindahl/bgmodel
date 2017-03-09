'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
from core import misc
import pprint
pp=pprint.pprint

def get(index=[]):
    
    def get_perturbation_dics(c, w_rel):
        d = {}
        for key in c.keys():
            for conn in c[key]:
                u = {key:{'nest':{conn:{'weight':w_rel}}}}
                d = misc.dict_update(d, u)
        return d
                
    c={'FS_FS':['FS_FS_gaba'],
       'FS_MS':['FS_M1_gaba',
                'FS_M2_gaba'],
       'FS_M1':['FS_M1_gaba'],
       'FS_M2':['FS_M2_gaba'],
       'GA_FS':['GA_FS_gaba'],
       'GA_GA':['GA_GA_gaba'],
       'GA_GI':['GA_GI_gaba',
                'GA_GF_gaba'],
       'GA_M1':['GA_M1_gaba'],
       'GA_M2':['GA_M2_gaba'],
       'GI_FS':['GF_FS_gaba'],
       'GI_GA':['GI_GA_gaba'],
       'GI_GI':['GI_GI_gaba',
                'GI_GF_gaba',
                'GF_GI_gaba',
                'GF_GF_gaba'],
       'GI_SN':['GI_SN_gaba'],
       'GI_ST':['GI_ST_gaba'],
       'GP_FS':['GA_FS_gaba',
                'GF_FS_gaba'],
       'M1_SN':['M1_SN_gaba'],
       'M2_GI':['M2_GI_gaba'],
       'MS_MS':['M1_M1_gaba',
                'M1_M2_gaba',
                'M2_M1_gaba',
                'M2_M2_gaba'],
       'M1_M1':['M1_M1_gaba'],
       'M1_M2':['M1_M2_gaba'],
       'M2_M1':['M2_M1_gaba'],
       'M2_M2':['M2_M2_gaba'],
       'ST_GA':['ST_GA_ampa'],
       'ST_GI':['ST_GI_ampa',
                'ST_GF_ampa'],
       'ST_SN':['ST_SN_ampa'],
       }
#     mod=[0.06125, 0.125, 0.25, 0.5, 0.75, # 0.75, 
#          1.33, 2, 4, 8, 16]
    mod=[0.0, 0.001, 0.01, 0.1, 2, 4, 5, 8, 10, 16]
    
    if index:
        mod=[m for i,m in enumerate(mod) if i in index]
        
    l=[]
    
    l+=[pl(**{'name':'no_pert'})]
    
    for key, val in c.items():
        for w_rel in mod:
            d=get_perturbation_dics({key:val}, w_rel)
            
            l.append(pl(d.values()[0],'*', **{'name':d.keys()[0]+'_pert_'+str(w_rel)}))
        
    return l 
 
if __name__=='__main__':
    get()
 