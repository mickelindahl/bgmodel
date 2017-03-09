'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
from core import misc
import pprint
pp=pprint.pprint


def get_change_to(flag, E_rev='low'):
    x=0.
    #1 *
    if flag=='M1':
        if E_rev=='low':
            d={'nest':{'M1_low':{'beta_d':x,
                                 'beta_E_L':x,
                                 'beta_V_b':x,
                             }}}
    #2 *
    if flag=='CTX_M1':
        if E_rev=='low':
            d={'nest':{'M1_low':{
                             'beta_I_NMDA_1':x,
                                }}}
    #3 *
    if flag=='CTX_M2':
        if E_rev=='low':
            d={'nest':{'M2_low':{'beta_I_AMPA_1':x,
                         }}}    

    #4 *
    if flag=='FS_FS':
        if E_rev=='low':
            d={'nest':{'FS_low':{'beta_I_GABAA_1':x,
                                }},
               }

    #5 * //not used (shoudl also be GA not GP
    if flag=='GP_FS':
        if E_rev=='low':
            d={'nest':{'FS_low':{'beta_I_GABAA_2':x,
                                 'beta_I_GABAA_3':x,
                                 }},
                }

    #6 * 
    if flag=='FS_M2':
        if E_rev=='low':
            d={
               'conn':{'FS_M2_gaba':{'beta_fan_in':x}}}
    
    #7 *
    if flag=='CTX_ST':
        d={'nest':{'ST':{'beta_I_AMPA_1':x,
                         'beta_I_NMDA_1':x,
                         }}}
    
    #8 *
    if flag=='GP_ST':
        d={'nest':{'ST':{'beta_I_GABAA_1':x,
                         }}}

    #9 *
    if flag=='GP':
        d={'nest':{'GI':{'beta_E_L':x,
                         'beta_V_a':x},
                   'GF':{'beta_E_L':x,
                         'beta_V_a':x},
                   'GA':{'beta_E_L':x,
                         'beta_V_a':x},
           }}
    
    #10 *
    if flag=='ST_GP':
            d={'nest':{'GI':{'beta_I_AMPA_1':x},
                       'GA':{'beta_I_AMPA_1':x}
                       }}
    
    #11 *
    if flag=='GP_GP':
        d={'nest':{'GI':{'beta_I_GABAA_2':x},
                   'GA':{'beta_I_GABAA_2':x},
                   'GF':{'beta_I_GABAA_2':x}
                   }}

    #12 *
    if flag=='SN':
        d={'nest':{'SN':{'beta_E_L':x,
                         'beta_V_a':x,
                         }}}
    
    #13 *
    if flag=='M1_SN':
        d={'nest':{'SN':{'beta_I_GABAA_1':x,
                         }}}
        
    #14 *
    if flag=='MS_MS':
        if E_rev=='low':
            d={'nest':{'M1_low':{'beta_I_GABAA_2':x},
                       'M2_low':{'beta_I_GABAA_2':x}}}
        
        for conn in ['M1_M1_gaba', 'M1_M2_gaba','M2_M1_gaba','M2_M2_gaba']:
            misc.dict_update(d,{'conn':{conn:{'beta_fan_in': x}}})         
    
    #15 *
    if flag=='M2_GI':
        d={'nest':{'GI':{'beta_I_GABAA_1':x,
                         }}}



    #16 *
    if flag=='FS':
        d={'nest':{'FS':{'beta_E_L':x,
                         }}}
        

    
    return d
    
def get():
    
    l=[]
    
    d={}
    for name in ['M1', 'CTX_M1', 'CTX_M2', 'MS_MS', 'FS_FS', 
                 'GP_FS', 'FS_M2', 'M2_GI', 'FS']:

        d=misc.dict_update(d, get_change_to(name))
        
    l+=[pl(d,'*', **{'name':'no_ch_dop-striatum'})]
    

    d={}
    for name in ['CTX_ST', 'GP_ST', 'GP', 'ST_GP', 'GP_GP', 'SN', 'M1_SN']:
        d=misc.dict_update(d, get_change_to(name))
        
    l+=[pl(d,'*', **{'name':'no_ch_dop-GP_ST_SN'})]
    
    
    d={}
    for name in ['M1', 'CTX_M1', 'CTX_M2', 'MS_MS', 'FS_FS', 
                 'GP_FS', 'FS_M2', 'CTX_ST', 'GP_ST', 'GP',
                 'ST_GP', 'GP_GP', 'SN', 'M1_SN',
                     'M2_GI', 'FS']:
        d=misc.dict_update(d, get_change_to(name))
        pp(d)
        
    l+=[pl(d,'*', **{'name':'no_ch_dop-all'})]

    for s in [ 'CTX_M1', 'CTX_M2', 'MS_MS', 'FS_FS', 
              'GP_FS', 'FS_M2', 'CTX_ST', 'GP_ST', 'GP',
              'ST_GP', 'GP_GP', 'SN', 'M1_SN', 'M2_GI',
              'FS','M1']:
        d={}
        for name in [s]:
            d=misc.dict_update(d, get_change_to(name))
            
        l+=[pl(d,'*', **{'name':'no_ch-'+s})]    
    
    d={}
    l+=[pl(d,'*', **{'name':'-Normal'})]   
        
    return l
 
 
if __name__=='__main__':
    l=get()
    for i, e in enumerate(l):
        print i,e