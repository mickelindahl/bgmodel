'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
from toolbox import misc
import pprint
pp=pprint.pprint


def get_change_to(flag, E_rev='low'):
    x=0.
    if flag=='M1':
        if E_rev=='low':
            d={'nest':{'M1_low':{'beta_d':x,
                             'beta_E_L':x,
                             'beta_V_b':x,
                             'beta_I_NMDA_1':x,
                             }}}
    
    if flag=='M2':
        if E_rev=='low':
            d={'nest':{'M2_low':{'beta_I_AMPA_1':x,
                         }}}    

    if flag=='FS':
        if E_rev=='low':
            d={'nest':{'FS_low':{'beta_I_GABAA_1':x,
                                 'beta_I_GABAA_2':x,
                             }},
               'conn':{'FS_M2_gaba':{'beta_fan_in':x}}}

    if flag=='ST':
        d={'nest':{'ST':{'beta_I_AMPA_1':x,
                         'beta_I_NMDA_1':x,
                         'beta_I_GABAA_1':x,
                         }}}

    if flag=='GI':
        d={'nest':{'GI':{'beta_E_L':x,
                         'beta_V_a':x,
                         'beta_I_AMPA_1':x,
                         'beta_I_GABAA_2':x,
                         }}}

    if flag=='GA':
        d={'nest':{'GA':{'beta_E_L':x,
                         'beta_V_a':x,
                         'beta_I_AMPA_1':x,
                         'beta_I_GABAA_2':x,
                         }}}

    if flag=='SN':
        d={'nest':{'SN':{'beta_I_GABAA_1':x,
                         'beta_E_L':x,
                         'beta_V_a':x,
                         }}}
       
    return d

def get():
    
    l=[]
    
    d={}
    for name in ['M1', 'M2', 'FS']:
        d=misc.dict_update(d, get_change_to(name))
        
    l+=[pl(d,'*', **{'name':'no_ch_dop_striatum'})]
    

    d={}
    for name in ['ST', 'GI', 'GA','SN']:
        d=misc.dict_update(d, get_change_to(name))
        
    l+=[pl(d,'*', **{'name':'no_ch_dop_rest'})]
    
    
    d={}
    for name in ['M1', 'M2', 'FS', 'ST', 'GI', 'GA','SN']:
        d=misc.dict_update(d, get_change_to(name))
        
    l+=[pl(d,'*', **{'name':'no_ch_dop_all'})]


    d={}
    for name in ['M2']:
        d=misc.dict_update(d, get_change_to(name))
        
    l+=[pl(d,'*', **{'name':'no_ch_M2'})]    
     
    return l
 