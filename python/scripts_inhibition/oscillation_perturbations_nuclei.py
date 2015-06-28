'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
from core import misc
import pprint
pp=pprint.pprint

def get():
    
    def get_perturbation_dics(val, hz_mod, i):
        d = {}
#         for key in c.keys():
        for neuron, hz_pA in val:
            hz=hz_mod[neuron][i]
            u = {key:{'node':{neuron:{'I_vivo':1./hz_pA*hz}}}}
            d = misc.dict_update(d, u)
        return d
                
    c={
       'M1':[['M1',0.16]], #5/32 Hz/pA 
       'M2':[['M2',0.13]], #5/40 Hz/pA
       'FS':[['FS',0.17]], #10/60 Hz/pA
       'GP':[['GA',0.47],
             ['GI',0.47]], #0.47 Hz/pA
       'GA':[['GA',0.47]], #0.47 Hz/pA
       'GI':[['GI',0.47]], #0.47 Hz/pA
       'ST':[['ST',0.54]], #0.54 Hz/pA
       'SN':[['SN',0.16]]} #0.16 Hz/pA
    
       
    #in hz
    
    mod={'M1':[a*2 for a in [-8, -6,-4,-2, #  
                             2, 4, 6, 8]],
         'M2':[a*2 for a in [-8, -6,-4,-2, #  
                             2, 4, 6, 8]],
         'FS':[a*1.1 for a in [-20, -15,-10,-5, #  
                               5, 10, 15, 20 ]],
         'GA':[1.5*a for a in [-20, -15,-10,-5, #  
                               5, 10, 15, 20 ]],
         'GI':[a*4 for a in [-60, -45,-30,-15, #  
                              15, 30, 45, 60]],
         'ST':[a*2 for a in [-30, -22.5,-15,-7.5, #  
                               7.5, 15, 22.5, 30 ]],
         'SN':[a*4 for a in  [-40, -30,-20,-10, #  
                              10, 20, 30, 40]]}
         
    l=[]
    
    l+=[pl(**{'name':'no_pert'})]
    
    for key, val in c.items():
        for i in range(8):
            
            d=get_perturbation_dics(val, mod, i)
            l.append(pl(d.values()[0],'+', **{'name':(key
                                          +'_pert_mod'
                                          +str(int(i)))}))
# neuron, hz_pA=val
#         for neuron, mods in mod.items():
#             for hz in mods:
#                 d=get_perturbation_dics({key:val}, mod, i)
#                 d={key:{'node':{neuron:{'I_vivo':1./hz_pA*hz}}}}
#                 
#                 l.append(pl(d.values()[0],'+', **{'name':(d.keys()[0]
#                                                           +'_pert_mod'
#                                                           +str(int(hz*hz_pA)))}))
#         
    return l 
 
if __name__=='__main__':

    l=get()
    for i, e in enumerate(l):
        print i,e
 