'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
from toolbox import misc
import pprint
pp=pprint.pprint

def get():
    
    def get_perturbation_dics(c, hz_mod, i):
        d = {}
        for key in c.keys():
            for neuron, hz_pA in c[key]:
                hz=hz_mod[neuron][i]
                u = {key:{'nest':{neuron:{'I_e':1./hz_pA*hz}}}}
                d = misc.dict_update(d, u)
        return d
                
    c={'MS':[['M1_low',5./32], #5/32 Hz/pA 
             ['M2_low',5./40]], #5/40 Hz/pA
       'M1':[['M1_low',5./32]], #5/32 Hz/pA 
       'M2':[['M2_low',5./40]], #5/40 Hz/pA
       'FS':[['FS_low',10./60]], #10/60 Hz/pA
       'GP':[['GA',0.47], #0.47 Hz/pA
             ['GI',0.47]], #0.47 Hz/pA
       'GA':[['GA',0.47]], #0.47 Hz/pA
       'GI':[['GI',0.47]], #0.47 Hz/pA
       'ST':[['ST',0.54]], #0.54 Hz/pA
       'SN':[['SN',0.16]]} #0.16 Hz/pA
    
       
    #in hz
    mod={'M1_low':[-3,-2,-1, #  
                   1, 2, 3 ],
         'M2_low':[-3,-2,-1, #  
                    1, 2, 3 ],
         'FS_low':[-15,-10,-5, #  
                   5, 10, 15 ],
         'GA':[-15,-10,-5, #  
                5, 10, 15 ],
         'GI':[-30,-20,-10, #  
                10, 20, 30],
         'ST':[-15,-10,-5, #  
                5, 10, 15 ],
         'SN':[-30,-20,-10, #  
                10, 20, 30]}
         
    l=[]
    
    l+=[pl(**{'name':'no_pert'})]
    
    for key, val in c.items():
        for i in range(6):
            d=get_perturbation_dics({key:val}, mod, i)
            
            l.append(pl(d.values()[0],'+', **{'name':(d.keys()[0]
                                                      +'_pert_mod'
                                                      +str(i))}))
        
    return l 
 
if __name__=='__main__':
    get()
    pp(get())
 