'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl

import pprint
pp=pprint.pprint


def get():
    
    l=[]
    for rate in [680., 700., 720.]:
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                        'GA_M2_gaba':{'weight':0.25},
                        'M1_M1_gaba':{'weight':0.25},
                        'M1_M2_gaba':{'weight':0.25},
                        'M2_M1_gaba':{'weight':0.25},
                        'M2_M2_gaba':{'weight':0.25},
                        'ST_GA_ampa':{'weight':0.25},
                        'GA_GA_gaba':{'weight':0.25},
                        'GI_GA_gaba':{'weight':0.25}
                        }},
               '*',
                **{'name':'MsGa-MS-weight0.25_C2r-'+str(rate)})]  
        l[-1]+=pl({
                  'node':{'C1':{'rate':560.0},
                          'C2':{'rate':rate},
                          'EI':{'rate':1060.0},
                          'EA':{'rate':330.0}}},
               '=',
                **{'name':''})  

        
    for w in [0.75, 0.5, 0.25]:
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':0.25},
                        'GA_M2_gaba':{'weight':0.25},
                        'M1_M1_gaba':{'weight':0.25},
                        'M1_M2_gaba':{'weight':0.25},
                        'M2_M1_gaba':{'weight':0.25},
                        'M2_M2_gaba':{'weight':0.25},
                        'ST_GA_ampa':{'weight':0.25},
                        'GA_GA_gaba':{'weight':0.25},
                        'GI_GA_gaba':{'weight':0.25},
                        'GI_ST_gaba':{'weight':w},
                        'ST_GI_ampa':{'weight':w}
                        }},
               '*',
                **{'name':'MsGa-MS-weight0.25_ST-GI-'+str(w)})]  
        l[-1]+=pl({
                  'node':{'C1':{'rate':560.0},
                          'C2':{'rate':700.},
                          'EI':{'rate':1060.0},
                          'EA':{'rate':330.0}}},
               '=',
                **{'name':''})  
   
    for w in [0.75, 0.5, 0.25]:
        for x in [2.5, 5.]:
            v=.25
            l+=[pl({'nest':{
                            'M1_M1_gaba':{'weight':0.25},
                            'M1_M2_gaba':{'weight':0.25},
                            'M2_M1_gaba':{'weight':0.25},
                            'M2_M2_gaba':{'weight':0.25},
                            'ST_GA_ampa':{'weight':0.25},
                            'GA_GA_gaba':{'weight':0.25},
                            'GI_GA_gaba':{'weight':0.25},
                            'GI_ST_gaba':{'weight':w},
                            'ST_GI_ampa':{'weight':w}
                            }},
                   '*',
                    **{'name':'MsGa-MS-weight0.25_ST-GI-'+str(w)})]  
            l[-1]+=pl({
                      'node':{'C1':{'rate':560.0},
                              'C2':{'rate':700.},
                              'EI':{'rate':1060.0},
                              'EA':{'rate':330.0}}},
                   '=',
                    **{'name':'down-C2-EiEa-mod'})    
       
            l[-1]+=pl({'nest':{'GA_M1_gaba':{'weight':v*x}, 
                               'GA_M2_gaba':{'weight':v*x},
                               'GA_FS_gaba':{'weight':x},
                               'M1_low':{'GABAA_3_Tau_decay':1./5},  
                               'M2_low':{'GABAA_3_Tau_decay':1./5},
                               'FS_low':{'GABAA_2_Tau_decay':1./5},
                       
                       }},
              '*',
               **{'name':'fast-'+str(x)})   

    for y in [0.25, 0.3, 0.35]:
        w=0.75
        l+=[pl({'nest':{'GA_M1_gaba':{'weight':y},
                        'GA_M2_gaba':{'weight':y},
                        'M1_M1_gaba':{'weight':0.25},
                        'M1_M2_gaba':{'weight':0.25},
                        'M2_M1_gaba':{'weight':0.25},
                        'M2_M2_gaba':{'weight':0.25},
                        'ST_GA_ampa':{'weight':0.25},
                        'GA_GA_gaba':{'weight':0.25},
                        'GI_GA_gaba':{'weight':0.25},
                        'GI_ST_gaba':{'weight':w},
                        'ST_GI_ampa':{'weight':w}
                        }},
               '*',
                **{'name':'MsGa-MS-weight0.25_ST-GI-'+str(w)
                   + '-GaMs-'+str(y)})] 
        l[-1]+=pl({
                  'node':{'C1':{'rate':560.0},
                          'C2':{'rate':700.},
                          'EI':{'rate':1060.0},
                          'EA':{'rate':330.0}}},
               '=',
                **{'name':'down-C2-EiEa-mod'})  


    for y in [0.2, 0.25, 0.3, 0.35]:
        x=5.
        w=0.75
        l+=[pl({'nest':{'M1_M1_gaba':{'weight':0.25},
                        'M1_M2_gaba':{'weight':0.25},
                        'M2_M1_gaba':{'weight':0.25},
                        'M2_M2_gaba':{'weight':0.25},
                        'ST_GA_ampa':{'weight':0.25},
                        'GA_GA_gaba':{'weight':0.25},
                        'GI_GA_gaba':{'weight':0.25},
                        'GI_ST_gaba':{'weight':w},
                        'ST_GI_ampa':{'weight':w}
                        }},
               '*',
                **{'name':'MsGa-MS-weight0.25_ST-GI-'+str(w)
                   + '-GaMs-'+str(y)})]  
        l[-1]+=pl({
                  'node':{'C1':{'rate':560.0},
                          'C2':{'rate':700.},
                          'EI':{'rate':1060.0},
                          'EA':{'rate':330.0}}},
               '=',
                **{'name':'down-C2-EiEa-mod'}) 

        l[-1]+=pl({'nest':{'GA_M1_gaba':{'weight':y*x}, 
                           'GA_M2_gaba':{'weight':y*x},
                           'GA_FS_gaba':{'weight':x},
                           'M1_low':{'GABAA_3_Tau_decay':1./5},  
                           'M2_low':{'GABAA_3_Tau_decay':1./5},
                           'FS_low':{'GABAA_2_Tau_decay':1./5},
                   
                   }},
          '*',
           **{'name':'fast-'+str(x)})   
  
    return l
 