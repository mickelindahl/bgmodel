'''
Created on Jul 4, 2013

@author: lindahlm
'''

from copy import deepcopy
from toolbox.network.handling_single_units import Activity_model_dic
from toolbox import misc, plot_settings
import pylab

labels_dic={0:'$FSN_{+d}^{l}$',  
            1:'$FSN_{-d}^{l}$', 
            2:'$FSN_{+d}^{h}$',  
            3:'$FSN_{-d}^{h}$'}

def get_kwargs_dics(stop):

    k={}
    for name, label in labels_dic.items():
        k[name]={'input_models':['CFp', 'GAp', 'FSp', 'FS'], 

                   'label':label,
                   'par_rep':{'simu':{'threads':1, 
                                       'start_rec':1000.0, 
                                        'stop_rec':stop, 
                                        'sim_time':stop,
                                        'sim_stop':stop, 
                                        'print_time':False},
                              'netw':{'size':50.}},
                    'stim_model': 'CFp',
                    'study_name':'FS',
                    'verbose':True,
                   }
         
    
    l=[ {'keys':['par_rep.node.FS.model'], 'val':'FS_low', 'names':[0,1]},
        {'keys':['par_rep.node.FS.model'], 'val':'FS_high', 'names':[2,3]},
        
        {'keys':['par_rep.node.CFp.lesion',
                 'par_rep.node.GAp.lesion',
                 'par_rep.node.FSp.lesion',
                 'par_rep.node.FS.lesion'], 'val':False, 'names':[0,1,2,3]},    
             
        {'keys':['par_rep.netw.tata_dop'], 'val':0.8, 'names':[0,2]},
        {'keys':['par_rep.netw.tata_dop'], 'val':0.0, 'names':[1,3]},]
    
    
    for d in l:
        for name in d['names']:
            for key in d['keys']:
                misc.dict_recursive_add(k[name], 
                                        key.split('.'), d['val'])
                
    return k
        
def main():
    stop=21000.0
    amd=Activity_model_dic()
    for name, k in get_kwargs_dics(stop).items():
        amd.add(name, **k)

#    for i in range(2):         
#        kwargs[i].update({'par_xopt':{'node':{'CFp':{'rate': None}}}, 
#                          'par_key_ftar':[['node','FS','target_rate']], 
#                          'fun_ftar':['get_mean_rate'], 
#                          'par_key_input':['node','CFp'],
#                          'args_ftar':[['FS']],
#                          'kwargs_ftar':[{}]})
#
#    kwargs[0]['par_xopt']['node']['CFp']['rate']=970.0
#    kwargs[1]['par_xopt']['node']['CFp']['rate']=870.0
#    
#    
#    
#    setup_models=[['$FSN_{low}$',         'FS_low-FS-all-dop',     kwargs[0]],
#                  ['$FSN_{low-no-dop}$',  'FS_low-FS-all-no_dop',  kwargs[1]],
#                  ['$FSN_{high}$',        'FS_high-FS-all-dop',    kwargs[2]],
#                  ['$FSN_{high-no-dop}$', 'FS_high-FS-all-no_dop', kwargs[3]],
#                  ]
#    
#    labels_models=[sl[0] for sl in setup_models]
#
#    stop=21000.0
#    lables_fmin=[[labels_models[0]], [labels_models[2]]]
#    
#    lesion_setup={'all':[]}

 #   amd=Activity_model_dic(lesion_setup, setup_models)
    amd.simulate_input_output(  [0]*4, range(4), range(400,1050,100), 
                                stim_time=1000.0)   
    fig, ax_list=plot_settings.get_figure(n_rows=1, n_cols=1, 
                                              w=1000.0, h=800.0, fontsize=12)     
    colors=['g','g','b', 'b', 'c', 'k']*2
    coords=[[0.05, 0.9-i*0.1] for i in range(len(colors))]
    linestyles=['-','--','-','--','-','--','-','--',]  
    linestyles_hist=['solid', 'dashed','solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']
    amd.plot_input_output(ax_list[0], labels_dic.values(), colors, coords, linestyles)


    
#    kwargs={'par_rep': {'simu':{'threads':4}}, 'rand_setup':{'n':100, 'rand_params':['C_m', 'V_th']}
#    amd_fmin=amd.label_slice(lables_fmin).update_kwargs(kwargs)
#    par_var_dic=Fmin_dic(am_fmin)
#     
#    
#    amd.find_params([0]*2, labels_fmin)
#    rand_setup={'rand_params': ['C_m','V_th'], 'n':100}   
#    amd.simulate_variable_population([0]*2, labels_fmin, rand_setup)    
#    amd.rheobase_variable_population([0]*1, labels_models, rand_setup)
#    
#    amd.show2x2(labels_models, labels_fmin)
    pylab.show()
    
if __name__ == "__main__":
    main()     
    
    
