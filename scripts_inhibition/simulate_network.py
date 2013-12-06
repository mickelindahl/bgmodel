'''
Created on Jun 27, 2013

@author: lindahlm
'''
from toolbox.network_construction import Inhibition_base
from toolbox.network_handling import Network_models_dic, Network_model 
from copy import deepcopy
from toolbox.default_params import Perturbation_list as pl
import pylab  

def main():
    stop=11000.0
    sub_sampling=10.0
    kwargs = {'class_network_construction':Inhibition_base, 
              'kwargs_network':{'save_conn':False, 'verbose':True}, 
              'par_rep':{'simu':{'threads':4, 'sd_params':{'to_file':True, 'to_memory':False},
                                 'print_time':True, 'start_rec':1000.0, 
                                 'stop_rec':stop, 'sim_time':stop},
                             'netw':{'size':10000.0/sub_sampling, 'sub_sampling':{'M1':sub_sampling, 
                                                                                  'M2':sub_sampling}}}}          
    
    pert=pl('MS-sub-samp', [['nest.M1_GI_gaba.weight',  sub_sampling, '*'],
                            ['nest.M2_SN_gaba.weight',  sub_sampling, '*'],
                            ['nest.M1_M1_gaba.weight',  sub_sampling, '*'],
                            ['nest.M1_M2_gaba.weight',  sub_sampling, '*'],
                            ['nest.M2_M1_gaba.weight',  sub_sampling, '*'],
                            ['nest.M2_M2_gaba.weight',  sub_sampling, '*']])
    
    record_from_models=['M1', 'M2', 'FS', 'GA', 'GI', 'ST', 'SN']
    labels=['Control', 'No_dopamine']
    dopamine=[0.8, 0.0]
    
    setup_list=[]
    for l, d in zip(*[labels, dopamine]): 
        kwargs['par_rep']['netw'].update({'tata_dop':d})      
        kwargs['perturbations']=pert
        setup_list.append([l, deepcopy(kwargs)])
    
    
    pds_setup    =[256, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}]
    cohere_setup =[256, 40., 'gaussian',{'std_ms':20,'fs':1000.0}, 20]
    pds_models=record_from_models+['GP']
    cohere_relations=['GP_GP', 'GA_GA', 'GA_GI','GI_GI','ST_GP',
                      'ST_GA', 'ST_GI']
    plot_models=pds_models[0:5]
    plot_relations=cohere_relations[0:5]
    
    nms=Network_models_dic(setup_list, Network_model)
    nms.simulate([1]*2, labels, record_from_models)
    nms.signal_pds([0]*2, labels, pds_models, pds_setup)
    nms.signal_coherence([0]*2, labels, cohere_relations, cohere_setup)
    
    #fig=nms.show_signal_processing_example( labels[0], 'GPE_I')
    fig=nms.show(labels, plot_models, plot_relations)
    fig=nms.show_compact(labels, plot_models, plot_relations)
    #fig.savefig( nms.path_pictures +'example_sp'+'.svg', format = 'svg') 
    pylab.show()
    #fig.savefig( nms.path_pictures +'.svg', format = 'svg') 
    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

   


    

    
