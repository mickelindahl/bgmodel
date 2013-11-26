'''
Created on Jun 27, 2013

@author: lindahlm
'''
from toolbox.network_construction import Inhibition_base
from toolbox.network_handling import Network_models_dic, Network_model 

import pylab  

def main():

    record_from_models=['M1', 'M2', 'FS', 'GA', 'GI', 'ST', 'SN']
    setup_list=[['Control',     'dop'],
                ['No_dopamine', 'no_dop']]
    
    #Inhibition_no_parrot
    #for setup in setup_list: setup.extend([10000., 1000.0, 11000.0, Inhibition_no_parrot, {}])
    for setup in setup_list: setup.extend([20000., 1000.0, 21000.0, Inhibition_base, {}])
    labels=[sl[0] for sl in setup_list]
    
    
    pds_setup    =[256, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}]
    cohere_setup =[256, 40., 'gaussian',{'std_ms':20,'fs':1000.0}, 20]
    pds_models=record_from_models+['GP']
    cohere_relations=['GP_GP', 'GA_GA', 'GA_GI','GI_GI','ST_GP',
                      'ST_GA', 'ST_GI']
    plot_models=pds_models[0:5]
    plot_relations=cohere_relations[0:5]
    
    #kwargs_simulation={'sd_params':{'to_file':True, 'to_memory':False}}
    kwargs_simulation={'sd_params':{'to_file':False, 'to_memory':True}}
    
    nms=Network_models_dic(4, setup_list, Network_model)
    nms.simulate([0]*2, labels, record_from_models, **kwargs_simulation)
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

   


    

    
