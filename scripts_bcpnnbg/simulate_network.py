'''
Created on Jun 27, 2013

@author: lindahlm
'''
from copy import deepcopy
from toolbox.network_construction import Bcpnn_h0, Bcpnn_h1
from toolbox.network_handling import Network_models_dic, Network_model 
from toolbox import data_to_disk

import pylab  

def main():

    record_from_models=[['CO', 'M1', 'M2', 'FS', 'GA', 'GI', 'ST', 'SN'],
                        ['CO', 'M1', 'M2', 'F1', 'F2','GA', 'GI', 'ST', 'SN']]
    labels=['Control-h0', 'Control-h1']
    start=1000.0
    stop=1000.0+100.*10.
    kwargs = {'kwargs_network':{'save_conn':False, 'verbose':True}, 
              'par_rep':{'simu':{'threads':2, 'sd_params':{'to_file':True, 'to_memory':False},
                                 'print_time':True, 'start_rec':start, 
                                 'stop_rec':stop, 'sim_time':stop},
                         'netw':{'size':5000.0, 'tata_dop':0.8}}}   
    use_class=[Bcpnn_h0, Bcpnn_h1]
    #Inhibition_no_parrot
    #for setup in setup_list: setup.extend([10000., 1000.0, 11000.0, Inhibition_no_parrot, {}])
    setup_list=[]
    for l, uc in zip(labels, use_class): 
        kwargs['class_network_construction']=uc
        setup_list.append([l, deepcopy(kwargs)])
        
    
    nms=Network_models_dic(setup_list, Network_model)
    nms.simulate([0]*2, labels, record_from_models)

    
    plot_models=[['CO', 'M1', 'M2','FS', 'GI', 'SN'],
                 ['M1', 'M2', 'F1', 'F2', 'GI', 'SN']]
    plot_lables_models=[['Cortex', '$MSN_{D1}$', '$MSN_{D2}$','FSN', '$GPe_{Type I}$', 'SNr'],
                        ['$MSN_{D1}$','$MSN_{D2}$', '$FSN_{1}$', '$FSN_{2}$','$GPe_{Type I}$', 'SNr']]
    plot_lables_prefix_models=[['State', 'Action', 'Action','Action', 'Action', 'Action'],
                        ['Action', 'Action', 'Action','Action', 'Action', 'Action']]    
    figs=nms.show_bcpnn(labels, plot_models, xlim=[start, stop], plot_lables_models=plot_lables_models, plot_lables_prefix_models=plot_lables_prefix_models)
    
    pylab.show()
    i=0
    
    for fig in figs:
        import os
        if not os.path.isdir( nms.path_pictures):
            data_to_disk.mkdir( nms.path_pictures)
        fig.savefig( nms.path_pictures+ '-fig'+str(i)+'.svg', format = 'svg') 
        i+=1
    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

   


    

    
