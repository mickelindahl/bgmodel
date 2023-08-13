'''
Created on Aug 9, 2013

@author: lindahlm
'''

from core.network_construction import Inhibition_base
import pylab
from core.network_handling import Network_models_dic, Network_model 
from core.default_params import Perturbation_list as pl

from core.default_params import Par
from copy import deepcopy


def perturbations():
    
    l=[]
    l+=[pl('Size-'+str(val), ['netw.size',  val, '*']) for val in [0.5, 1.0, 1.5]] 
    l+=[pl('M2r-' +str(val), ['node.C2.rate', val, '*']) for val in [1.3, 1.2, 1.1, 0.9, 0.8]] 
    l+=[pl('GAr-' +str(val), ['node.EA.rate', val, '*']) for val in [0.8, 0.6, 0.4, 0.2]] 
    l+=[pl('GISTd-0.5', ['nest.GI_ST_gaba.delay', 0.5, '*'])]    # from GPe type I to STN  
    l+=[pl('STGId-0.5', ['nest.ST_GA_ampa.delay', 0.5, '*'])]     # from STN to GPe type I and A  
    l+=[pl('Bothd-0.5', [['nest.GI_ST_gaba.delay',0.5, '*'], ['nest.ST_GA_ampa.delay',0.5, '*']])]
    l+=[pl('V_th-0.5', ['netw.V_th_sigma',0.5, '*'])]
    l+=[pl('GAfan-'+str(val), ['netw.prop_fan_in_GPE_A', val, '*' ]) for val in [2, 4, 6]]
    l+=[pl('GArV-0.5', [['netw.V_th_sigma',0.5, '*'], ['nest.GI_ST_gaba.delay', 0.5, '*']])]
        
    return l

def pert_MS_subsampling(sub_sampling):
    p=pl('MS-sub-samp', [['nest.M1_SN_gaba.weight',  sub_sampling, '*'],
                             ['nest.M2_GI_gaba.weight',   sub_sampling, '*'],
                             ['nest.M1_M1_gaba.weight',   sub_sampling, '*'],
                             ['nest.M1_M2_gaba.weight',   sub_sampling, '*'],
                             ['nest.M2_M1_gaba.weight',   sub_sampling, '*'],
                             ['nest.M2_M2_gaba.weight',   sub_sampling, '*']])  
    return p
def check_perturbations(setup, par):


    for s in setup:
        dic=deepcopy(par.dic)
        s.update(dic, display=True)

def main():
    pds_setup=[256, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}]
    cohere_setup=[256, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}, 40]
    pds_models=['GP', 'GA', 'GI', 'ST', 'SN']
    cohere_relations=['GA_GA', 'GI_GI', 'GA_GI','ST_GA', 'ST_GA']
    plot_models=pds_models
    plot_relations=cohere_relations
        
    record_from_models=['M1', 'M2', 'FS', 'GA', 'GI', 'ST', 'SN']
    stop=31000.0 
    sub_sampling=10.0
    kwargs = {'class_network_construction':Inhibition_base, 
              'kwargs_network':{'save_conn':False, 'verbose':True}, 
              'par_rep':{'simu':{'threads':6, 'sd_params':{    'record_to': 'file'},
                                 'print_time':True, 'start_rec':1000.0, 
                                 'stop_rec':stop, 'sim_time':stop},
                         'netw':{'size':20000.0/sub_sampling, 'sub_sampling':{'M1':sub_sampling, 'M2':sub_sampling}}}}        

    pert0= pert_MS_subsampling(sub_sampling)
    setup_list=[]
    #check_perturbations()
    for s in perturbations():
        s.append(pert0)
        check_perturbations([s],Par())
        kwargs['perturbation']=s
        kwargs['par_rep']['netw'].update({'tata_dop':0.8})      
        setup_list.append([s.name+'-dop',   deepcopy(kwargs)])
        kwargs['par_rep']['netw'].update({'tata_dop':0.0})
        setup_list.append([s.name+'-no_dop', deepcopy(kwargs)])
    
    labels=[sl[0] for sl in setup_list]
    
    nms=Network_models_dic(setup_list, Network_model)
    nms.simulate([0]*len(labels), labels, record_from_models)
    nms.signal_pds([0]*len(labels), labels, pds_models, pds_setup)
    nms.signal_coherence([0]*len(labels), labels, cohere_relations, cohere_setup)
    
    figs=[]
    #figs.append(nms.show_exclude_rasters(labels[0:8:2], plot_models, plot_relations, xlim=[5000,6000]))
    #figs.append(nms.show_exclude_rasters(labels[1:8:2], plot_models, plot_relations, xlim=[5000,6000]))
    #figs.append(nms.show_exclude_rasters(labels[8:16:2], plot_models, plot_relations, xlim=[5000,6000]))
    #figs.append(nms.show_exclude_rasters(labels[9:16:2], plot_models, plot_relations, xlim=[5000,6000]))
    #figs.append(nms.show_exclude_rasters(labels[16:22:2], plot_models, plot_relations, xlim=[5000,6000]))
    #figs.append(nms.show_exclude_rasters(labels[17:22:2], plot_models, plot_relations, xlim=[5000,6000]))    
    #figs.append(nms.show_exclude_rasters(labels[20:26:2], plot_models, plot_relations, xlim=[5000,6000]))
    #figs.append(nms.show_exclude_rasters(labels[21:26:2], plot_models, plot_relations, xlim=[5000,6000]))    
    #fig2=nms.show_exclude_rasters(labels[4:6], plot_models, plot_relations)
    #fig3=nms.show_exclude_rasters(labels[6:8], plot_models, plot_relations)
    
    #fig=nms.show_compact(labels[2:4]+labels[16:18]+labels[24:28], plot_models, plot_relations)
    fig=nms.show_compact(labels, plot_models, plot_relations)
    pylab.show()
    
    
    #fig.savefig( nms.path_pictures +'.svg', format = 'svg') 
    #fig2.savefig( nms.path_pictures +'.svg', format = 'svg') 
    #fig3.savefig( nms.path_pictures +'.svg', format = 'svg') 
    #pylab.show()     
      
if __name__ == "__main__":
    main()    
    
    