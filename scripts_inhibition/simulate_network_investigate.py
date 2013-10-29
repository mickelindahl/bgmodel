'''
Created on Aug 9, 2013

@author: lindahlm
'''

from toolbox.network_construction import Inhibition_base
import pylab
from toolbox.network_handling import Network_models_dic, Network_model 
from toolbox.default_params import Pertubation_list as pl

from toolbox.default_params import Par
from copy import deepcopy




def get_setup():
    
    l=[]
    l+=[['size-'+str(val), pl(['netw.size', val , '*'])] for val in [0.5, 1.0, 1.5]] 
    l+=[['MSNr-'+str(val), pl(['node.C2.rate', val, '*'])] for val in [1.3, 1.2, 1.1, 0.9, 0.8]] 
    l+=[['GAr-'+str(val), pl(['node.EA.rate', val, '*'])]  for val in [0.8, 0.6, 0.4, 0.2]] 
    l+=[['GISTd-'+str(0.5), pl(['nest.GI_ST_gaba.delay', 0.5, '*'])]]    # from GPe type I to STN  
    l+=[['STGId-'+str(0.5), pl(['nest.ST_GA_ampa.delay', 0.5, '*'])]]     # from STN to GPe type I and A  
    l+=[['Bothd-'+str(0.5), pl([['nest.GI_ST_gaba.delay',0.5, '*'], ['nest.ST_GA_ampa.delay',0.5, '*']])]]
    l+=[['V_th-'+str(0.5),pl(['netw.V_th_sigma',0.5, '*'])]]
    l+=[['GAfan-'+str(val),pl(['netw.prop_fan_in_GPE_A', val, '*' ])] for val in [2, 4, 6]]
    l+=[['GArV-'+str(0.5),pl([['netw.V_th_sigma',0.5, '*'], ['nest.GI_ST_gaba.delay', 0.5, '*']])]]
    
        
    return l

def check_setup():
    par=Par()
    setup=get_setup()

    for s in setup:
        dic=deepcopy(par.dic)
        s[1].update(dic, display=True)

def main():
    pds_setup=[256, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}]
    cohere_setup=[256, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}, 40]
    pds_models=['GP', 'GA', 'GI', 'ST', 'SN']
    cohere_relations=['GA_GA', 'GI_GI', 'GA_GI','ST_GA', 'ST_GA']
    plot_models=pds_models
    plot_relations=cohere_relations
        
    record_from_models=['M1', 'M2', 'FS', 'GA', 'GI', 'ST', 'SN']
    start, stop=1000.0, 21000.0
        
    setup_list=[]
    size=40000.0
    start=1000.0
    stop=41000.0
    use_class=Inhibition_base
    
    check_setup()
    for s in get_setup():
        kwargs={}
        kwargs['perturbation']=s[1]
        setup_list.append([s[0]+'-dop', 'dop', size, start, stop, use_class, kwargs])
        setup_list.append([s[0]+'-no_dop', 'no_dop', size, start, stop, use_class, kwargs])
    labels=[sl[0] for sl in setup_list]
    
    lesion_setup={}
    nms=Network_models_dic(16,  setup_list, Network_model)
    nms.simulate_example([0]*len(labels), labels, record_from_models)
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
    
    