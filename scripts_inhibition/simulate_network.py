'''
Created on Jun 27, 2013

@author: lindahlm
'''
import pylab
import toolbox.plot_settings as ps

from toolbox import misc
from toolbox.data_to_disk import Storage_dic
from toolbox.network import manager
from toolbox.network.manager import compute, run, save, load
from toolbox.network.manager import Builder_network as Builder
from toolbox.network.manager import Director
import pprint
pp=pprint.pprint
    


def show_fr(d):
    _, axs=ps.get_figure(n_rows=7, n_cols=1, w=1000.0, h=800.0, fontsize=10)  
    labels=['Dop','No dop']
    colors=['b', 'g']
    linestyles=['-','-']
    
    j=0
    for k in sorted(d.keys()):
        v=d[k]
#         axs[0].set_title(k)
        for model, i in [['M1',0], ['M2', 1], ['FS',2],['GA',3], ['GI',4],
                         ['ST',5],['SN',6]]:
            v[model]['firing_rate'].plot(ax=axs[i], 
                                         **{'label':model+' '+labels[j],
                                            'linestyle':linestyles[j],
                                            'color':colors[j]})
        j+=1 
            
def show_hr(d):
    _, axs=ps.get_figure(n_rows=7, n_cols=1, w=1000.0, h=800.0, fontsize=10)   
    labels=['Dop','No dop']
    colors=['b', 'g']
    linestyles=['solid','solid']
    j=0
    
    for k in sorted(d.keys()):
        v=d[k]
#         axs[0].set_title(k)
        for model, i in [['M1',0], ['M2', 1], ['FS',2],['GI',3], ['GA',4],
                         ['ST',5],['SN',6]]:
            st=v[model]['spike_statistic']
            st.rates={'mean':round(st.rates['mean'],2),
                      'std':round(st.rates['std'],2),
                      'CV':round(st.rates['CV'],2)}
            h=v[model]['mean_rates'].hist(ax=axs[i],
                                          **{'label':(model+' '
                                                      +labels[j]+' '
                                                      +str(st.rates)),
                                             'histtype':'step',
                                             'bins':20,
                                              'linestyle':linestyles[j],
                                              'color':colors[j]}) 
            
            ylim=list(axs[i].get_ylim())
            ylim[0]=0.0
            axs[i].set_ylim(ylim)
        j+=1 

def get_kwargs_builder():
    return {'print_time':True, 
            'threads':4, 
            'save_conn':{'overwrite':True},
            'sim_time':5000.0, 
            'sim_stop':5000.0, 
            'size':10000.0, 
            'start_rec':0.0, 
            'sub_sampling':1}

def get_kwargs_engine():
    return {'verbose':True}

def get_networks():
    return manager.get_networks(Builder, 
                                get_kwargs_builder(), 
                                get_kwargs_engine())




def main():
    k=get_kwargs_builder()
    
    from os.path import expanduser
    home = expanduser("~")

    attr=[ 'firing_rate', 
           'mean_rates', 
           'spike_statistic']  
    
    kwargs_dic={'mean_rates': {'t_stop':k['start_rec']},
                'spike_statistic': {'t_stop':k['start_rec']},}
    file_name=(home+ '/results/papers/inhibition/network/'
               +__file__.split('/')[-1][0:-3])
    
    models=['M1', 'M2', 'FS', 'GI', 'GA', 'ST', 'SN']
    
    info, nets = get_networks()

    sd=Storage_dic.load(file_name)
    sd.add_info(info)
    sd.garbage_collect()
    
    d={}
    for net, from_disk in zip(nets, [1]*2):
        if not from_disk:
            dd = run(net)  
            dd = compute(dd, models,  attr, **kwargs_dic)      
            save(sd, dd)
        elif from_disk:
            filt=[net.get_name()]+models+attr
            dd=load(sd, *filt)
        d=misc.dict_update(d, dd)
                     
    
    show_fr(d)
    show_hr(d)

    pylab.show()
    
#     stop=11000.0
#     sub_sampling=10.0
#     kwargs = {'class_network_construction':Inhibition_base, 
#               'kwargs_network':{'save_conn':False, 'verbose':True}, 
#               'par_rep':{'simu':{'threads':4, 'sd_params':{'to_file':True, 'to_memory':False},
#                                  'print_time':True, 'start_rec':1000.0, 
#                                  'stop_rec':stop, 'sim_time':stop},
#                              'netw':{'size':10000.0/sub_sampling, 'sub_sampling':{'M1':sub_sampling, 
#                                                                                   'M2':sub_sampling}}}}          
#     
#     pert=pl('MS-sub-samp', [['nest.M1_GI_gaba.weight',  sub_sampling, '*'],
#                             ['nest.M2_SN_gaba.weight',  sub_sampling, '*'],
#                             ['nest.M1_M1_gaba.weight',  sub_sampling, '*'],
#                             ['nest.M1_M2_gaba.weight',  sub_sampling, '*'],
#                             ['nest.M2_M1_gaba.weight',  sub_sampling, '*'],
#                             ['nest.M2_M2_gaba.weight',  sub_sampling, '*']])
#     
#     record_from_models=['M1', 'M2', 'FS', 'GA', 'GI', 'ST', 'SN']
#     labels=['Control', 'No_dopamine']
#     dopamine=[0.8, 0.0]
#     
#     setup_list=[]
#     for l, d in zip(*[labels, dopamine]): 
#         kwargs['par_rep']['netw'].update({'tata_dop':d})      
#         kwargs['perturbations']=pert
#         setup_list.append([l, deepcopy(kwargs)])
#     
#     
#     pds_setup    =[256, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}]
#     cohere_setup =[256, 40., 'gaussian',{'std_ms':20,'fs':1000.0}, 20]
#     pds_models=record_from_models+['GP']
#     cohere_relations=['GP_GP', 'GA_GA', 'GA_GI','GI_GI','ST_GP',
#                       'ST_GA', 'ST_GI']
#     plot_models=pds_models[0:5]
#     plot_relations=cohere_relations[0:5]
#     
#     nms=Network_models_dic(setup_list, Network_model)
#     nms.simulate([1]*2, labels, record_from_models)
#     nms.signal_pds([0]*2, labels, pds_models, pds_setup)
#     nms.signal_coherence([0]*2, labels, cohere_relations, cohere_setup)
#     
#     #fig=nms.show_signal_processing_example( labels[0], 'GPE_I')
#     fig=nms.show(labels, plot_models, plot_relations)
#     fig=nms.show_compact(labels, plot_models, plot_relations)
#     #fig.savefig( nms.path_pictures +'example_sp'+'.svg', format = 'svg') 
#     pylab.show()
#     #fig.savefig( nms.path_pictures +'.svg', format = 'svg') 
    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

   


    

    
