




from toolbox import my_nest
from toolbox import misc, data_to_disk
from toolbox.my_population import MyPoissonInput, MyNetworkNode
from toolbox.network import default_params
from toolbox import plot_settings as ps
from toolbox.parallelization import comm


import sys
import pylab
import pprint
import shutil
pp=pprint.pprint

HOME=default_params.HOME

path, sli_path=my_nest.get_default_module_paths(HOME)
my_nest.install_module(path, sli_path, model_to_exist='izhik_cond_exp' )


def default_kwargs_net(n, n_sets=1):
    
    sets=[misc.my_slice(i, n,n_sets) for i in range(n_sets)]
    return  {'n':n, 
             'model':'my_aeif_cond_exp', 
             'mm':{'active':True,
                   'params':{'interval':1.0,
                             'to_memory':True, 
                             'to_file':False,
                             'record_from':['V_m']}},
             'sd':{'active':True,
                   'params':{'to_memory':True, 
                             'to_file':False}},
             'sets':sets,
             'rate':10.0}


def default_kwargs_inp(n):
    
    sets=[misc.my_slice(i, n,1) for i in range(1)]
    return  {'n':n, 
             'model':'poisson_generator', 
             'sets':sets,
             'rate':1000.0}

def default_spike_setup(n, stop):
    d={'rates':[1000.0],
       'times':[1.0],
       't_stop':stop,
       'idx':range(n)}
    return d

def sim_group(n_net, n_inp, sim_time):
    g=MyNetworkNode(*['node'], **default_kwargs_net(n_net))
    i=MyPoissonInput(*['ino'], **default_kwargs_inp(n_inp))
    i.set_spike_times(**default_spike_setup(i.n, sim_time))
    
    df=my_nest.GetDefaults(g.model)['receptor_types']
    receptor='AMPA_1'
    
    my_nest.Connect(i.ids*len(g.ids), 
                    g.ids, 
                    params={'delay':1.0,
                            'weight':10.0,
                             'receptor_type':df[receptor],}, 
                    model='static_synapse')
    my_nest.Simulate(sim_time)
    return g   

import os
if __name__=='__main__':
    t=10000.0
    currdir=os.getcwd()
    home=default_params.HOME+'/results/unittest/test_milner/simulation'
    script_name=__file__.split('/')[-1][0:-3]
    if len(sys.argv)>1:
        datadir, datadir_sd= sys.argv[1:]
        
    else:
        datadir=home+'/nest'
        datadir_sd=home+'/spike_dic'

    data_path='/'.join(datadir.split('/')[0:-1])
    for path in [datadir, datadir_sd]:
        if os.path.isdir(path):
            if comm.rank()==0:
                shutil.rmtree(path)
    
    for path in [datadir_sd+'.pkl', datadir_sd+'.svg']:
        if os.path.isfile(path):
            if comm.rank()==0:
                os.remove(path)
    
    if comm.rank()==0:
        data_to_disk.mkdir(datadir)
    
    comm.barrier()
    print comm.obj
    
    my_nest.SetKernelStatus({'data_path':datadir,
                             'overwrite_files': True})
    
    g=sim_group(10, 1, t)
    ss=g.get_spike_signal()
    fr=ss.get_firing_rate()
    mr=ss.get_mean_rate()
    
    
    
    sd=data_to_disk.Storage_dic.load(datadir_sd)
    sd_figs=data_to_disk.Storage_dic.load(datadir_sd)
    sd.save_dic({'fr':fr,
                 'mr':mr})
    
    fig, axs=ps.get_figure(n_rows=1, n_cols=1, 
                            w=1000.0, h=800.0, fontsize=10)  
    fr.plot(axs[0])
    sd_figs.save_fig(fig)
    print mr.y
    
#     print mr['y']
#     fr.plot(pylab.subplot(111))
#     pylab.show()
