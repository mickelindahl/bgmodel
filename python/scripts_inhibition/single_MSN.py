'''
Created on Mar 19, 2014

@author: lindahlm
'''


from core.network import default_params
from core import my_population
import pprint
pp=pprint.pprint

import fig_01_and_02_pert as op


def simulate(nest_model, neuron_type):
    par=default_params.Inhibition(**{'perturbations':op.get()[0]})
    
    kw={'model':nest_model,
        'n':1,
        'mm':{"withgid": True, 
                      'record_to': 'memory'
                  'record_from':['V_m']
                  },
        'params':par.dic[neuron_type]}
    
    
    n=my_population.MyNetworkNode(**kw)
            model=kwargs.get('model', 'iaf_neuron')
            n=kwargs.get('n', 1)
            params=kwargs.get('params',{}) 
    inh=my_nest.Create('poisson_generator',n=1,  params={'rate':rate_inh})        
    exc=my_nest.Create('poisson_generator',n=1,  params={'rate':rate_exc})

    if nest_mode='my_aeif_cond_exp':
        rec=par.rec['aeif']
    else:
        rec=par.rec['aeif']
        
    my_nest.CreateModel('INH', 'static_synapse', {'GABAA_1_Tau_decay':10,
                                                  'GABAA_1_E_rev':0,
                                                  'receptor_type':rec['GABAA_1']})
    
    my_nest.CreateModel('EXC', 'static_synapse', {'AMPA_1_Tau_decay':10,
                                                  'AMPA_1_E_rev':0,
                                                  'receptor_type':rec['AMPA_1']})
        
    my_nest.Connect(inh,n.ids, 0.25, 1., model='INH')
    my_nest.Connect(ecx,n.ids, 0.25, 1., model='EXC')
    
    

def main():    
    IV=build_cases(**{'lesion':True, 'mm':True})
    IF=build_cases(**{'lesion':True})
    FF=build_cases(**{'lesion':False})
    
    curr_IV=range(-200,300,100)
    curr_IF=range(0,500,100)
    rate_FF=range(100,1500,100)
    _, axs=pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16)     
    
    plots('plot_IV_curve', IV, 1, **{'ax':axs[0],'curr':curr_IV, 'node':'FS'})
    plots('plot_IF_curve', IF, 1, **{'ax':axs[1],'curr':curr_IF, 'node':'FS'})
    plots('plot_FF_curve', FF, 1, **{'ax':axs[2],'rate':rate_FF, 'node':'FS',
                                     'input':'CFp'})    
    beautify(axs)
    pylab.show()
    
if __name__ == "__main__":
    main()  