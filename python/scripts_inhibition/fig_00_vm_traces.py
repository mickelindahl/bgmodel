'''
Created on Nov 1, 2015

@author: mikael
'''
from core import plot_settings as ps
from core import data_to_disk
from core.my_signals import MyVmList
from scripts_inhibition.base_simulate import save_figures
import matplotlib.gridspec as gridspec

import pylab
import pprint
pp=pprint.pprint
# filename=('/home/mikael/results/papers/inhibition/network/supermicro/fig_01_and_02_sim_beta_vm/'
#           +'script_0000_rEI_1700.0_rEA_200.0_rCS_250.0_rES_1800.0_rM2_740.0-amp_0.02_0.975_stn_3.0/Net_0/vm_traces.pkl')


filename=('/home/mikael/results/papers/inhibition/network/supermicro/fig_00_sim_beta_vm_traces/'
          +'script_0000_rEI_1700.0_rEA_200.0_rCS_250.0_rES_1800.0_rM2_740.0-amp_0.08_0.975_stn_3.0/Net_0/vm_traces.pkl')

filename_spk=('/home/mikael/results/papers/inhibition/network/supermicro/fig_00_sim_beta_vm_traces/'
          +'script_0000_rEI_1700.0_rEA_200.0_rCS_250.0_rES_1800.0_rM2_740.0-amp_0.08_0.975_stn_3.0/Net_0/Net_0-{}-spike_signal.pkl')

d=data_to_disk.pickle_load(filename)

for model in d.keys():
    spk=data_to_disk.pickle_load(filename_spk.format(model))
    spk=spk.wrap.m[0,0]
    signal=zip(*[d[model]['senders'],d[model]['V_m']])
    vmlist=MyVmList(signal, sorted(list(set(d[model]['senders']))), 1,
              min(d[model]['times'])-1, max(d[model]['times']))
    
    dvt=vmlist.Factory_voltage_traces(**{'spike_signal':spk})
    d[model]=dvt

pp(d)


def gs_builder(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.8 ), 
              hspace=kwargs.get('hspace', 0.4))
# 
#     iterator = ([[i, 1] for i in range(1,6)]+
#                 [[i, 2] for i in range(1,6)]+
#                 [[i, 3] for i in range(1,6)]+
#                 [[i, 4] for i in range(1,6)]+
#                 [[i, 5] for i in range(1,6)])
#     
    iterator = ([[slice(0,1), i] for i in range(0,1)]+
                [[slice(1,2), i] for i in range(0,1)]+
                [[slice(2,3), i] for i in range(0,1)]+
                [[slice(3,4), i] for i in range(0,1)]+
                [[slice(4,5), i] for i in range(0,1)]+
                [[slice(5,6), i] for i in range(0,1)]+
                [[slice(6,7), i] for i in range(0,1)]

#                 [[slice(9,10), i] for i in range(0,7)]
#                 [[slice(6,7), i] for i in range(0,7)]
#                 [[slice(7,8), i] for i in range(0,4)]
#                 [[4, i] for i in range(1,6)]+
#                 [[5, i] for i in range(1,6)]
                )
    
    return iterator, gs,
scale=1
figs=[]
for j in range(10):
    fig, axs=ps.get_figure2(n_rows=7, 
                                n_cols=1,
                                w=72/2.54*11.*scale,
                                h=72/2.54*11.*scale*1.4,  
                                fontsize=7*scale,
                                title_fontsize=7*scale,
                                gs_builder=gs_builder) 
    
    for i, model in enumerate(sorted(d.keys())):
        
        d[model].plot(axs[i], id_list=[j],
                      **{ 'color':'k', 'linewidth':0.5*scale})
        axs[i].set_title(model)
        if model in ['M1', 'M2']:
            pass
        else:  
            axs[i].set_xlim([0,5000.])
        axs[i].set_ylabel('Potential (mV)')
    figs.append(fig)
save_figures(figs, __file__.split('/')[-1][0:-3]+'/data',dpi=200)
# pylab.show()
