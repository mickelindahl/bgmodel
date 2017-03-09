'''
Created on Apr 6, 2015

@author: mikael
'''

from scripts_inhibition import base_effect_conns
from core.network.manager import get_storage
from scripts_inhibition.base_simulate import get_file_name, save_figures
from core import misc
from matplotlib import ticker

import core.plot_settings as ps
import matplotlib.gridspec as gridspec
import pylab
import numpy
import pprint

pp=pprint.pprint

from_disk=0


scale=1
path=[
#     ('/home/mikael/results/papers/inhibition/network/'
#            +'milner/eNeuro_rev_TA_TI_FS_sim_beta/'),
    ('/home/mikael/results/papers/inhibition/network/'
           +'milner/eNeuro_rev_TA_TI_FS_v6_sim_sw_only_M2/')
]


def gs_builder(*args, **kwargs):
    n_rows=kwargs.get('n_rows',5)
    n_cols=kwargs.get('n_cols',3)
#     order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 1.2 ), 
              hspace=kwargs.get('hspace', 0.65 ))

    iterator=[]
    for i in range(4):
        for j in range(2):
            iterator+= [[slice(i,i+1), slice(j,j+1) ]]
    
    return iterator, gs  

def create_name(file_name):
#     print file_name.split('/')[-1]
    if file_name.split('/')[-1] in ['std', 'jobbs', 'params']:
        return file_name.split('/')[-1]
    else:
        print file_name.split('/')[-1][7:11]+file_name.split('/')[-2][18:]
        v=file_name.split('/')[-1][7:11]+file_name.split('/')[-2][18:]
        return v

def conn_setup_beta():
    d= {'key_no_pert':'0000',#control_sim',
        'name_maker':create_name,
        'add_midpoint':False,
        'psd':{'NFFT':128, 
               'fs':256., 
               'noverlap':128/2},
        'oi_min':15.,
        'oi_max':25.,
        'oi_upper':1000.,
        'oi_fs':256,
                   'keep':['data']}
    return d

def conn_setup_sw():
    d= {'key_no_pert':'0000',#control_sim',
        'name_maker':create_name,
        'add_midpoint':False,
        'psd':{'NFFT':128*4, 
               'fs':256.*4, 
               'noverlap':128*4/2},
        'oi_min':.5,
        'oi_max':1.5,
        'oi_upper':1000.,
        'oi_fs':256*4,
        'keep':['data'],
        'compute_performance_name_and_x':lambda x: [x[0],1]
                   }
    return d

fig, axs=ps.get_figure2(n_rows=4,
                         n_cols=3,  
                         w=int(72/2.54*11.6*(1+1./2+0.2))*scale,
                         h=int(0.85*72/2.54*11.6*(1+1./2))*scale,
#                             w=k.get('w',500), 
#                             h=k.get('h',900), 
                        linewidth=1,
                        fontsize=7*scale,
                        title_fontsize=7*scale,
                        gs_builder=gs_builder) 

models=['M1', 'M2', 'FS', 'GA', 'GF', 'GI', 'GP', 'ST','SN',
        'GI_MS', 
        'GA_MS', 
        'GF_MS', 
        'GI_FS', 
        'GA_FS', 
        'GF_FS']

# models=[m for m in models if not ( m in exclude)]

nets=['Net_0', 'Net_1']
attrs=[
#        'spike_signal',
        'firing_rate', 
#         'mean_rate',
#         'mean_coherence', 
       'phases_diff_with_cohere',
#         'psd'
       ]

# attrs=['phases_diff_with_cohere']
# models=['M1', 'M2', 'SN']
# paths=[]



organize={'TATIFS':slice(3*7,4*7),
         }


import os
list_names_tmp=[]
for pa in path:
    list_names_tmp.append([])
    for f in os.listdir(pa):
        if f.split('/')[-1] in ['params', 'jobbs','std']:
            continue
        list_names_tmp[-1].append(f.split('/')[-1].split('_')[1:])

list_names=sorted(list_names_tmp[0], key=lambda x:x[0])
# list_names_sw=sorted(list_names_tmp[1], key=lambda x:x[0])
script_name=(__file__.split('/')[-1][0:-3] +'/data')
file_name = get_file_name(script_name)
 
attr_add=['mse_rel_control_fr', 'mse_rel_control_mc',
          'mse_rel_control_pdwc', 'mse_rel_control_mcm',
          'mse_rel_control_oi', 'mse_rel_control_si',
          'mse_rel_control_psd']
 
#     exclude+=['MS_MS', 'FS_MS', 'MS']
sd = get_storage(file_name, '')
d = base_effect_conns.get_data(models, nets, attrs, path[0], 
                          from_disk, attr_add, sd,
#                         **conn_setup_sw()
                        **conn_setup_sw()
                          )
# sd = get_storage(file_name+'_sw', '')
# d_sw = base_effect_conns.get_data(models, nets, attrs, path[1], 
#                           from_disk, attr_add, sd,
#                           **conn_setup_sw())
# # pp(d.keys())
for i, ln in enumerate(list_names):print i, ln[10:]
# pp(list_names)
'''
['d_gradients_lesion', 
'd_raw_control', 
'd_raw_lesion', 
'd_gradients_control']

[ 
'synchrony_index', 
'oscillation_index',
'mean_coherence_max', 
'firing_rate', '
'mean_coherence', 

['labelsx', 'labelsy', 'z', 'labelsx_meta']

d['d_gradients_lesion']['oscillation_index'].keys()
d['d_gradients_lesion']['oscillation_index']['labelsy']

'''
     
#########################################
# GF/GA to FS effect on phase shift GI_FS 
#########################################
ax=axs[0]
 
l1, l2=[], []
for key in sorted(d['data'].keys()):
    
    if not 'Net_0' in d['data'][key].keys():
        continue
    
    fr=d['data'][key]['Net_0']['FS']['firing_rates']
    fr.x=fr.x[64:]
    fr.y=fr.y[64:]
    
    ah=fr.get_activity_histogram(**{'period':256,'bins':16})
#     ah.plot(pylab.subplot(121))
    
    fr=d['data'][key]['Net_0']['GI']['firing_rates']
    fr.x=fr.x[64:]
    fr.y=fr.y[64:]
    
    ah_GP=fr.get_activity_histogram(**{'period':256,'bins':16})
#     ah_GP.plot(pylab.subplot(122))
#     
#     pylab.show()
#     v=[ah.y, ah_GP.y]
    

#     v=d['data'][key]['Net_0']['GI_FS']['phases_diff_with_cohere']

    l1.append(ah.y)
    l2.append(ah_GP.y)
    
l=numpy.array(l1[organize['TATIFS']])

pp(l)

# FSN
#####
pp(l)
colors=misc.make_N_colors('copper', len(l))

ln=list_names[organize['TATIFS']]
pp(ln)
ln=[a[2]+','+a[4] for a in ln]
pp(ln)
  
for i, trace in enumerate(l):
    x=numpy.linspace(0,1000,len(trace))
#     norm=sum(trace)*(x[-1]-x[0])/len(x)
#     norm=numpy.mean(x)
    ax.plot(x, (numpy.array(trace)-trace[0]), color=colors[i])


ax.set_xlim([0,1000])
  
 
sm = pylab.cm.ScalarMappable(cmap='copper', 
                             norm=pylab.normalize(vmin=2.5, vmax=20))
sm._A = []
  
box = ax.get_position()
pos=[box.x0+1.03*box.width, box.y0+box.height*0.1,
     0.01,  box.height*0.8]
axColor=pylab.axes(pos)
cbar=pylab.colorbar(sm, cax=axColor)
tick_locator = ticker.MaxNLocator(nbins=7)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.tick_params( length=1, )
cbar.ax.set_yticklabels(ln, fontsize=5*scale)  
  
ax.text(1.3, 0.5,  r'$w_{perturb}^{GA\to FS}$ (ms)', 
        transform=ax.transAxes,  va='center', rotation=270) 
ax.set_xlabel(r'Angle (rad)')
ax.set_ylabel('Norm. count ST-TI')
ax.my_set_no_ticks(xticks=4)
ax.my_set_no_ticks(yticks=4)
# ax.set_ylim([0,0.31])


# GP
####

ax=axs[1]
l=l2[organize['TATIFS']]

pp(l)
colors=misc.make_N_colors('copper', len(l))

ln=list_names[organize['TATIFS']]
pp(ln)
ln=[a[2]+','+a[4] for a in ln]
pp(ln)
  
for i, trace in enumerate(l):
    x=numpy.linspace(0,1000,len(trace))
#     norm=sum(trace)*(x[-1]-x[0])/len(x)
#     norm=numpy.mean(x)
    ax.plot(x, (numpy.array(trace)-trace[0]), color=colors[i])


ax.set_xlim([0,1000])
  
 
sm = pylab.cm.ScalarMappable(cmap='copper', 
                             norm=pylab.normalize(vmin=2.5, vmax=20))
sm._A = []
  
box = ax.get_position()
pos=[box.x0+1.03*box.width, box.y0+box.height*0.1,
     0.01,  box.height*0.8]
axColor=pylab.axes(pos)
cbar=pylab.colorbar(sm, cax=axColor)
tick_locator = ticker.MaxNLocator(nbins=7)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.tick_params( length=1, )
cbar.ax.set_yticklabels(ln, fontsize=5*scale)  
  
ax.text(1.3, 0.5,  r'$w_{perturb}^{GA\to FS}$ (ms)', 
        transform=ax.transAxes,  va='center', rotation=270) 
ax.set_xlabel(r'Angle (rad)')
ax.set_ylabel('Norm. count ST-TI')
ax.my_set_no_ticks(xticks=4)
ax.my_set_no_ticks(yticks=4)



l2=l2[organize['TATIFS']]

save_figures([fig], script_name, dpi=400)

pylab.show()
