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


from_disk=1
scale=1
path=[
    ('/home/mikael/results/papers/inhibition/network/'
           +'milner/eNeuro_fig_08_sim_control_beta/'),
      ('/home/mikael/results/papers/inhibition/network/'
             +'milner/eNeuro_fig_08_sim_control_sw/')]


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
                   'keep':['data']}
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

models=['M1', 'M2', 'FS', 'GA', 'GI', 'GP', 'ST','SN',
                   'GI_ST', 'GP_GP', 'GA_GA', 'GI_GA', 'GI_GI']

# models=[m for m in models if not ( m in exclude)]

nets=['Net_0', 'Net_1']
attrs=[
       'firing_rate', 
       'mean_coherence', 
       'phases_diff_with_cohere',
       'psd'
       ]

# attrs=['phases_diff_with_cohere']
# models=['M1', 'M2', 'SN']
# paths=[]



organize={'control':slice(0,1),
          'GAtau':slice(2,12),
          'CSdelay':slice(12,20),
          'M2GAfan':slice(20,25),
          'GIM2fan':slice(25,30),
          'STGA':slice(30,38),
          'GAGItoGA':slice(38,43),
          'MSMS':slice(43,50),
          'CTXSTRdelay':slice(50,58),
          'STGPdelay':slice(58,66)
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
list_names_sw=sorted(list_names_tmp[1], key=lambda x:x[0])
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
                          **conn_setup_beta())
sd = get_storage(file_name+'_sw', '')
d_sw = base_effect_conns.get_data(models, nets, attrs, path[1], 
                          from_disk, attr_add, sd,
                          **conn_setup_sw())
# pp(d.keys())
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
# MSMS
#########################################

ax=axs[7]
# v=d['data']
   
l=[]
for key in sorted(d_sw['data'].keys()):
    v=d_sw['data'][key]['Net_1']['GP_GP']['mean_coherence']
    l.append(v)
 
l=l[organize['MSMS']]
    
ln=list_names_sw[organize['MSMS']]
ln=[str(float(a[13])) for a in ln]
 
colors=misc.make_N_colors('copper', len(l))
   
for i, trace in enumerate(l):
    x=numpy.linspace(0,128,len(trace)) #half of sampleing frequency
    ax.plot(x, trace, color=colors[i])
ax.set_xlim([0,5])
ax.set_ylim([0,1])  
sm = pylab.cm.ScalarMappable(cmap='copper', 
                            norm=pylab.normalize(vmin=0, vmax=len(ln)-1)
                             )
sm._A = []
     
box = ax.get_position()
pos=[box.x0+1.03*box.width, box.y0+box.height*0.1,
     0.01, box.height*0.8]
axColor=pylab.axes(pos)
cbar=pylab.colorbar(sm, cax=axColor, ticks=range(len(ln)))
 
cbar.ax.tick_params( length=1, )
cbar.ax.set_yticklabels(ln, fontsize=5*scale)  
 
  
ax.text(1.4,  0.5, r'Multiplier $g_{MSN \to MSN}$', 
        transform=ax.transAxes, va='center', rotation=270) 
 
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('GPe-GPe Coherence')
ax.my_set_no_ticks(xticks=4)
ax.my_set_no_ticks(yticks=4)
ax.set_title('Lesion')
#########################################
# TA to MS tau snapse figure
#########################################
ax=axs[0]
  
dd=d['d_raw_control']['oscillation_index']['labelsy']
bol= numpy.array([l in ['M1', 'M2'] for l in dd])
v=d['d_raw_control']['oscillation_index']['y'][bol,organize['GAtau']]
  
  
ax.plot(range(5,55,5),v.transpose())
ax.set_xlim([0,60])
ax.set_xlabel(r'$\tau_{GPe_{TA}\to str}$ (ms)')
ax.set_ylabel('Oscillation index')
ax.my_set_no_ticks(xticks=4)
ax.my_set_no_ticks(yticks=4)
 
for s, c, coords in zip(['M1', 'M2'],
                        ['b', 'g'],
                        [[0.87,0.19], [0.87,0.07]]):
       
    ax.text(coords[0], coords[1], s, color=c, 
        transform=ax.transAxes, va='center', rotation=0)
     
#########################################
# CS-STN delay effect on phase shift ST-TI #TA-TI 
#########################################
ax=axs[2]
 
l=[]
for key in sorted(d['data'].keys()):
    v=d['data'][key]['Net_1']['GI_ST']['phases_diff_with_cohere']
    l.append(v)
l=l[organize['CSdelay']]
colors=misc.make_N_colors('copper', len(l))
  
for i, trace in enumerate(l):
    x=numpy.linspace(-numpy.pi*3,numpy.pi*3,len(trace))
    norm=sum(trace)*(x[-1]-x[0])/len(x)
    ax.plot(x, trace/norm, color=colors[i])
ax.set_xlim([-numpy.pi,numpy.pi])
  
 
sm = pylab.cm.ScalarMappable(cmap='copper', 
                             norm=pylab.normalize(vmin=2.5, vmax=20))
sm._A = []
  
box = ax.get_position()
pos=[box.x0+1.03*box.width, box.y0+box.height*0.1,
     0.01,  box.height*0.8]
axColor=pylab.axes(pos)
cbar=pylab.colorbar(sm, cax=axColor)
tick_locator = ticker.MaxNLocator(nbins=4)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.tick_params( length=1, )
  
ax.text(1.3, 0.5,  r'$t_{delay}^{CTX\to STN}$ (ms)', 
        transform=ax.transAxes,  va='center', rotation=270) 
ax.set_xlabel(r'Angle (rad)')
ax.set_ylabel('Norm. count ST-TI')
ax.my_set_no_ticks(xticks=4)
ax.my_set_no_ticks(yticks=4)
ax.set_ylim([0,0.31])
 

# CS-STN delay effect on phase shift ST-TI #TA-TI 
#########################################
ax=axs[3]
 
l=[]
for key in sorted(d['data'].keys()):
    v=d['data'][key]['Net_1']['GI_ST']['phases_diff_with_cohere']
    l.append(v)
l=l[organize['CTXSTRdelay']]
colors=misc.make_N_colors('copper', len(l))
  
for i, trace in enumerate(l):
    x=numpy.linspace(-numpy.pi*3,numpy.pi*3,len(trace))
    norm=sum(trace)*(x[-1]-x[0])/len(x)
    ax.plot(x, trace/norm, color=colors[i])
ax.set_xlim([-numpy.pi,numpy.pi])
  
 
sm = pylab.cm.ScalarMappable(cmap='copper', 
                             norm=pylab.normalize(vmin=2.5, vmax=20))
sm._A = []
  
box = ax.get_position()
pos=[box.x0+1.03*box.width, box.y0+box.height*0.1,
     0.01,  box.height*0.8]
axColor=pylab.axes(pos)
cbar=pylab.colorbar(sm, cax=axColor)
tick_locator = ticker.MaxNLocator(nbins=4)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.tick_params( length=1, )
  
ax.text(1.3, 0.5,  r'$t_{delay}^{CTX\to str}$ (ms)', 
        transform=ax.transAxes,  va='center', rotation=270) 
ax.set_xlabel(r'Angle (rad)')
ax.set_ylabel('Norm. count ST-TI')
ax.my_set_no_ticks(xticks=4)
ax.my_set_no_ticks(yticks=4)
ax.set_ylim([0,0.31])
 
#########################################
# fan in  str to GA 
#########################################
 
ax=axs[1]
   
l=[]
for key in sorted(d['data'].keys()):
    v=d['data'][key]['Net_1']['GI_GA']['phases_diff_with_cohere']
    l.append(v)
l=[l[0]]+l[organize['M2GAfan']]
 
colors=misc.make_N_colors('copper', len(l))
     
for i, trace in enumerate(l):
    x=numpy.linspace(-numpy.pi*3,numpy.pi*3,len(trace))
    norm=sum(trace)*(x[-1]-x[0])/len(x)
    ax.plot(x, trace/norm, color=colors[i])
ax.set_xlim([-numpy.pi,numpy.pi])
   
sm = pylab.cm.ScalarMappable(cmap='copper', 
                             norm=pylab.normalize(vmin=0, vmax=125))
sm._A = []
   
   
box = ax.get_position()
pos=[box.x0+1.03*box.width, box.y0+box.height*0.1,
     0.01, box.height*0.8]
axColor=pylab.axes(pos)
cbar=pylab.colorbar(sm, cax=axColor)
tick_locator = ticker.MaxNLocator(nbins=4)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.tick_params( length=1, )
   
ax.text(1.35, 0.5, r'$Fan_{in}^{MSN\to TA}$ (#)', 
        transform=ax.transAxes, va='center', rotation=270)
  
ax.set_xlabel(r'Angle (rad)')
ax.set_ylabel('Norm. count TI-TA')
ax.my_set_no_ticks(xticks=4)
ax.my_set_no_ticks(yticks=4)
ax.set_ylim([0,0.31])
 
 
 
#########################################
# fan in TI to str 
#########################################
 
ax=axs[4]
 
l=[]
for key in sorted(d['data'].keys()):
    v=d['data'][key]['Net_1']['GI_GA']['phases_diff_with_cohere']
    l.append(v)
l=[l[0]]+l[organize['GIM2fan']]
 
colors=misc.make_N_colors('copper', len(l))
   
for i, trace in enumerate(l):
    x=numpy.linspace(-numpy.pi*3,numpy.pi*3,len(trace))
    norm=sum(trace)*(x[-1]-x[0])/len(x)
    ax.plot(x, trace/norm, color=colors[i])
ax.set_xlim([-numpy.pi,numpy.pi])
   
sm = pylab.cm.ScalarMappable(cmap='copper', 
                             norm=pylab.normalize(vmin=0, vmax=5))
sm._A = []
   
box = ax.get_position()
pos=[box.x0+1.03*box.width, box.y0+box.height*0.1,
     0.01, box.height*0.8]
axColor=pylab.axes(pos)
cbar=pylab.colorbar(sm, cax=axColor, ticks=range(len(l)))
tick_locator = ticker.MaxNLocator(nbins=4)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.tick_params( length=1, )
   
ax.text(1.35, 0.5, r'$Fan_{in}^{TI\to str}$ (#)',
        transform=ax.transAxes, va='center',  rotation=270) 
ax.set_xlabel(r'Angle (rad)')
ax.set_ylabel('Norm. count TI-TA')
ax.my_set_no_ticks(xticks=4)
ax.my_set_no_ticks(yticks=4)
ax.set_ylim([0,0.31])

#########################################
# CTX-STN rate and g conductance GP-STN
#########################################

ax=axs[6]

l=[]
for key in sorted(d['data'].keys()):
    v=d['data'][key]['Net_1']['GI_GA']['phases_diff_with_cohere']
    l.append(v)

l=[l[0]]+l[organize['STGA']]

ln=list_names[organize['STGA']]
ln=[str(int(float(a[17])))+', ' +str(round(float(a[18]),2)) for a in ln]

colors=misc.make_N_colors('copper', len(l))
  
for i, trace in enumerate(l):
    x=numpy.linspace(-numpy.pi*3,numpy.pi*3,len(trace))
    norm=sum(trace)*(x[-1]-x[0])/len(x)
    ax.plot(x, trace/norm, color=colors[i])
ax.set_xlim([-numpy.pi,numpy.pi])
  
sm = pylab.cm.ScalarMappable(cmap='copper', 
                             norm=pylab.normalize(vmin=0, vmax=len(ln)-1))
sm._A = []
    
box = ax.get_position()
pos=[box.x0+1.03*box.width, box.y0+box.height*0.1,
     0.01, box.height*0.8]
axColor=pylab.axes(pos)
cbar=pylab.colorbar(sm, cax=axColor, ticks=range(len(ln)))

cbar.ax.tick_params( length=1, )
cbar.ax.set_yticklabels(ln, fontsize=5*scale)  

 
ax.text(1.55,  0.5, r'$v^{CTX\to STN}$, $g_{gaba}^{GPe_{TI}\to STN}$', 
        transform=ax.transAxes, va='center', rotation=270) 

ax.set_xlabel(r'Angle (rad)')
ax.set_ylabel('Norm. count TI-TA')
ax.my_set_no_ticks(xticks=4)
ax.my_set_no_ticks(yticks=4)
ax.set_ylim([0,0.31])


# dd=d['d_raw_control']['oscillation_index']['labelsy']
# bol= numpy.array([l in ['GA', 'GI', 'SN','ST'] for l in dd])
# v=d['d_raw_control']['oscillation_index']['y'][bol,organize['STGA']]
#   
# ln=list_names[organize['STGA']]
# ln=[str(int(float(a[4])))+', '
#     +str(round(float(a[5].split('-')[0]),2)) for a in ln]
# ax.set_xlabel(r'$r_{CTX\to STN}$, $g_{GPe\to STN}$') 
# ax.plot(range(1,len(ln)+1),v.transpose())
#  
#  
# ax.set_xlim([0,len(ln)+1.2])
# ax.set_ylim([0,1.])
# ax.set_xticks(range(1,len(ln)+1))
# ax.set_xticklabels(ln, rotation=50, horizontalalignment='right')
# ax.set_ylabel('Oscillation index')
# ax.my_set_no_ticks(yticks=4)
#  
# for s, c, coords in zip(['TA', 'TI', 'SN', 'ST'],
#                         ['b', 'g', 'r', 'c'],
#                         [[0.88,0.5], [0.88,0.78], [0.88,0.66], [0.88,0.9]]):
#       
#     ax.text(coords[0], coords[1], s, color=c, 
#         transform=ax.transAxes, va='center', rotation=0)



#########################################
# fan in TI TA to TA 
#########################################

ax=axs[5]
# v=d['data']
  
l=[]
for key in sorted(d['data'].keys()):
    v=d['data'][key]['Net_1']['GI_GA']['phases_diff_with_cohere']
    l.append(v)

l=[l[0]]+l[organize['GAGItoGA']]

ln=list_names[organize['GAGItoGA']]
ln=[str(int(float(a[15])))+','+str(int(float(a[17]))) for a in ln]

colors=misc.make_N_colors('copper', len(l))
  
for i, trace in enumerate(l):
    x=numpy.linspace(-numpy.pi*3,numpy.pi*3,len(trace))
    norm=sum(trace)*(x[-1]-x[0])/len(x)
    ax.plot(x, trace/norm, color=colors[i])
ax.set_xlim([-numpy.pi,numpy.pi])
  
sm = pylab.cm.ScalarMappable(cmap='copper', 
                             norm=pylab.normalize(vmin=0, vmax=len(ln)-1))
sm._A = []
    
box = ax.get_position()
pos=[box.x0+1.03*box.width, box.y0+box.height*0.1,
     0.01, box.height*0.8]
axColor=pylab.axes(pos)
cbar=pylab.colorbar(sm, cax=axColor, ticks=range(len(ln)))

cbar.ax.tick_params( length=1, )
cbar.ax.set_yticklabels(ln, fontsize=5*scale)  

ax.text(1.35,  0.5, r'$Fan_{in}^{[TA,TI]\to TA}$ (#)', 
        transform=ax.transAxes, va='center', rotation=270) 

ax.set_xlabel(r'Angle (rad)')
ax.set_ylabel('Norm. count TI-TA')
ax.my_set_no_ticks(xticks=4)
ax.my_set_no_ticks(yticks=4)
ax.set_ylim([0,0.31])

for ax in axs:
    ax.tick_params(direction='out',
                   length=2,
                   width=0.5,
                   pad=0.01,
                    top=False, right=False
                    )
save_figures([fig], script_name, dpi=400)

pylab.show()



