'''
Created on Nov 14, 2014

@author: mikael
'''
import matplotlib.gridspec as gridspec
import core.plot_settings as ps
import pylab
import pprint
pp=pprint.pprint

from scripts_inhibition import effect_conns
from core import misc
from scripts_inhibition.base_simulate import save_figures


def gs_builder(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.2 ), 
              hspace=kwargs.get('hspace', 0.2))
# 
    iterator=[]
    for j in range(1,6,2):
        for i in range(1,10,2):
        
        
            iterator.append([slice(i, i+2),slice(j, j+2)])
    
#     iterator = ([[i, 1] for i in range(1,6)]+
#                 [[i, 2] for i in range(1,6)]+
#                 [[i, 3] for i in range(1,6)]+
#                 [[i, 4] for i in range(1,6)]+
#                 [[i, 5] for i in range(1,6)])
#     
#     iterator = ([[1, i] for i in range(1,6)]+
#                 [[2, i] for i in range(1,6)]+
#                 [[3, i] for i in range(1,6)]+
#                 [[4, i] for i in range(1,6)]+
#                 [[5, i] for i in range(1,6)])
    
    return iterator, gs, 

attrs=['mean_rate_slices', 'set_0', 'set_1']
models=['M1', 'M2', 'SN']
paths=[]
paths.append('/home/mikael/results/papers/inhibition/network/'
             +'milner/simulate_Go_NoGo_XXX_no_ss_scaling_v2_oscillation_amp_1.1/')
paths.append('/home/mikael/results/papers/inhibition/network/'
             +'milner/simulate_Go_NoGo_XXX_scaling_MS_active_ST_pulse_v2/')
paths.append('/home/mikael/results/papers/inhibition/network/'
             +'milner/simulate_Go_NoGo_XXX_scaling_MS_active_v2/')

s1='script_000{0}_exc1.0_EIEACS_600.0_STGAw_0.1875_GAGA_5.0_GIGA_25.0-ss-1.0'
s2='script_000{}_MsGa-MS-weight0.25_ST-GI-0.75-GaMs-0.4-down-C2-EiEa-mod-fast-5.0-ss-{}'
s3='script_{}_MsGa-MS-weight0.25_ST-GI-0.75-GaMs-0.4-down-C2-EiEa-mod-fast-5.0-ss-{}'
nets1=['Net_0', 'Net_1', 'Net_2', 'Net_3', 'Net_4']
nets2=['Net_0']
files={
        '10-noss':[paths[0]+s1.format(0), nets1],
#         '25-noss':[paths[0]+s1.format(1), nets1],
        '50-noss':[paths[0]+s1.format(1), nets1],
#         '75-noss':[paths[0]+s1.format(3), nets1],
       '100-noss':[paths[0]+s1.format(2), nets1],
#        '20-ss':[paths[2]+s3.format(2,5.0), nets1],
#        '50-ss':[paths[2]+s3.format(1,2.5), nets1],
#        '100-ss':[paths[2]+s3.format(0,1.0), nets1], 
#        '20-ST-ss':[paths[1]+s2.format(1,5.0), nets2],
#        '100-ST-ss':[paths[1]+s2.format(0,1.0), nets2] 
       }


d={}
for key, val in files.items():
    nets=val[1]
    d_tmp=effect_conns.gather('', nets, models, attrs, 
                              dic_keys=[key], 
                              file_names=[val[0]])
    misc.dict_update(d, d_tmp)
print d.keys()
pp(d)

from Go_NoGo_compete import show_heat_map

builder=[['10-noss', nets1],
#         ['25-noss', nets1],
        ['50-noss', nets1],
#         ['75-noss', nets1],
        ['100-noss', nets1]
         ]
dd={}
titles=[]
i=0
for name, nets in builder:
    for net in nets:
        print name, net
        if not (net in d[name].keys()):
            i+=1
            continue
        dd['Net_{:0>2}'.format(i)]=d[name][net]
        
        titles.append(name+'_'+net)
        i+=1 
pp(dd)

val=int(72/2.54*17.6*(1-17./48))
scale=1
fig, axs=ps.get_figure2(n_rows=11, 
                        n_cols=11,
                        w=val*scale,
                        h=300*scale,  
                        fontsize=7*scale,
                        title_fontsize=7*scale,
                        gs_builder=gs_builder) 

k={'axs':axs,
   'do_colorbar':False, 
   'fig':fig,
   'models':['SN'],
   'print_statistics':False,
   'resolution':10,
   'titles':['']*5*5,
    'type_of_plot':'mean',
    'vlim_rate':[-100, 100], 
    'marker_size':8}

show_heat_map(dd, 'mean_rate_slices', **k)
# pylab.show()
for ax in axs:
        ax.tick_params(direction='out',length=2,
                       width=0.5, pad=0.01,
                        top=False, right=False
                        )  


im=axs[0].collections[0]
box = axs[0].get_position()
pos=[box.x0+0.5*box.width, 
     box.y0+box.height+box.height*0.45, 
     box.width*1.5, 
     0.025]
axColor=pylab.axes(pos)
#     axColor = pylab.axes([0.05, 0.9, 1.0, 0.05])
cbar=pylab.colorbar(im, cax = axColor, orientation="horizontal")
cbar.ax.set_title('Contrast (Hz)',
                  fontsize=7*scale)#, rotation=270)
cbar.set_ticks([-90,0,90])
# cl = pylab.getp(cbar.ax, 'ymajorticklabels') 
# pylab.setp(cl, fontsize=20) 
cbar.ax.tick_params(labelsize=7*scale,
                    length=2, ) 
# cbar.ax.set_yticks(fontsize=18)
# cbar.set_ticklabels( fontsize=18)

axs[0].legend(['Dual selection','Selection', 'No selection'], 
#               ncol=1, 
          scatterpoints=1,
          frameon=False,
          labelspacing=0.1,
          handletextpad=0.1,
          columnspacing=0.3,
          bbox_to_anchor=(5.5, 2.),
          prop={'size':7*scale},
          markerscale=2.5)

labels= ['{} %'.format(i) for i in [10,50,100]]
for i, s in zip(range(4,15,5),labels):
    axs[i].text( 0.5, -.3, s, 
                transform=axs[i].transAxes,
                horizontalalignment='center')

axs[9].text( 0.5, -0.65, 'Action pool activation', 
                transform=axs[9].transAxes,
                horizontalalignment='center')

labels= ['Only D1',
             'D1 & D2',
             r'No MSN$\to$MSN',
             r'No FSN$\to$MSN',
             r'No $GPe_{TA}$$\to$MSN']
for i, s in enumerate(labels):
    axs[i].text(k.get('cohere_ylabel_ypos', -0.1), 
                0.5, 
                s, 
                transform=axs[i].transAxes,
                horizontalalignment='right', 
                rotation=0,
                fontsize=5)    

for i, ax in enumerate(axs): 
    ax.set_xlabel('')
    ax.set_ylabel('')
#     a=ax.get_xticklabels()
    ax.tick_params(axis='both', which='major', labelsize=7*scale)
#     ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
#     ax.set_yticklabels(fontsize=20)

    ax.my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True)
    
#     axs[0].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)
#     axs[1].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True)
#     axs[2].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)
#     axs[3].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True)
#     axs[4].my_remove_axis(xaxis=False, yaxis=False,keep_ticks=True)
#     axs[5].my_remove_axis(xaxis=False, yaxis=True,keep_ticks=True)
    
    
#     
#     if i==4:
#         ax.text(1.05, -.3, 
#                   'Cortical input action 1',
#                     horizontalalignment='center', 
#                     transform=axs[i].transAxes) 
#         ax.set_xticks([1,1.5, 2, 2.5])
#         ax.set_xticklabels(['1.0','1.5','2.0','2.5'])
#             
#     if i==2:
#         ax.set_ylabel('Cortical input action 2')
       
save_figures([fig], __file__.split('/')[-1][0:-3]+'/data', dpi=200)

pylab.show()