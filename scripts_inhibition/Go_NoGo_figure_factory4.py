'''
Created on Nov 14, 2014

@author: mikael
'''
import matplotlib.gridspec as gridspec
import toolbox.plot_settings as ps
import pylab
import pprint
pp=pprint.pprint

from scripts_inhibition import effect_conns
from toolbox import misc
from simulate import save_figures



def gs_builder(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0. ), 
              hspace=kwargs.get('hspace', 0.4 ))

    iterator = [[slice(2,5), slice(1,4)],]
    
    return iterator, gs, 

attrs=['mean_rate_slices', 'set_0', 'set_1']
models=['M1', 'M2', 'SN']
paths=[]


paths.append('/home/mikael/results/papers/inhibition/network/'
             +'supermicro/simulate_Go_NoGo_XXX_star2_v2_5x5/')



s1='script_000{}_MsGa-MS-weight0.25_ST-GI-0.75-GaMs-0.4-down-C2-EiEa-mod-fast-5.0-ss-6.25'

nets1=['Net_1']
files={
       'D1D2':[paths[0]+s1.format(0), nets1],
       }


d={}
for key, val in files.items():
    nets=val[1]
    d_tmp=effect_conns.gather('', nets, models, attrs, 
                              dic_keys=[key], 
                              file_names=[val[0]])
    misc.dict_update(d, d_tmp)
print d.keys()
# pp(d)

from Go_NoGo_compete import show_heat_map

builder=[['D1D2', nets1]]
dd={}
i=0
for name, nets in builder:
    for net in nets:
        dd['Net_{}'.format(i)]=d[name][net]
        i+=1 
pp(dd)
print len(dd['Net_0']['set_1']['SN']['mean_rate_slices'].y)

figs=[]
fig, axs=ps.get_figure2(n_rows=6, 
                        n_cols=6,
                        w=450,
                        h=370,  
                        fontsize=24,
                        title_fontsize=24,
                        gs_builder=gs_builder) 
figs.append(fig)
for ax in axs:
    ax.tick_params(direction='out',
                   length=6, 
                   top=False, 
                   right=False)  

k={'axs':axs,
   'do_colorbar':False, 
   'fig':fig,
   'models':['SN'],
   'print_statistics':False,
   'resolution':5,
   'titles':[''],
    'type_of_plot':'mean',
    'vlim_rate':[-100, 100]}

show_heat_map(dd, 'mean_rate_slices', **k)


im=axs[0].collections[0]
box = axs[0].get_position()
pos=[box.x0 + box.width * 1.06, 
      box.y0+box.height*0.1, 
      0.02, 
      box.height*0.8]

axColor=pylab.axes(pos)
#     axColor = pylab.axes([0.05, 0.9, 1.0, 0.05])

cbar=pylab.colorbar(im, cax = axColor, orientation="vertical")
cbar.set_ticks([-90,0,90])
cbar.set_label('Contrast (Hz)', rotation=270)

# from matplotlib import ticker
# tick_locator = ticker.MaxNLocator(nbins=4)
# cbar.locator = tick_locator
# cbar.update_ticks()


for i, ax in enumerate(axs): 
    ax.set_xlabel('Input action 2')
    ax.set_ylabel('Input action 1')
    ax.legend(['Dual selection','Selection', 'No selection'], 
#               ncol=1, 
              scatterpoints=1,
              frameon=False,
              labelspacing=0.1,
              handletextpad=0.1,
              columnspacing=0.3,
              bbox_to_anchor=(1.2, 1.8),)
#     ax.set_xticks([1.2, 1.8, 2.4, 3.0])
    ax.set_xticks([1,2,3])
    ax.set_yticks([1,2,3])
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
#     ax.my_set_no_ticks(xticks=4, yticks=4)
#     ax.my_push_axis()
#     a=ax.get_xticklabels()
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
#     ax.set_yticklabels(fontsize=20)


#     axs[0].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)
#     axs[1].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True)
#     axs[2].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)
#     axs[3].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True)
#     axs[4].my_remove_axis(xaxis=False, yaxis=False,keep_ticks=True)
#     axs[5].my_remove_axis(xaxis=False, yaxis=True,keep_ticks=True)
#     
#     
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
       

script_name=(__file__.split('/')[-1][0:-3]+'/data')
save_figures(figs, script_name)

pylab.show()