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
from scripts_inhibition.simulate import save_figures



def gs_builder(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0. ), 
              hspace=kwargs.get('hspace', 0.4 ))

    iterator = [[slice(2,7), slice(1,8)],
                [slice(2,7), slice(9,16)],
                [slice(8,13), slice(1,8)],
                [slice(8,13), slice(9,16)],
                [slice(14,19), slice(1,8)],
                [slice(14,19), slice(9,16)]]
    
    return iterator, gs, 

attrs=['mean_rate_slices', 'set_0', 'set_1']
models=['M1', 'M2', 'SN']
paths=[]


# paths.append('/home/mikael/results/papers/inhibition/network/'
#              +'milner/simulate_Go_NoGo_XXX_no_ss_act0.2_v2/')
# paths.append('/home/mikael/results/papers/inhibition/network/'
#              +'milner/simulate_Go_NoGo_XXX_no_ss_act0.2_ST_pulse_v2/')

paths.append('/home/mikael/results/papers/inhibition/network/'
             +'milner/fig5_simulate_Go_NoGo_XXX_no_ss_act0.2_ST_pulse_v2/')
paths.append('/home/mikael/results/papers/inhibition/network/'
             +'milner/fig5_simulate_Go_NoGo_XXX_no_ss_scaling_act_ST_pulse_v2/')

s1='script_000{}_GAGA_25.0_GIGA_5.0-ss-1.0'

nets1=['Net_0', 'Net_1', 'Net_2', 'Net_3', 'Net_4']
nets2=['Net_0']
files={
       '20':[paths[0]+s1.format(0), nets1],
       '20-ST':[paths[1]+s1.format(0), nets2],
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

builder=[['20', nets1],
         ['20-ST', nets2]]
dd={}
i=0
for name, nets in builder:
    for net in nets:
        print name, net
        if not (net in d[name].keys()):
            i+=1
            continue
        dd['Net_{:0>2}'.format(i)]=d[name][net]
        
#         titles.append(name+'_'+net)
        i+=1 
# for name, nets in builder:
#     for net in nets:
#         dd['Net_{}'.format(i)]=d[name][net]
#         i+=1 
pp(dd)
print len(dd['Net_0']['set_1']['SN']['mean_rate_slices'].y)


fig, axs=ps.get_figure2(n_rows=19, 
                        n_cols=16,
                        w=int(72/2.54*17.6*(17./48)),
                        h=300,  
                        fontsize=7,
                        title_fontsize=7,
                        gs_builder=gs_builder) 

k={'axs':axs,
   'do_colorbar':False, 
   'fig':fig,
   'models':['SN'],
   'print_statistics':False,
   'resolution':10,
   'titles':['Only D1',
             'D1 & D2',
             r'No MSN$\to$MSN',
             r'No FSN$\to$MSN',
             r'No $GPe_{TA}$$\to$MSN',
             r'Pulse STN'],
    'type_of_plot':'mean',
    'vlim_rate':[-100, 100]}


show_heat_map(dd, 'mean_rate_slices', **k)
for ax in axs:
        ax.tick_params(direction='out',
                       length=2,
                       width=0.5,
                       pad=0.01,
                        top=False, right=False
                        )  

im=axs[0].collections[0]
box = axs[0].get_position()
pos=[box.x0+0.0*box.width, 
     box.y0+box.height+box.height*0.45, 
     box.width*0.6, 
     0.025]
axColor=pylab.axes(pos)
#     axColor = pylab.axes([0.05, 0.9, 1.0, 0.05])
cbar=pylab.colorbar(im, cax = axColor, orientation="horizontal")
cbar.ax.set_title('Contrast (Hz)',
                  fontsize=7)#, rotation=270)
cbar.set_ticks([-90,0,90])
# cl = pylab.getp(cbar.ax, 'ymajorticklabels') 
# pylab.setp(cl, fontsize=20) 
cbar.ax.tick_params(labelsize=7,
                    length=2, ) 
# cbar.ax.set_yticks(fontsize=18)
# cbar.set_ticklabels( fontsize=18)

axs[1].legend(['Dual selection',
               'Selection', 
               'No selection'], 
#               ncol=1, 
          scatterpoints=1,
          frameon=False,
          labelspacing=0.1,
          handletextpad=0.1,
          columnspacing=0.3,
          bbox_to_anchor=(1.1, 1.75),
          markerscale=1.5,
          prop={'size':7})

# from matplotlib import ticker
# tick_locator = ticker.MaxNLocator(nbins=4)
# cbar.locator = tick_locator
# cbar.update_ticks()


for i, ax in enumerate(axs): 
    ax.set_xlabel('')
    ax.set_ylabel('')
#     a=ax.get_xticklabels()
    ax.tick_params(axis='both', which='major', 
#                    labelsize=20
                   )
#     ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
#     ax.set_yticklabels(fontsize=20)
    ax.set_xticks([1, 1.5, 2, 2.5, 3])
    ax.set_yticks([1, 1.5, 2, 2.5, 3])
 
    if i==4:
        ax.text(1.05, -.37, 
                  'Cortical input action 1',
                    horizontalalignment='center', 
                    transform=axs[i].transAxes,
#                     fontsize=20
                    ) 
        ax.set_xticks([1,1.5, 2, 2.5])
#         ax.set_xticklabels(['1.0','1.5','2.0','2.5'])
            
    if i==2:
        ax.set_ylabel('Cortical input action 2',
#                       fontsize=20
                      )
       
axs[0].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)
axs[1].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True)
axs[2].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)
axs[3].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True)
axs[4].my_remove_axis(xaxis=False, yaxis=False,keep_ticks=True)
axs[5].my_remove_axis(xaxis=False, yaxis=True,keep_ticks=True)

save_figures([fig], __file__.split('/')[-1][0:-3]+'/data', dpi=200)

pylab.show()