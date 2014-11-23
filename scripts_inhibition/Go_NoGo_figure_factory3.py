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
    gs.update(wspace=kwargs.get('wspace', 0.1 ), 
              hspace=kwargs.get('hspace', 0.0))
# 
#     iterator = ([[i, 1] for i in range(1,6)]+
#                 [[i, 2] for i in range(1,6)]+
#                 [[i, 3] for i in range(1,6)]+
#                 [[i, 4] for i in range(1,6)]+
#                 [[i, 5] for i in range(1,6)])
#     
    iterator = ([[slice(2,4), i] for i in range(0,5)]+
                [[slice(5,7), i] for i in range(0,5)]+
                [[slice(8,10), i] for i in range(0,4)]
#                 [[4, i] for i in range(1,6)]+
#                 [[5, i] for i in range(1,6)]
                )
    
    return iterator, gs, 

attrs=['mean_rate_slices', 'set_0', 'set_1']
models=['M1', 'M2', 'SN']
paths=[]

paths.append('/home/mikael/results/papers/inhibition/network/'
             +'milner/simulate_Go_NoGo_XXX_nodop_FS_recovery2_v2/')
paths.append('/home/mikael/results/papers/inhibition/network/'
             +'milner/simulate_Go_NoGo_XXX_nodop_FS_v2/')

s1='script_000{}_MsGa-MS-weight0.25_ST-GI-0.75-GaMs-0.4-down-C2-EiEa-mod-fast-5.0-ss-1.0-{}'
nets1=['Net_0']
files={'GA_M2_2.6':[paths[0]+s1.format(0,'GA_M2_pert_2.6'), nets1],
       'FS_M2_2.6':[paths[0]+s1.format(1,'FS_M2_pert_2.6'), nets1],
       'ST_GI_0.2':[paths[0]+s1.format(2,'ST_GI_pert_0.2'), nets1],
       'M2_M2_2.6':[paths[0]+s1.format(3,'M2_M2_pert_2.6'), nets1],
       'GI_GA_0.2':[paths[0]+s1.format(4,'GI_GA_pert_0.2'), nets1],
       'M2_GI_0.2':[paths[0]+s1.format(5,'M2_GI_pert_0.2'), nets1],
       'GI_GI_2.6':[paths[0]+s1.format(6,'GI_GI_pert_2.6'), nets1],
       'M2_mod0':[paths[0]+s1.format(7,'M2_pert_mod0'), nets1],
       'GI_mod0':[paths[0]+s1.format(8,'GI_pert_mod0'), nets1],
       'ST_mod0':[paths[0]+s1.format(9,'ST_pert_mod0'), nets1], }


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

builder=[
         
         ['FS_M2_2.6', nets1],
         ['GA_M2_2.6', nets1],
         ['M2_M2_2.6', nets1],
         ['ST_GI_0.2', nets1],
         ['GI_GA_0.2', nets1],
         ['M2_GI_0.2', nets1],
         ['GI_GI_2.6', nets1],
         ['GI_mod0', nets1],
         ['M2_mod0', nets1],
         ['ST_mod0', nets1],
         ]
dd={}
titles=[]
i=0
for name, nets in builder:
    for net in nets:
        dd['Net_{:0>2}'.format(i)]=d[name][net]
        titles.append(name+'_'+net)
        i+=1 
pp(dd)


fig, axs=ps.get_figure2(n_rows=10, 
                        n_cols=5,
                        w=800,
                        h=600,  
                        fontsize=24,
                        title_fontsize=24,
                        gs_builder=gs_builder) 

k={'axs':axs,
   'do_colorbar':False, 
   'fig':fig,
   'models':['SN'],
   'print_statistics':False,
   'resolution':10,
   'titles':['']*15,
   'fontsize_ax_titles':16,
   'pos_ax_titles':1.08,
    'type_of_plot':'mean',
    'vlim_rate':[-100, 100], 
    'marker_size':30}

show_heat_map(dd, 'mean_rate_slices', **k)


titles=[[r'FSN$\to$$MSN_{D2}$','$\uparrow$2.6*$w_{0}$'],
        [r'$GPe_{TA}$$\to$$MSN_{D2}$',r'$\uparrow$2.6*$w_{0}$'],
        [r'$MSN_{D2}$$\to$$MSN_{D2}$',r'$\uparrow$2.6*$w_{0}$'],
        [r'STN$\to$$GPe_{TI}$',r'$\downarrow$0.2*$w_{0}$'],
        [r'$GPe_{TI}$$\to$$GPe_{TA}$',r'$\downarrow$0.2*$w_{0}$'],
        [r'$MSN_{D2}$$\to$$GPe_{TI}$',r'$\downarrow$0.2*$w_{0}$'],
        [r'$GPe_{TI}$$\to$$GPe_{TI}$',r'$\uparrow$2.6*$w_{0}$'],
        [r'$GPe_{TI}$',r'$\downarrow$30 Hz'],
        [r'$MSN_{D2}$',r'$\downarrow$6 Hz'],
        [r'STN',r'$\downarrow$20 Hz']]
             
im=axs[2].collections[0]
box = axs[2].get_position()
pos=[box.x0-0.75*box.width, 
     box.y0+box.height+box.height*1., 
     box.width*2.5, 
     0.025]
axColor=pylab.axes(pos)
#     axColor = pylab.axes([0.05, 0.9, 1.0, 0.05])
cbar=pylab.colorbar(im, cax = axColor, orientation="horizontal")
cbar.ax.set_title('Contrast (spikes/s')#, rotation=270)
from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=4)
cbar.locator = tick_locator
cbar.update_ticks()

# labels= ['{} %'.format(i) for i in [10,25,50,75,100]]
# for i, s in zip(range(4,25,5),labels):
#     axs[i].text( 0.5, -.3, s, 
#                 transform=axs[i].transAxes,
#                 horizontalalignment='center')
# 
# # axs[2].text( 0.5, -0.65, 'Action pool activation', 
#                 transform=axs[14].transAxes,
#                 horizontalalignment='center')

# labels= ['Only D1',
#              'D1 & D2',
#              r'No MSN$\to$MSN',
#              r'No FSN$\to$MSN',
#              r'No $GPe_{TA}$$\to$MSN']
# for i, s in enumerate(labels):
#     axs[i].text(k.get('cohere_ylabel_ypos', -0.1), 
#                 0.5, 
#                 s, 
#                 transform=axs[i].transAxes,
#                 horizontalalignment='right', 
#                 rotation=0)    

for i, ax in enumerate(axs): 
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    if i<len(titles):
        axs[i].text(0.5, 1.28, titles[i][0],
                            horizontalalignment='center', 
                            transform=axs[i].transAxes,
                            fontsize=k.get('fontsize_ax_titles',24)) 
        axs[i].text(0.5, 1.06, titles[i][1],
                            horizontalalignment='center', 
                            transform=axs[i].transAxes,
                            fontsize=k.get('fontsize_ax_titles',24)) 
#     a=ax.get_xticklabels()
    ax.tick_params(axis='both', which='major', labelsize=20)
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

save_figures([fig], __file__.split('/')[-1][0:-3]+'/data')

pylab.show()