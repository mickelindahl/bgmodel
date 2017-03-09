'''
Created on Nov 14, 2014

@author: mikael
'''
import matplotlib.gridspec as gridspec
import core.plot_settings as ps
import os
import pylab
import pprint
pp=pprint.pprint

from scripts_inhibition import base_effect_conns
from core import misc
from scripts_inhibition.base_simulate import save_figures
from scripts_inhibition.base_Go_NoGo_compete import show_heat_map, show_variability_several

def add_to_files(files, path, nets, **kw):
    l=os.listdir(path)
    names=kw.get('names',[])
    i=0
    for f in l:
        if f[0:6]!='script':
            
            continue
#         if names:
#             name=names[i]
#         else:
        name=f.split('-')[-1]
        if name in names:
            continue
        
            
        files.update({name:[path+f,nets]})
        
        i+=1
def gs_builder(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.1 ), 
              hspace=kwargs.get('hspace', 0.4))
# 
#     iterator = ([[i, 1] for i in range(1,6)]+
#                 [[i, 2] for i in range(1,6)]+
#                 [[i, 3] for i in range(1,6)]+
#                 [[i, 4] for i in range(1,6)]+
#                 [[i, 5] for i in range(1,6)])
#     
    iterator = ([[slice(1,2), i] for i in range(0,10)]+
                [[slice(2,3), i] for i in range(0,10)]+
                [[slice(4,5), i] for i in range(0,10)]+
                [[slice(5,6), i] for i in range(0,10)]+
                [[slice(6,7), i] for i in range(0,10)]+
                [[slice(7,8), i] for i in range(0,10)]
#                 [[slice(9,10), i] for i in range(0,7)]
#                 [[slice(6,7), i] for i in range(0,7)]
#                 [[slice(7,8), i] for i in range(0,4)]
#                 [[4, i] for i in range(1,6)]+
#                 [[5, i] for i in range(1,6)]
                )
    
    return iterator, gs, 

attrs=['mean_rate_slices', 'set_0', 'set_1']
models=['M1', 'M2', 'SN']
paths=[]

paths.append('/home/mikael/results/papers/inhibition/network/'
             +'milner/eNeuro_fig_06_0.2_rest_7x7/')

paths.append('/home/mikael/results/papers/inhibition/network/'
             +'milner/eNeuro_fig_07_recovery_cases_7x7/')
# paths.append('/home/mikael/results/papers/inhibition/network/'
#              +'milner/fig6_Go_NoGo_XXX_nodop_FS/')
# 
# paths.append('/home/mikael/results/papers/inhibition/network/'
#              +'milner/fig5_Go_NoGo_XXX_no_ss_act0.2_v2/')


files={}
nets1=['Net_1']
nets2=['Net_0']

add_to_files(files, paths[0], nets1)
add_to_files(files, paths[1], nets2)
pp(files)

d={}
for key, val in files.items():
    nets=val[1]
    d_tmp=base_effect_conns.gather('', nets, models, attrs, 
                              dic_keys=[key], 
                              file_names=[val[0]])
    misc.dict_update(d, d_tmp)
pp(d)

builder=[['1.0', 'Net_1', 'Control'],
        ['Control', 'Net_0', 'Lesion'],
        
        ['CTX_M1',  'Net_0', r'$\beta^{CTX\to MSN_{D1}}_{I_{NMDA}}=0$'],
        ['CTX_M2',  'Net_0', r'$\beta^{CTX\to MSN_{D2}}_{I_{AMPA}}=0$'],
        ['CTX_ST', 'Net_0', r'$\beta^{CTX\to STN}_{I_{AMPA}}=0$'],
        ['FS', 'Net_0',  r'$\beta^{FSN}_{E_{L}}=0$'],
        ['FS_FS', 'Net_0', r'$\beta^{FSN\to FSN}_{N}=0$'],
        ['FS_M2', 'Net_0', r'$\beta^{FSN\to MSN_{D2}}_{N}=0$'],
        ['GA_FS', 'Net_0', r'$\beta^{TA\to FSN}_{I_{GABA}}=0$'],
        ['GA_MS', 'Net_0', r'$\beta^{TA\to MSN}_{I_{GABA}}=0$'],
        ['GI_FS', 'Net_0', r'$\beta^{TI\to FSN}_{I_{GABA}}=0$'],
        ['GP', 'Net_0', r'$\beta^{GPe}_{E_{L}}=0$'],
#         ['GP_FS', 'Net_0', r'$\beta^{GPe_{TA}\to FSN}_{N}=0$'],
        ['GP_GP', 'Net_0', r'$\beta^{GPe\to GPe}_{I_{GABA}}=0$'],
        ['GP_ST', 'Net_0', r'$\beta^{GPe\to STN}_{I_{GABA}}=0$'],
        ['M1', 'Net_0', r'$\beta^{MSN_{D1}}_{E_{L}}=0$'],
        ['M1_SN','Net_0', r'$\beta^{MSN_{D1}\to SNr}_{I_{GABA}}=0$'],
        ['M2_GI', 'Net_0', r'$\beta^{MSN_{D2}\to GPe}_{I_{GABA}}=0$'],
        ['MS_MS', 'Net_0', r'$\beta^{MSN\to MSN}_{I_{GABA}/N}=0$'],
        ['SN', 'Net_0', r'$\beta^{SNr}_{E_{L}}=0$'],
        ['ST_GP', 'Net_0', r'$\beta^{STN\to GPe}_{I_{GABA}}=0$'],
        
         ['FS_pert_mod1', 'Net_0', r'$r_{FSN} \uparrow$'],
#          ['GA_M1_pert_5','Net_0', r'$w_{TA\to MSN_{D1}} \uparrow$'],
         ['GA_M2_pert_0.0','Net_0', r'$w_{TA\to MSN_{D2}} \downarrow$'],
         ['GA_M2_pert_5', 'Net_0', r'$w_{TA\to MSN_{D2}} \uparrow$'],
#          ['GA_pert_mod0', 'Net_0', r'$r_{TA} \downarrow$'],
         ['GA_pert_mod1', 'Net_0', r'$r_{TA} \uparrow$'],
#          ['GI_GA_pert_0.0', 'Net_0', r'$w_{GPe_{TI}\to GPe_{TA}} \downarrow$'],
         ['GI_GI_pert_0.0', 'Net_0', r'$w_{TI\to TI} \downarrow$'],
         ['GI_GI_pert_5', 'Net_0', r'$w_{TI\to TI} \uparrow$'],
         ['GI_ST_pert_5', 'Net_0', r'$w_{TI\to STN} \uparrow$'],
#          ['GI_SN_pert_5', 'Net_0', r'$w_{GPe_{TI}\to SNr} \uparrow$'],
#          ['GI_ST_pert_5', 'Net_0', r'$w_{GPe_{TI}\to STN} \uparrow$'],
         ['GI_pert_mod0', 'Net_0', r'$r_{TI} \downarrow$'],
          ['GI_pert_mod1', 'Net_0', r'$r_{TI} \uparrow$'],
#          ['GI_pert_mod7', 'Net_0', r'$r_{GPe_{TI}} \uparrow$'],
         ['M1_pert_mod1', 'Net_0', r'$r_{MSN_{D1}} \uparrow$'],
         
         ['M2_GI_pert_0.0', 'Net_0', r'$w_{MSN_{D2} \to TI} \downarrow$'],
         ['M2_GI_pert_5','Net_0', r'$w_{MSN_{D2} \to TI} \uparrow$'],
#          ['M2_M2_pert_0.0', 'Net_0', r'$w_{MSN_{D2} \to MSN_{D2}} \downarrow$'],
         ['M2_M2_pert_5', 'Net_0', r'$w_{MSN_{D2} \to MSN_{D2}} \uparrow$'],
         ['M2_pert_mod0', 'Net_0', r'$r_{MSN_{D2}} \downarrow$'],  
         ['M2_pert_mod1', 'Net_0', r'$r_{MSN_{D2}} \uparrow$'],  

#          ['SN_pert_mod0', 'Net_0', r'$r_{SNr} \downarrow$'],
         ['ST_GA_pert_0.0', 'Net_0', r'$w_{STN \to TA} \downarrow$'],
         ['ST_GA_pert_5', 'Net_0', r'$w_{STN \to TA} \uparrow$'],
        ['ST_GI_pert_0.0', 'Net_0', r'$w_{STN \to TI} \downarrow$'],
#          ['ST_GI_pert_5', 'Net_0', r'$w_{STN \to GPe_{TI}} \uparrow$'],
         ['ST_pert_mod0', 'Net_0', r'$r_{STN} \downarrow$'],
         ['ST_pert_mod1', 'Net_0', r'$r_{STN} \uparrow$'],
#          ['M2_pert_mod0', 'Net_0'],       
#          ['ST_pert_mod7', 'Net_0']
         ['GP_pert_mod0', 'Net_0', r'$r_{GPe} \downarrow$'],
          ['GP_pert_mod1', 'Net_0', r'$r_{GPe} \uparrow$'],
#    
         ]


titles=[v[2] for v in builder]

dd={}
translation={}
# titles=[]
i=0
# for name, items in files.items():
#     nets=items[1]
for name, net, _ in builder:
    print name,net
#     for net in nets:
        
    if name not in d.keys():
        continue
    s='Net_{:0>2}'.format(i)
    dd[s]=d[name][net]
    translation[s]=name+'_'+net
#     titles.append(name+'_'+net)
    i+=1 
pp(dd)
pp(translation)
scale=1
figs=[]
fun_call=[show_heat_map, show_variability_several]

for iFig in range(2):
    fig, axs=ps.get_figure2(n_rows=9, 
                            n_cols=10,
                            w=72/2.54*17.6*scale,
                            h=72/2.54*17.6*scale*1.1,  
                            fontsize=7*scale,
                            title_fontsize=7*scale,
                            gs_builder=gs_builder) 
    
    for ax in axs:
        ax.tick_params(direction='in',
                       length=0,
                       width=0.5,
    #                        pad=1,
                        top=False, right=False,
                        left=False, bottom=False,
                        ) 
        ax.tick_params(pad=2) 
    
    k={'axs':axs,
       'bbox_to_anchor':[2.2, 1.6],
       'do_colorbar':False, 
       'fig':fig,
       'models':['SN'],
       'print_statistics':False,
       'resolution':7,
       'titles':['']*len(builder),
       'fontsize_ax_titles':6*scale,
       'pos_ax_titles':1.08,
       'type_of_plot':'mean',
       'vlim_rate':[-100, 100], 
       'marker_size':7*scale,
       'threshold':14}
    
    fun_call[iFig](dd, 'mean_rate_slices', **k)
    
    
    if iFig==0:          
        
        im=axs[2].collections[0]
        box = axs[2].get_position()
        pos=[box.x0+0.0*box.width, 
             box.y0+box.height+box.height*2.1, 
             box.width*0.8, 
             0.01]
        axColor=pylab.axes(pos)
        #     axColor = pylab.axes([0.05, 0.9, 1.0, 0.05])
        cbar=pylab.colorbar(im, cax = axColor, orientation="horizontal")
        cbar.ax.set_title('Contrast (Hz)')#, rotation=270)
        cbar.ax.tick_params( length=1, ) 
        cbar.set_ticks([-90,0,90])

        axs[0].legend(['Dual selection','Selection', 'No selection'], 
        #               ncol=1, 
                  scatterpoints=1,
                  frameon=False,
                  labelspacing=0.1,
                  handletextpad=0.1,
                  columnspacing=0.3,
                  bbox_to_anchor=[2.3, 2.],
                  markerscale=1.5
        #           prop={'size':24}
                  )
    
    
    
    for i, ax in enumerate(axs): 
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        
        if i==2+5:
            ax.text(0.5, -.35, 
                      'Cortical input action 1',
                        horizontalalignment='center', 
                        transform=ax.transAxes) 
#             ax.set_xticks([1,1.5, 2, 2.5])
        
        if i==28+6:
            ax.text(0.5, -.35, 
                      'Cortical input action 1',
                        horizontalalignment='center', 
                        transform=ax.transAxes) 
#             ax.set_xticks([1,1.5, 2, 2.5])
                
        if i==0:
            ax.text(-.3, 0.55,'Cortical input action 2',
                    horizontalalignment='center', transform=ax.transAxes,
                    rotation=90) 

        if i==11+6:
            ax.text(-.3, -0.1,'Cortical input action 2',
                    horizontalalignment='center', va='center',transform=ax.transAxes,
                    rotation=90) 
        
        
        if i<len(titles):
    #         axs[i].text(0.5, 1.28, titles[i],
    #                             horizontalalignment='center', 
    #                             transform=axs[i].transAxes,
    #                             fontsize=k.get('fontsize_ax_titles',24)) 
            axs[i].text(0.5, 1.1, titles[i],
                                horizontalalignment='center', 
                                transform=axs[i].transAxes,
                                fontsize=k.get('fontsize_ax_titles',24)) 
        a=ax.get_xticklabels()
        ax.tick_params(axis='both', which='major', labelsize=20)
    #     ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    #     ax.set_yticklabels(fontsize=20)
    
        ax.my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True)

    figs.append(fig)

save_figures(figs, 
             __file__.split('/')[-1][0:-3]+'/data',
             dpi=200)

pylab.show()