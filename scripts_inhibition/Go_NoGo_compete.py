'''
Created on 25 mar 2014

@author: mikael
'''
import matplotlib.gridspec as gridspec
import numpy
import pylab
import toolbox.plot_settings as ps


from os.path import expanduser
from simulate import (main_loop, show_fr, get_file_name, 
                      get_file_name_figs, get_path_nest)

from toolbox.data_to_disk import Storage_dic
from toolbox import misc
from toolbox.network import manager
from toolbox.network.manager import (add_perturbations, get_storage_list)
# from toolbox.network.manager import Builder_Go_NoGo_with_lesion as Builder
from toolbox.network.manager import Builder_Go_NoGo_with_lesion_FS as Builder
from toolbox.my_signals import Data_bar

import pprint
pp=pprint.pprint

THREADS=4

def get_kwargs_builder(**k_in):
    
    res=k_in.get('resolution',5)
    rep=k_in.get('repetition',5)
    sub=k_in.get('sub_sampling',6.25)
    laptime=k_in.get('laptime',1500.0)
    duration=k_in.get('duration',[1000.,500.])
    prop_conn=k_in.get('proportion_connected', 1)
    
    # Parameers if oscillations is used
    amp_base=k_in.get('amp_base')
    freqs=k_in.get('freqs')
    freq_oscillations=k_in.get('freq_oscillations')

    input_type=k_in.get('input_type','burst3')
    STN_amp_mod=k_in.get('STN_amp_mod') 
    d= {'amp_base':amp_base,
        'duration':duration,
        'freqs':freqs,
        'freq_oscillations':freq_oscillations,
        'input_type':input_type,
        'local_num_threads':THREADS,
        'print_time':False,
        'proportion_connected':prop_conn,
        'resolution':res, 
        'repetition':rep,
        'save_conn':{'overwrite':False},
        'sim_time':laptime*res*res*rep, 
        'sim_stop':laptime*res*res*rep, 
        'size':4000.0, 
        'start_rec':0.0,  
        'stop_rec':numpy.inf,
        'sub_sampling':sub,
        'STN_amp_mod':STN_amp_mod, 
        } 
    
    k_in.update(d)
    return  k_in
    
def get_kwargs_engine():
    return {'verbose':True}

def get_networks(builder, k_builder, k_director):
    info, nets, builder=manager.get_networks(builder,
                                             get_kwargs_builder(**k_builder),
                                             k_director,
                                             get_kwargs_engine())
    
    intervals=builder.dic['intervals']
    rep=builder.dic['repetitions']
    x=builder.dic['x']
    y=builder.dic['y'] 
    
    return info, nets, intervals, rep, x, y

    
def classify(x, y, threshold):
    if (x < threshold) and (y < threshold):
        return 0
    if (x >= threshold):
        return 1
    if (y >= threshold):
        return 2
    else:
        return 3
    
def process_data(data, threshold=5):
    
    outs=[]
    for d in data:
        x,y=d[2][0]['y'], d[2][1]['y'] 
        outs.append(numpy.abs(x-y))
    return outs

def show_neuron_numbers(d, models, **k):
    attr='firing_rate'    
    linestyle=['-','--']
    
    labels=k.pop('labels', models) 
    colors=misc.make_N_colors('jet', max(len(labels), 1))
    
    fig, axs=ps.get_figure(n_rows=1, n_cols=1, w=500.0, h=500.0, fontsize=10)    
    l_ids=[]
    ax=axs[0]
    for k, model in enumerate(models):

        max_set=0
        
        for j, name in enumerate(sorted([d.keys()[0]])):
            v=d[name]
            sets=[s for s in sorted(v.keys()) if s[0:3]=='set']
            
            ids=0
            for i, _set in enumerate(sets):
                
                if not model in v[_set].keys():
                    break
                
                obj=v[_set][model][attr]
                
                ids+=len(obj.ids)
                if max_set<=i:
                    max_set+=1            
        l_ids.append(ids)
    
    
     
    obj=Data_bar(**{'y':numpy.array(l_ids)})   
    obj.bar(ax)

    ax.set_xticklabels(labels)
    return fig
def show_bulk(d, models, attr, **k):
    
#     attr='mean_rate_slices'    
    linestyle=['-','--']
    
    labels=k.pop('labels', sorted(d.keys()))  
    colors=misc.make_N_colors('jet', max(len(labels), 1))
    
    fig, axs=ps.get_figure(n_rows=7, n_cols=1, w=1200.0, h=800.0, fontsize=10)    
    
    for k, model in enumerate(models):        
        ax=axs[k]
#         max_set=0
        for j, name in enumerate(sorted(d.keys())):
            v=d[name]
            sets=[s for s in v.keys() if s[0:3]=='set']
            for i, _set in enumerate(sets):
                
                if not model in v[_set].keys():
                    break
                
                obj=v[_set][model][attr]
                obj.plot(ax, **{'color':colors[j],
                                'linestyle':linestyle[i],
                                'label':labels[j]
                                })
#                 if max_set<=i:
#                     max_set+=1
        
        max_set=len(sets)
        ax.set_ylabel(model+' (spikes/s)')
        
        #Get artists and labels for legend and chose which ones to display
        if k!=0:
            continue    
        handles, _labels = ax.get_legend_handles_labels()
        display = range(0,len(d.keys())*max_set, max_set)
            

        #Create custom artists
        linetype_labels=[]
        linetype_handles=[]
        for i in range(max_set):
            linetype_handles.append(pylab.Line2D((0,1),(0,0),
                                                 color='k', 
                                                 linestyle=linestyle[i]))
            linetype_labels.append('Action '+str(i))

                
        #Create legend from custom artist/label lists
        ax.legend([handle for i,handle in enumerate(handles) 
                   if i in display]+linetype_handles,
          [label for i,label in enumerate(_labels) 
                   if i in display]+linetype_labels, bbox_to_anchor=(1.22, 1.))
        
    return fig       
                
     

def show_3d(d,attr,**k):
    models=['SN']
    res=k.get('resolution')
    titles=k.get('titles')
    n=len(models)
    m=len(d.keys())
#     attr='mean_rate_slices'
    fig, axs=ps.get_figure(n_rows=m, n_cols=1, w=500.0, h=800.0, fontsize=12, 
                           projection='3d')        
     
    i=0
    
    for model in models:
        alpha=0.8
        dcm={'Net_0':'jet',
             'Net_1':'coolwarm',}
        for key in sorted(d.keys()):
            obj0=d[key]['set_0'][model][attr]
            obj1=d[key]['set_1'][model][attr]
            args=[obj0.x_set, obj1.x_set,
                  # obj1.y-obj0.y, 
                  numpy.mean(obj1.y_raw_data-obj0.y_raw_data, axis=0),
                  numpy.std(obj1.y_raw_data-obj0.y_raw_data, axis=0)]
            for j, arg in enumerate(args):
                arg.shape
                args[j]=numpy.reshape(arg, [res,res])
            x,y,z, z_std=args
             
            axs[i].plot_surface(x, y, z, cmap='coolwarm', rstride=1, cstride=1, 
                                linewidth=0, 
                                shade=True,
                                alpha=alpha,
                                antialiased=False,
                                vmin=-40,
                                vmax=40)
            axs[i].set_zlim([-40, 40])
#             axs[i].set_zlabel('SNr firing rate (Hz)')
#             axs[i].set_xlabel('CTX increase A1 (proportion)')
#             axs[i].set_ylabel('CTX increase A2 (proportion)')
            
            
            axs[i].set_title(titles[i])
#             axs[i+1].plot_surface(x, y, z_std, cmap='coolwarm', rstride=1, cstride=1, 
#                                 linewidth=0, 
#                                 shade=True,
#                                 alpha=alpha,
#                                 antialiased=False)
             
             
    #                 alpha-=0.3
            i+=1
    #                 pylab.show()
    #                 print v
               
    for ax in axs:
        ax.view_init(elev=15)
    
    return fig




def selection_marker(d0,d1, x2, y2, thr, marker):
    if marker=='n':
        m=(d0 > thr)*(d1 > thr)==False
        x = numpy.ma.array(x2, mask=m)
        y = numpy.ma.array(y2, mask=m) #             x0,y0,z0=get_optimal(z.shape[0])
        color='k'
        marker='-'
        nbr=0

    elif marker=='d':
        m=(d0 <= thr)*(d1 <= thr)==False
        x = numpy.ma.array(x2, mask=m)
        y = numpy.ma.array(y2, mask=m) #             x0,y0,z0=get_optimal(z.shape[0])
        color='k'
        marker='+'
        nbr=1
#     print m
#     print x
    
    elif marker=='s':
        m=((d0 > thr)*(d1 <= thr)+(d0 <= thr)*(d1 > thr))==False
        x = numpy.ma.array(x2, mask=m)
        y = numpy.ma.array(y2, mask=m) #             x0,y0,z0=get_optimal(z.shape[0])
        color='k'
        nbr=2

    elif marker=='1':
        m=((d0 <= thr)*(d1 > thr))==False
        x = numpy.ma.array(x2, mask=m)
        y = numpy.ma.array(y2, mask=m) #             x0,y0,z0=get_optimal(z.shape[0])
        color='k'
        nbr=2
        
    elif marker=='2':
        m=((d0 > thr)*(d1 <= thr))==False
        x = numpy.ma.array(x2, mask=m)
        y = numpy.ma.array(y2, mask=m) #             x0,y0,z0=get_optimal(z.shape[0])
        color='k'
        nbr=3
        
    return x,y,color, marker, nbr, m



def set_action_selection_marker( i, d0, d1, x2, y2, thr, marker,**k):
    axs=k.get('axs')
    
    
    x, y, color, marker=selection_marker(d0, d1, x2, y2, thr, marker)
#     for i,j in zip(x,y):
    axs[i].scatter(x, y, color=color, 
                    edgecolor=color,linewidth=0.1,
                    s=k.get('marker_size',10), 
#                    linewidths =0.05,
#                    mew=10, ms=20,
                   marker=r'$'+marker+'$')



def show_rate_all(d, kw):
        fig, axs=ps.get_figure2(**kw.get('fig_and_axes'))
#         pylab.show()
        show_fr(d, ['M1', 'M2', 'GA', 'GI', 'SN'], axs, **kw)
        axs[0].set_ylim([0,6])
        axs[1].set_ylim([0,6])
        return fig
    
def show_rate_D1_D2_SNR(d, d_plot_fr2):
        fig, axs=ps.get_figure2(**d_plot_fr2.get('fig_and_axes'))

        show_fr(d, ['M1', 'M2', 'SN'], axs, **d_plot_fr2)
        
        for i, s, c0, c1, rotation in [
                                   [1, 'Firing rate (Hz)', -0.45, 0.5, 90],
                                   [0, r'$D1$', -0.25, 0.5, 90],
                                   [1, r'$D2$', -0.25, 0.5, 90],
                                   [2, r'SNr', -0.25, 0.5, 90],
                                   ]:
                                   
            axs[i].text(c0, c1, s, 
                        fontsize=7,
                        transform=axs[i].transAxes,
                        verticalalignment='center', 
                        horizontalalignment='center', 
                        rotation=rotation) 
        
        axs[0].legend(axs[0].lines[0:6], 
                  ['Action 1', 'Action 2'], 
                  bbox_to_anchor=(1.15, 1.85), 
                  ncol=2,
                  handletextpad=0.1,
                  
                  frameon=False,
                  columnspacing=0.3,
                  labelspacing=0.2) 
        
        for i, ax in enumerate(axs):      
            
            ax.my_set_no_ticks(xticks=3, yticks=2)
            if i==2:
                ax.set_yticks([0,60])
                ax.set_xlabel('')
                ax.text(0.5,-0.65,'Time (ms)',
                        fontsize=7,
                        transform =ax.transAxes,
                        ha='center',va='center') 
            if i==1:
                ax.set_yticks([0,20])
# 
            if i==0:
                ax.set_yticks([0,30])#           

            ax.set_ylabel('')
            axs[0].my_remove_axis(xaxis=True, yaxis=False,
                                  keep_ticks=True) 
            axs[1].my_remove_axis(xaxis=True, yaxis=False,
                                  keep_ticks=True)
            
        for ax in axs:
            ax.tick_params(direction='out',
                           length=1,
                           width=0.5,
                            top=False, right=False
                            ) 
        #think you have to to this after toget it to work
        for ax in axs:
            ax.tick_params(
                           pad=1,
                            ) 
        
            
        for i, ax in enumerate(axs[1:3]):
            if not ax.legend():
                continue
            ax.legend().set_visible(False)         
            
        return fig

def gs_builder(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.5 ), 
              hspace=kwargs.get('hspace', 0.6 ))

    iterator = [[0, slice(0,3)],
                [0, slice(4,7)],
                [1, slice(0,3)],
                [1, slice(4,7)],
                [2, slice(0,3)],
                [2, slice(4,7)]]
    
    return iterator, gs, 

def gs_builder_var(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.05 ), 
              hspace=kwargs.get('hspace', 0.05 ))

    iterator = [[slice(0,10), slice(0,10)]]
    for i in range(0,10):
        for j in range(12,22):
            iterator+=[[slice(i,i+1), slice(j,j+1)]]

    
    return iterator, gs, 

def gs_builder_var_mul(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    grid=kwargs.get('grid')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.05 ), 
              hspace=kwargs.get('hspace', 0.05 ))

    iterator = []
    
    
    def add_to_grid(pos, res, iterator):
        for i in range(pos[0]*(res+2), pos[0]*(res+2)+res):
            for j in range(pos[1]*(res+2),pos[1]*(res+2)+res):
                iterator+=[[slice(i,i+1), slice(j,j+1)]]

    for p0 in range(grid[0]):
        for p1 in range(grid[1]):
            add_to_grid([p0,p1], 10, iterator)
    
    return iterator, gs, 

def gs_builder_all(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.5 ), 
              hspace=kwargs.get('hspace', 0.6 ))

    for i in range(3):
        for j in range(3):
            iterator = [[i, slice(0+j*8,3+j*8)],
                        [i, slice(4+j*8,7+j*8)]]
#             ,
#                         [0, slice(4,7)],
#                         [1, slice(0,3)],
#                         [1, slice(4,7)],
#                         [2, slice(0,3)],
#                         [2, slice(4,7)]]
    
    return iterator, gs,


def gs_builder3(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.5 ), 
              hspace=kwargs.get('hspace', 0.6 ))

    iterator=[]
    for i in range(n_rows):
        iterator += [[slice(i,i+1), slice(0,1)]]
        
    return iterator, gs,


def show_variability_several(d, attr, nets=['Net_00'],  **k):
    do_colorbar=k.get('do_colorbar',True)
    model='SN'
    print_statistics=k.get('print_statistics', True)
    res=k.get('resolution')
    titles=k.get('titles')


    axs=k.get('axs')
    fig=k.get('fig')
            
    for ax in axs[1:]: 
        ax.my_remove_axis(xaxis=True, yaxis=True)    
    
     
    i=0

    m=len(d.keys())
    
    for net in nets:
        key=net
        i=int(key.split('_')[1])
        obj0=d[key]['set_0'][model][attr]
        obj1=d[key]['set_1'][model][attr]
        
        
        args=[obj0.x_set, obj1.x_set]
        for j, arg in enumerate(args):
            arg.shape
            args[j]=numpy.reshape(arg, [res,res])
        x,y=args
        
        
        thr=k.get('threshold',14)
        stepx=(x[0,-1]-x[0,0])/res
        stepy=(y[-1,0]-y[0,0])/res
    
        x2,y2=numpy.meshgrid(numpy.linspace(x[0,0]+stepx/2, 
                                            x[0,-1]-stepx/2, res),
                             numpy.linspace(y[0,0]+stepy/2, 
                                            y[-1,0]-stepy/2, res))     
        
        x_raw=obj0.y_raw_data
        y_raw=obj1.y_raw_data
        x_mean=numpy.mean(x_raw, axis=0)
        y_mean=numpy.mean(y_raw, axis=0)
        
        matrix=numpy.ones([x_raw.shape[0]+1, len(x), len(y)])*-1
        for m in ['d', 'n', '1', '2']:
            
            x_mean=numpy.reshape(x_mean, [res,res])
            y_mean=numpy.reshape(y_mean, [res,res])
            x, y, col, mar, nbr, bol=selection_marker(x_mean, y_mean, x2, y2, thr, m)
            matrix[0, bol==False]=nbr
        
        index=1
        for trace0, trace1 in zip(x_raw, y_raw):
            for m in ['d', 'n', '1', '2']:
                trace0=numpy.reshape(trace0, [res,res])
                trace1=numpy.reshape(trace1, [res,res])
                i, j, _,_,  nbr, bol=selection_marker(trace0, trace1, 
                                              x2, y2, thr, m)
                matrix[index,bol==False]=nbr
            index+=1
        
        
        iAxis=0
        for i in reversed(range(k.get('resolution'))):
            for j in range(k.get('resolution')):
                
                
                sizes=numpy.bincount( list(matrix[1:,i,j]), minlength=4)
    
                colors=['w', 'k', 'b', 'r']
                labels=['-', '+', '1', '2']
                explode = (0, 0, 0, 0) 
                
                if i==k.get('resolution')-1 and j==k.get('resolution')-1:
                    patches, texts=axs[index].pie([1]*4, explode=explode, 
                                              colors=colors, shadow=True,) 
                    axs[iAxis].legend(patches, labels,
                                 bbox_to_anchor=[9.,1.])
                
                if numpy.all(sizes[1:]==0):stop=1
                elif numpy.all(sizes[2:]==0):stop=2
                elif numpy.all(sizes[3:]==0):stop=3
                else:stop=4
                
                patches, _=axs[index].pie(sizes[0:stop], explode=explode[0:stop], 
                                            colors=colors[0:stop], shadow=False, )
                index+=iAxis   
    
    return fig

def show_variability(d, attr, net='Net_00',  **k):
    do_colorbar=k.get('do_colorbar',True)
    model='SN'
    print_statistics=k.get('print_statistics', True)
    res=k.get('resolution')
    titles=k.get('titles')
    vlim_variance=k.get('vlim_variance')
    vlim_CV=k.get('vlim_CV')
    vlim_rate=k.get('vlim_rate')

    axs=k.get('axs')
    fig=k.get('fig')
            
    for ax in axs[1:]: 
        ax.my_remove_axis(xaxis=True, yaxis=True)    
    

    type_of_plot=k.get('type_of_plot', 'mean')
     
    i=0
    performance={}
    m=len(d.keys())
    

    key=net
    i=int(key.split('_')[1])
    obj0=d[key]['set_0'][model][attr]
    obj1=d[key]['set_1'][model][attr]
    
    
    args=[obj0.x_set, obj1.x_set]
    for j, arg in enumerate(args):
        arg.shape
        args[j]=numpy.reshape(arg, [res,res])
    x,y=args
    
    
    thr=k.get('threshold',14)
    stepx=(x[0,-1]-x[0,0])/res
    stepy=(y[-1,0]-y[0,0])/res
    x1,y1=numpy.meshgrid(numpy.linspace(x[0,0], x[0,-1], res+1),
                       numpy.linspace(y[0,0], y[-1,0], res+1))
    x2,y2=numpy.meshgrid(numpy.linspace(x[0,0]+stepx/2, 
                                        x[0,-1]-stepx/2, res),
                         numpy.linspace(y[0,0]+stepy/2, 
                                        y[-1,0]-stepy/2, res))     
    
    
    
    xy_diff=numpy.mean(obj1.y_raw_data-obj0.y_raw_data, axis=0)
    xy_diff=numpy.reshape(xy_diff, [res,res])
    
    im = axs[0].pcolor(x1, y1, xy_diff, cmap='coolwarm', 
                               vmin=-90, vmax=90)


    x_raw=obj0.y_raw_data
    y_raw=obj1.y_raw_data
    x_mean=numpy.mean(x_raw, axis=0)
    y_mean=numpy.mean(y_raw, axis=0)
    
    
    dmarkers={}
    matrix=numpy.ones([x_raw.shape[0]+1, len(x), len(y)])*-1
    for m in ['d', 'n', '1', '2']:
        
        x_mean=numpy.reshape(x_mean, [res,res])
        y_mean=numpy.reshape(y_mean, [res,res])
        x, y, col, mar, nbr, bol=selection_marker(x_mean, y_mean, x2, y2, thr, m)
        
        
        axs[0].scatter(x, y, color=col, 
                edgecolor=col,linewidth=0.1,
                s=k.get('marker_size',10*k.get('scale',1)), 
    #                    linewidths =0.05,
    #                    mew=10, ms=20,
               marker=r'$'+mar+'$')
        
        
        
        matrix[0, bol==False]=nbr
    axs[0].set_xlim([1, 3])
    axs[0].set_ylim([1, 3]) 
    
    
    index=1
    for trace0, trace1 in zip(x_raw, y_raw):
        for m in ['d', 'n', '1', '2']:
            trace0=numpy.reshape(trace0, [res,res])
            trace1=numpy.reshape(trace1, [res,res])
            i, j, _,_,  nbr, bol=selection_marker(trace0, trace1, 
                                          x2, y2, thr, m)
            matrix[index,bol==False]=nbr
        index+=1
    
    
    index=1
    for i in reversed(range(k.get('resolution'))):
        for j in range(k.get('resolution')):
            
            
            sizes=numpy.bincount( list(matrix[1:,i,j]), minlength=4)

            colors=['w', 'k', 'b', 'r']
            labels=['-', '+', '1', '2']
            explode = (0, 0, 0, 0) 
            
            if i==k.get('resolution')-1 and j==k.get('resolution')-1:
                patches, texts=axs[index].pie([1]*4, explode=explode, 
                                          colors=colors, shadow=True,) 
                axs[index].legend(patches, labels,
                             bbox_to_anchor=[9.,1.])
            
            if numpy.all(sizes[1:]==0):stop=1
            elif numpy.all(sizes[2:]==0):stop=2
            elif numpy.all(sizes[3:]==0):stop=3
            else:stop=4
            
            patches, _=axs[index].pie(sizes[0:stop], explode=explode[0:stop], 
                                        colors=colors[0:stop], shadow=False, )
            index+=1    
    
    return fig

def show_heat_map(d, attr, **k):
    do_colorbar=k.get('do_colorbar',True)
    models=['SN']
    print_statistics=k.get('print_statistics', True)
    res=k.get('resolution')
    titles=k.get('titles')
    vlim_variance=k.get('vlim_variance')
    vlim_CV=k.get('vlim_CV')
    vlim_rate=k.get('vlim_rate')

    axs=k.get('axs')
    fig=k.get('fig')
    if not axs or not fig:

        fig, axs=ps.get_figure2(n_rows=3, 
                                n_cols=8,  
                                w=780/(24./7.),
                                h=910/(24./7.),  
                                fontsize=7,
                                title_fontsize=7,
                                gs_builder=gs_builder) 
  
        k['fig']=fig
        k['axs']=axs
  
    type_of_plot=k.get('type_of_plot', 'mean')
     
    i=0
    performance={}
    m=len(d.keys())
    
    for model in models:

        for key in sorted(d.keys()):
            print key
            i=int(key.split('_')[1])
            obj0=d[key]['set_0'][model][attr]
            obj1=d[key]['set_1'][model][attr]
            args=[obj0.x_set, obj1.x_set,
                  
                  numpy.mean(obj1.y_raw_data-obj0.y_raw_data, axis=0),
                  numpy.std(obj1.y_raw_data-obj0.y_raw_data, axis=0),
                  numpy.mean(obj0.y_raw_data, axis=0),
                  numpy.mean(obj1.y_raw_data, axis=0)]
            
            for j, arg in enumerate(args):
                arg.shape
                args[j]=numpy.reshape(arg, [res,res])
            x, y, z, z_std, d0, d1=args
            if type_of_plot=='variance':
                z=z_std
                _vmin, _vmax=vlim_variance

            elif type_of_plot=='CV':
                z=z_std/numpy.abs(z)
                _vmin, _vmax=vlim_CV
            else:
                _vmin, _vmax=vlim_rate
            
            
            stepx=(x[0,-1]-x[0,0])/res
            stepy=(y[-1,0]-y[0,0])/res
            x1,y1=numpy.meshgrid(numpy.linspace(x[0,0], x[0,-1], res+1),
                               numpy.linspace(y[0,0], y[-1,0], res+1))
            x2,y2=numpy.meshgrid(numpy.linspace(x[0,0]+stepx/2, 
                                                x[0,-1]-stepx/2, res),
                                 numpy.linspace(y[0,0]+stepy/2, 
                                                y[-1,0]-stepy/2, res))
            
            thr=k.get('threshold',14)
            
            im = axs[i].pcolor(x1, y1, z, cmap='coolwarm', 
                               vmin=_vmin, vmax=_vmax)
        
           
            for m in ['d', 's', 'n']:
                set_action_selection_marker(i,  d0, d1, 
                                                  x2, y2, thr, 
                                                 marker=m, 
                                                 **k)
             
            performance[key] = numpy.round(numpy.sum(numpy.abs(z)), 2)
            
            box = axs[i].get_position()
            

            axs[i].set_xlabel('Action 1')
            axs[i].set_ylabel('Action 2')
            axs[i].set_xlim([x[0,0], x[0,-1]])
            axs[i].set_ylim([y[0,0], y[-1,0]])
            # create color bar
            
            if key=='Net_0' and do_colorbar:
                label='Contrast (spike/s)'
                box = axs[0].get_position()
                axColor = pylab.axes([box.x0 + box.width * 2.56, 
                                      box.y0+box.height*0.1, 
                                      0.02, 
                                      box.height*0.8])
                cbar=pylab.colorbar(im, cax = axColor, orientation="vertical")
                cbar.ax.set_ylabel(label, rotation=270)
                cbar.set_ticks([_vmin,_vmax])
#                 from matplotlib import ticker
#                 
#                 tick_locator = ticker.MaxNLocator(nbins=3)
#                 cbar.locator = tick_locator
#                 cbar.update_ticks()
            
            axs[i].text(0.5, k.get('pos_ax_titles',1.05) , titles[i],
                        horizontalalignment='center', 
                        transform=axs[i].transAxes,
                        fontsize=k.get('fontsize_ax_titles',7)) 
#             axs[i].set_title(titles[i])

#             i+=1
    if print_statistics:
        ax=axs[-1]
        import pprint
        ax.text( 0.1, 0.1, pprint.pformat(performance), 
                       transform=ax.transAxes, 
                fontsize=7)
        
    return fig, performance
def get_optimal(size):
    x,y=numpy.meshgrid(numpy.linspace(1,3, size+1),numpy.linspace(1,3, size+1))
    z=numpy.ones([size,size])*40
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if j==i:
                z[i,j]=0
            if j>i:
                z[i,j]=-z[i,j]
    return x,y,z
def plot_optimal(size):
    fig, axs=ps.get_figure(n_rows=1, n_cols=1, w=400.0*0.8, h=250.0*0.8, fontsize=24)
    x,y,z=get_optimal(size)
       
    im=axs[0].pcolor(x, y, z, cmap='coolwarm',  vmin=-40, vmax=40)
#           
    box = axs[0].get_position()
    axs[0].set_position([box.x0*1.05, box.y0, box.width*0.85, box.height])
 
    axs[0].set_xlabel('Action 1')
    axs[0].set_ylabel('Action 2')
    axs[0].my_set_no_ticks(xticks=3,yticks=3, )
            
    return fig
    
# def plot_uniformity(d,attr,**k):
#     models=['SN']
#     res=k.get('resolution')
#     titles=k.get('titles')
#     n=len(models)
#     m=len(d.keys())
# #     attr='mean_rate_slices'
#     fig, axs=ps.get_figure(n_rows=1, 
#                            n_cols=1, 
#                            w=600.0*0.65*2, 
#                            h=700.0*0.65*2, 
#                            fontsize=24,
#                            frame_hight_y=0.5, 
#                            frame_hight_x=0.6, 
#                            title_fontsize=20)        
#     
#     type_of_plot=k.get('type_of_plot', 'mean')
#      
#     i=0
#     performance={}  

def gs_builder2(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0. ), 
              hspace=kwargs.get('hspace', 0.5 ))

    iterator = [[slice(1,6),slice(1,4)],
                [slice(6,11),slice(1,4)],
                [slice(11,16),slice(1,4)],
#                 [slice(10,15),slice(10,16)],
               ]
    
    return iterator, gs,

class Setup(object):

    def __init__(self, **k):
        self.amp_base=k.get('amp_base')
        self.duration=k.get('duration',[1000.,500.])
        self.freqs=k.get('freqs')
        self.freq_oscillations=k.get('freq_oscillations')
        self.fr_xlim=k.get('fr_xlim',[0,10000])
        self.input_type=k.get('input_type','burst3')
        self.laptime=k.get('laptime',1500.0)
        self.labels=k.get('labels',['Only D1', 
                           'D1,D2',
                           'MSN lesioned (D1, D2)',
                           'FSN lesioned (D1, D2)',
                           'GPe TA lesioned (D1,D2)'])
        self.local_num_threads=k.get('local_num_threads',1)
        self.l_mean_rate_slices=k.get('l_mean_rate_slices', 
                                      ['mean_rate_slices', 
                                       'mean_rate_slices_0',
                                       'mean_rate_slices_1'])

        self.nets_to_run=k.get('nets_to_run',[])
        self.other_scenario=k.get('other_scenario',False)
        self.proportion_connected=k.get('proportion_connected',1)
        self.p_pulses=k.get('p_pulses')
        self.res=k.get('resolution',2)
        self.rep=k.get('repetition',2)
        self.STN_amp_mod=k.get('STN_amp_mod')
        self.time_bin=k.get('time_bin',5)
 
    def builder(self):
        d= {
            'amp_base':self.amp_base,
            'duration':self.duration,
            'freqs':self.freqs,
            'freq_oscillations':self.freq_oscillations,
            'input_lists': [['C1'],
                            ['C1', 'C2']],
            'input_type':self.input_type,
            'laptime':self.laptime,                
            'other_scenario':self.other_scenario,
            'proportion_connected':self.proportion_connected,
            'p_pulses':self.p_pulses,
            'repetition':self.rep,
            'resolution':self.res,
            'STN_amp_mod':self.STN_amp_mod,
#             'sub_sampling':6.25,            
            }
        return d

    def director(self):
        return {'nets_to_run':self.nets_to_run}

    def firing_rate(self):
        d={'average':False, 
           'sets':[0,1],
#            't_stop':50000.,
           'time_bin':self.time_bin,
           'local_num_threads':self.local_num_threads,
           'proportion_connected':self.proportion_connected}
        return d
    
    def mean_rate_slices(self):
        d={'local_num_threads':self.local_num_threads}
        return d

    def plot_fr(self):
        d={'win':10.,
           'by_sets':True,
           't_start':3000.0,
           't_stop':6000.0,#self.res*self.res*self.rep*self.laptime,#10000.0,
           
           'fig_and_axes':{'n_rows':5, 
                        'n_cols':1, 
                        'w':72*17.6/2.54*2, 
                        'h':72*17.6/2.54*3/4., 
                        'fontsize':7,
                        'frame_hight_y':0.5,
                        'frame_hight_x':0.7,
                        'title_fontsize':7,
                        'font_size':7,
                        'text_fontsize':7,
                        'linewidth':1.,
                        'gs_builder':gs_builder3},}
        return d
    
    def plot_fr2(self):
        d={'win':10.,
           'by_sets':True,
           't_start':self.fr_xlim[0],
           't_stop':self.fr_xlim[1],
           'labels':['Action 1', 'action 2'],
           'fig_and_axes':{'n_rows':18, 
                        'n_cols':4, 
                        'w':int(72*11.6/2.54*(1./3))*0.9, 
                        'h':80, 
                        'fontsize':7,
                        'title_fontsize':7,
                        'font_size':20,
                        'text_fontsize':7,
                        'linewidth':.5,
                        'gs_builder':gs_builder2},
           }
        return d    
    
        
 
    def plot_3d(self):
        if self.other_scenario:
            vlim_variance=[0.,7]
            vlim_CV=[0.,3]
            vlim_rate=[-90.,90]
        
        else:
            vlim_variance=[0.,7]
            vlim_CV=[0.,3]
            vlim_rate=[-40.,40]
        
        
        d={'resolution':self.res,
           'titles':self.labels,
           'vlim_variance':vlim_variance,
           'vlim_CV':vlim_CV,
           'vlim_rate':vlim_rate}
        
        return d

        
    def plot_bulkt(self):
        d={'labels':self.labels}
        return d
    
    def get_l_mean_rate_slices(self):
        return self.l_mean_rate_slices

def simulate(builder, from_disk, perturbation_list, script_name, setup):
    home = expanduser("~")
    

   
#     file_name = get_file_name(script_name)
#     file_name_figs = get_file_name_figs(script_name)
     
    d_firing_rate = setup.firing_rate()
    d_mrs = setup.mean_rate_slices()
    
    attr = [
            'firing_rate',
            'mean_rate_slices'
            ]
    
    models = ['M1', 'M2', 'FS', 'ST', 'GA', 'GI', 'SN']
    sets = ['set_0', 'set_1']
    
    info, nets, intervals, rep, x_set, y_set = get_networks(builder, 
                                                            setup.builder(),
                                                            setup.director())

    add_perturbations(perturbation_list, nets)
    
    
    print nets.keys()
    key=nets.keys()[0]
    file_name = get_file_name(script_name, nets[key].par)
    file_name_figs = get_file_name_figs(script_name,  nets[key].par)
    path_nest=get_path_nest(script_name, nets.keys(), nets[key].par)

    for net in nets.values():
        net.set_path_nest(path_nest)

    x=range(len(x_set))
    xticklabels=[str(a)+'*'+str(b) for a, b in zip(x_set, y_set) ]
    d_mrs.update({'intervals':intervals[1], 
                  'repetition':rep, 
                  'set_0':{'x':x, 'xticklabels':xticklabels, 'x_set':x_set}, 
                  'set_1':{'x':x, 'xticklabels':xticklabels, 'x_set':y_set}, 
                  'sets':[0, 1]})
    
#     d_firing_rate['mean_rate_slices']=d_mrs
#     
#     d=d_mrs.copy()
#     d['intervals']=[[v[0], v[0]+100] for v in intervals[1]]
#     d_firing_rate['mean_rate_slices_0']=d
#  
#     d=d_mrs.copy()
#     d['intervals']=[[v[1]-100, v[1]] for v in intervals[1]]
#  
#     d_firing_rate['mean_rate_slices_1']=d
#     
    kwargs_dic = {
                'firing_rate':d_firing_rate, 
                'mean_rate_slices': d_mrs
                  }

    
    # Adding nets no file name
    sd_list=get_storage_list(nets, file_name, info)

    from_disks, d = main_loop(from_disk, attr, models, 
                              sets, nets, kwargs_dic, sd_list,
                              **{'attrs_load':['mean_rate_slices']})
     
    return file_name, file_name_figs, from_disks, d, models


def create_figs(setup, file_name_figs, d, models):
    
    sd_figs = Storage_dic.load(file_name_figs)
    figs = []
    d_plot_3d=setup.plot_3d()

    
    d_plot_fr = setup.plot_fr()
    d_plot_fr2 = setup.plot_fr2()
    
    d_plot_bulk=setup.plot_bulkt()
    l_mean_rate_slices=setup.get_l_mean_rate_slices()
    
    
    figs.append(plot_optimal(setup.res))


    fig, axs=ps.get_figure2(n_rows=3, 
                            n_cols=8,  
                            w=780/(24./7.),
                            h=910/(24./7.),  
                            fontsize=7,
                            title_fontsize=7,
                            gs_builder=gs_builder) 

    pp(d)
    for name in l_mean_rate_slices:
        d_plot_3d['type_of_plot']='mean'
        fig, perf=show_heat_map(d, name,  **d_plot_3d)
        figs.append(fig)
   
    for name in l_mean_rate_slices:
        d_plot_3d['type_of_plot']='variance'
          
        fig, perf=show_heat_map(d, name,  **d_plot_3d)
        figs.append(fig)
  
    for name in l_mean_rate_slices:
        d_plot_3d['type_of_plot']='CV'
           
        fig, perf=show_heat_map(d, name,  **d_plot_3d)
        figs.append(fig)
  
    for name in l_mean_rate_slices:   
        figs.append(show_bulk(d, models, name, **d_plot_bulk))
             
    for i in range(len(d.keys())):
        
        if i!=1 and 1<len(d.keys()):
            continue
        
        figs.append(show_rate_D1_D2_SNR(d['Net_'+str(i)], d_plot_fr2))
        figs.append(show_rate_all(d['Net_'+str(i)], d_plot_fr))
        
#         figs.append()
        
        
    sd_figs.save_figs(figs, format='png', dpi=400)
    sd_figs.save_figs(figs, format='svg', in_folder='svg')
    
class Main():    
    def __init__(self, **kwargs):
        self.kwargs=kwargs
    
    def __repr__(self):
        return self.kwargs['script_name']

    def do(self):
        main(**self.kwargs)    
    
    def get_nets(self):
        return self.kwargs['setup'].nets_to_run

    def get_script_name(self):
        return self.kwargs['script_name']

    def get_name(self):
        nets='_'.join(self.get_nets()) 
        script_name=self.kwargs['script_name']
        script_name=script_name.split('/')[1].split('_')[0:2]
        script_name='_'.join(script_name)+'_'+nets
        return script_name+'_'+str(self.kwargs['from_disk'])
        
def main(builder=Builder,
         from_disk=2,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3],
         setup=Setup(**{'local_num_threads':THREADS,
                        'resolution':5,
                        'repetition':5})):
    
    
    v=simulate(builder, from_disk, perturbation_list, script_name, setup)
    file_name, file_name_figs, from_disks, d, models= v
    
    if numpy.all(numpy.array(from_disks) > 1):
        create_figs(setup, file_name_figs, d, models)
    
    
#     pylab.show()
 
import unittest
class TestMethods(unittest.TestCase):     
    def setUp(self):
        from toolbox.network.default_params import Perturbation_list as pl
        from_disk=2
        
        import oscillation_perturbations4 as op
        
        rep, res=2, 3
        duration=[900.,100.0]
        laptime=1000.0
        p_size=0.1608005821
        ss=8.3
        l_mean_rate_slices= ['mean_rate_slices']
#         sim_time=rep*res*res*1500.0
      
        local_num_threads=16
        
        l=op.get()
        max_size=4000
        p=pl({'netw':{'size':int(p_size*max_size),
                      'sub_sampling':{'M1':ss,
                                      'M2':ss}},
              'simu':{'local_num_threads':local_num_threads}},
                  '=')
        p+=l[4+3] #data2
#         p+=l[4] #data        
#         p+=l[4+3] #data3
        
        self.setup=Setup(**{'duration':duration,
                            'laptime':laptime,
                            'other_scenario':True,
                            'l_mean_rate_slices':l_mean_rate_slices,
                            'local_num_threads':local_num_threads,
                            'resolution':res,
                            'repetition':rep})
        
        v=simulate(builder=Builder,
                            from_disk=from_disk,
                            perturbation_list=p,
                            script_name=(__file__.split('/')[-1][0:-3]
                                         +'/data2'),
                            setup=self.setup)
        
        file_name, file_name_figs, from_disks, d, models= v
        
        self.res=res
        self.file_name_figs=file_name_figs
        self.from_disks=from_disks
        self.d=d
        self.models=models
        

    def test_create_figs(self):
        create_figs(
                    self.setup,
                    self.file_name_figs, 
                    self.d, 
                    self.models)
        pylab.show()
    
#     def test_show_fr(self):
#         show_fr(self.d, self.models, **{'win':20.,
#                                         't_start':4000.0,
#                                         't_stop':5000.0})
#         pylab.show()
 

if __name__ == '__main__':
    test_classes_to_run=[
                         TestMethods
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)
    
    
    
    



    
