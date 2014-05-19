'''
Created on May 10, 2014

@author: mikael
'''

from toolbox import misc
import toolbox.plot_settings as ps
from toolbox.data_to_disk import Storage_dic

def cmp_mean_rates_intervals_sets(d, intervals, x, repetitions):
    kwargs={'intervals':intervals,
            'repetitions':repetitions}
    
    for keys, val in misc.dict_iter(d):
        
        val={}
        for j in [0,1]:
            v=val[:,j].get_mean_rate_slices(**kwargs)
            v.x=x[0: repetitions]
            val[j]=v
            
        d=misc.dict_recursive_add(d, (keys[0:-1]
                                      +['mean_rates_intervals']), val)
        
    return d

def cmp_mean_rates_intervals(d, intervals, x, repetitions):
    kwargs={'intervals':intervals,
            'repetitions':repetitions}
    
    for keys, val in misc.dict_iter(d):
        
        if not keys[-1] =='spike_signal':
            continue
        
        v=val.get_mean_rate_slices(**kwargs)
        v.x=x
            
        d=misc.dict_recursive_add(d, (keys[0:-1]
                                      +['mean_rates_intervals']), v)
        
    return d




def show_plot(name, d, models=['M1','M2','FS', 'GA', 'GI','ST', 'SN'], **k):
    fig, axs=ps.get_figure(n_rows=len(models), n_cols=1, w=1000.0, h=800.0, fontsize=10)  
    labels=k.get('labels', sorted(d.keys()))
    colors=misc.make_N_colors('jet', len(labels))
    linestyles=['-']*len(labels)
    
    j=0
    for k in sorted(d.keys()):
        v=d[k]
#         axs[0].set_title(k)
        
        for i, model in enumerate(models):
            
            v[model][name].plot(ax=axs[i], 
                                         **{'label':model+' '+labels[j],
                                            'linestyle':linestyles[j],
                                            'color':colors[j]})
        j+=1    
    
    return fig, axs

def show_coherence(d, models):
    fig, axs=show_plot('mean_coherence',d, models)
    for ax in axs:
        ax.set_xlim([0,50])
    return fig
def show_fr(d, models, **k):
    fig, _ =show_plot('firing_rate',d, models, **k)
    return fig

def show_mr(d, models, **k):
    fig, _ =show_plot('mean_rates_intervals',d, models, **k)
    return fig

def show_hr(d, models, **k):
    fig, _ =show_hist('mean_rates',d, models, **k)
    return fig

def show_hist(name, d, models=['M1','M2','FS', 'GA', 'GI','ST', 'SN'], **k):

    fig, axs=ps.get_figure(n_rows=7, n_cols=1, w=1000.0, h=800.0, fontsize=10)   
    labels=k.get('labels', sorted(d.keys()))
    del k['labels']
    colors=misc.make_N_colors('jet', len(labels))
    linestyles=['solid']*len(labels)
    linewidth=[2.0]*len(labels)
    j=0
    
    for key in sorted(d.keys()):
        v=d[key]
#         axs[0].set_title(k)
        
        for i, model in enumerate(models):
            if 'spike_stastistic' in v[model]:
                st=v[model]['spike_statistic']
                st.rates={'mean':round(st.rates['mean'],2),
                          'std':round(st.rates['std'],2),
                          'CV':round(st.rates['CV'],2)}
                s=str(st.rates)
            else:
                s=''
                
            k.update({'label':(model+' '+labels[j]+' ' +s),
                     'histtype':'step',
                     'linestyle':linestyles[j],
                     'color':colors[j],
                     'linewidth':linewidth[j]})
            
            h=v[model][name].hist(ax=axs[i],**k) 
            
            ylim=list(axs[i].get_ylim())
            ylim[0]=0.0
            axs[i].set_ylim(ylim)
            axs[i].legend_box_to_line()
        j+=1 
    return fig, axs

def show_phase_diff(d, models, **k):
    fig, _ =show_hist('phase_diff',d, models, **k)
    return fig

def show_psd(d, models):
    fig, axs=show_plot('psd',d, models)   
    for ax in axs:
        ax.set_xlim([0,50])
    return fig