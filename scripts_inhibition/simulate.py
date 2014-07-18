'''
Created on May 10, 2014

@author: mikael
'''

from toolbox import misc
from toolbox.network.manager import save, load, compute, run

import toolbox.plot_settings as ps

import pprint
pp=pprint.pprint

# def cmp_mean_rates_intervals_sets(d, intervals, x, repetitions):
#     kwargs={'intervals':intervals,
#             'repetitions':repetitions,
#             'x':x[0: repetitions]}
#     
#     for keys, val in misc.dict_iter(d):
#         
#         val={}
#         for j in [0,1]:
#             v=val[:,j].get_mean_rate_slices(**kwargs)
#             v.x=x[0: repetitions]
#             val[j]=v
#             
#         d=misc.dict_recursive_add(d, (keys[0:-1]
#                                       +['mean_rates_intervals']), val)
#         
#     return d
# 
# def cmp_mean_rates_intervals(d, intervals, x, repetitions,**k):
#     kwargs={'intervals':intervals,
#             'repetitions':repetitions}
#     
#     for keys, val in misc.dict_iter(d):
#         
#         if not keys[-1] =='spike_signal':
#             continue
#         
#         if 'sets' in k.keys():
#             for s in k['sets']:
#                 v=val[:,s].get_mean_rate_slices(**kwargs)
#                 v.x=x
#                 d=misc.dict_recursive_add(d, (keys[0:-1]
#                                       +['Set_'+str(s),'mean_rates_intervals']), v)
#         else:
#             v=val.get_mean_rate_slices(**kwargs)
#         
#         
#             v.x=x
#             d=misc.dict_recursive_add(d, (keys[0:-1]
#                                       +['mean_rates_intervals']), v)
#         
#     return d


def cmp_psd(d_pds, models, dd):
    for key1 in dd.keys():
        for model in models:
            psd=dd[key1][model]['firing_rate'].get_psd(**d_pds)
            dd[key1][model]['psd'] = psd


def get_file_name(script_name, home):
    file_name = home + '/results/papers/inhibition/network/' + script_name
    return file_name


def get_file_name_figs(script_name, home):
    file_name_figs = home + '/results/papers/inhibition/network/fig/' + script_name
    return file_name_figs



def main_loop(from_disk, attr, models, sets, nets, kwargs_dic, sd):
    d = {}
    from_disks = [from_disk] * len(nets.keys())
    for net, fd in zip(nets.values(), from_disks):
        if fd == 0:
            dd = run(net)
            save(sd, dd)
        elif fd == 1:
            filt = [net.get_name()] + models + ['spike_signal']
            dd = load(sd, *filt)
            dd = compute(dd, models, attr, **kwargs_dic)
            save(sd, dd)
        elif fd == 2:
            filt = [net.get_name()] + sets + models + attr
            dd = load(sd, *filt)
        d = misc.dict_update(d, dd)
    
    return from_disks, d

def show_plot(name, d, models=['M1','M2','FS', 'GA', 'GI','ST', 'SN'], **k):
    dd={}
    by_sets=k.pop('by_sets', False)
    for keys, val in misc.dict_iter(d):
        
        if keys[-1]!=name:
            continue
        if by_sets and keys[0][0:3]!='set':
            continue
        
        first_keys=keys[:-2]
        if type(first_keys)==str:
            first_keys=[first_keys]
        
        new_keys=['_'.join(first_keys)]+keys[-2:]
        
        dd=misc.dict_recursive_add(dd, new_keys, val)
        
    d=dd
    fig, axs=ps.get_figure(n_rows=len(models), n_cols=1, w=1000.0, h=800.0, fontsize=10)  
    labels=k.pop('labels', sorted(d.keys()))
#     colors=misc.make_N_colors('Paired', max(len(labels), 6))
    colors=misc.make_N_colors('jet', max(len(labels), 1))
    linestyles=['-']*len(labels)
    
    j=0
    for key in sorted(d.keys()):
        v=d[key]
#         axs[0].set_title(k)
        for i, model in enumerate(models):
            kk={'label':model+' '+labels[j],
                'linestyle':linestyles[j],
                'color':colors[j]}
            if 'win' in k.keys():
                kk['win']=k['win']
            if 't_start' in k.keys():
                kk['t_start']=k['t_start']
            if 't_stop' in k.keys():
                kk['t_stop']=k['t_stop']
            
            if model in v.keys():
                v[model][name].plot(ax=axs[i], **kk)
        j+=1    
    
    return fig, axs

def show_coherence(d, models, **k):
    fig, axs=show_plot('mean_coherence',d, models)
    for ax in axs:
        
        ax.set_xlim(k.get('xlim', [0,50]))
    return fig
def show_fr(d, models, **k):
    
    fig, _ =show_plot('firing_rate',d, models, **k)
    return fig

def show_fr_sets(d, models, **k):
    fig, _ =show_plot('firing_rate',d, models, **k)
    return fig

def show_mr(d, models, **k):
    fig, _ =show_plot('mean_rate_slices',d, models, **k)
    return fig

def show_mr_diff(d, models, **k):
    fig, _ =show_plot('mean_rate_diff',d, models, **k)
    return fig

def show_hr(d, models, **k):
    fig, _ =show_hist('mean_rates',d, models, **k)
    return fig

def show_hist(name, d, models=['M1','M2','FS', 'GA', 'GI','ST', 'SN'], **k):

    fig, axs=ps.get_figure(n_rows=len(models), n_cols=1, w=1000.0, h=800.0, fontsize=10)   
    labels=k.pop('labels', sorted(d.keys()))

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
#             print s 
#             print labels[j]
            k.update({'label':(model+' '+labels[j]+' ' +s),
                     'histtype':'step',
                     'linestyle':linestyles[j],
                     'color':colors[j],
                     'linewidth':linewidth[j]})
#             print k
            h=v[model][name].hist(ax=axs[i],**k) 
            
            ylim=list(axs[i].get_ylim())
            ylim[0]=0.0
            axs[i].set_ylim(ylim)
            axs[i].legend_box_to_line()
        j+=1 
#     import pylab
#     pylab.show()
    return fig, axs

def show_phase_diff(d, models, **k):
    fig, _ =show_hist('phase_diff',d, models, **k)
    return fig

def show_psd(d, models):
    fig, axs=show_plot('psd',d, models)   
    for ax in axs:
        ax.set_xlim([0,50])
    return fig