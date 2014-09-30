'''
Created on Sep 10, 2014

@author: mikael
'''
import numpy
import pylab
import os
import toolbox.plot_settings as ps

from os.path import expanduser
from toolbox import misc
from toolbox.data_to_disk import Storage_dic
from toolbox.network.manager import get_storage, save, load
from toolbox.my_signals import Data_bar
from simulate import get_file_name
import pprint
pp=pprint.pprint

from toolbox.

def gather(path, nets, models, attrs): 
    

    fs=os.listdir(path)
    d={}
    i=0
    for name in fs:
        if name[-4:]!='.pkl':
            continue
        file_name=path+name[:-4]
        sd = Storage_dic.load(file_name)
        args=nets+models+attrs
        dd=sd.load_dic(*args)
        print name[-12:-4]
        if name[-12:-4]=='GP-ST-SN':
            d = misc.dict_update(d, {'GP-ST-SN':dd})
        else:
            d = misc.dict_update(d, {name[:-4].split('-')[-1]:dd})
        i+=1
    return d



def extract_data(d, nets, models, attrs):
    
    out={}
    for keys, val in misc.dict_iter(d):
        
        if keys[-1]=='phases_diff_with_cohere':
            v=numpy.mean(val.y_val, axis=0)
        if keys[-1]=='mean_coherence':
            v=val.y[2:20]
            
        out=misc.dict_recursive_add(out,  keys, v)
    return out             
def compute_performance(d, nets, models, attr):       
    results={}
    
    for run in d.keys():
        for model in models:
            for attr in attrs:
                keys1=[run, 'Net_0', model, attr]
                keys2=[run, 'Net_1', model, attr]
                keys3=[run, model, attr]

                v1=misc.dict_recursive_get(d, keys1)
                v2=misc.dict_recursive_get(d, keys2)

                v=(v1-v2)/v1
                v[numpy.isnan(v)]=0
                v=numpy.mean((v)**2)
                
                results=misc.dict_recursive_add(results,  keys3, v)
                
    return results

def generate_plot_data(d, models, attrs, exclude=['all',
                                                  'striatum',
                                                  'GP-ST-SN']):
    out={}
    
    labels=[]
    keys=d.keys()
    for k in sorted(keys):
        if k in exclude:
            continue
        labels.append(k)
    
    for model in models:
        l=[]
        for attr in attrs:
            l.append([])
            for k in sorted(keys):
                if k in exclude:
                    continue
                l[-1].append(misc.dict_recursive_get(d, [k, model, attr]))
            
        lsum=[[a]*len(l[0]) for a in numpy.sum(l, axis=1)]
        obj=Data_bar(**{'y':numpy.array(l)/numpy.array(lsum)})
        out=misc.dict_recursive_add(out, [model], obj)
    
    return out, labels

def plot(d, labels):
    fig, axs=ps.get_figure(n_rows=4, n_cols=1, w=500.0*0.65*2, h=400.0*0.65*2, fontsize=16,
                           frame_hight_y=0.6, frame_hight_x=0.8, 
                           title_fontsize=20, text_usetex=False)        
    
    nice_labels={'CTX_M1':r'CTX$\to$$MSN_{D1}$',
                 'CTX_M2':r'CTX$\to$$MSN_{D2}$',
                 'CTX_ST':r'CTX$\to$STN',
                 'MS_MS':r'MSN$\to$MSN',
                 'M1':r'$MSN_{D1}$',
                 'M2':r'$MSN_{D2}$',
                 'GP':r'GPe',
                 'SN':r'SNr',
                 'FS_M2':r'FSN$\to$$MSN_{D2}$',
                 'GP_ST':r'GPe$\to$STN',
                 'GP_FS':r'GPe$\to$FSN',
                 'GP_GP':r'GPe$\to$GPe',
                 'FS_FS':r'FSN$\to$FSN',
                 'M1_SN':r'$MSN_{D1}$$\to$SNr',
                 'ST_GP':r'STN$\to$GPe',}
    
    for i in range(len(labels)):
        if labels[i] in nice_labels.keys():
            labels[i]=nice_labels[labels[i]]
            
    
    for i, key in enumerate(d.keys()):
        ax=axs[i]
        
        d[key].bar2(ax, **{'alpha':False,
                           'color_axis':1, 
                           'colors':['r', 'b'],
                           'edgecolor':'k',
                           'top_label':False})
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        
   
       
#         ax.set_title(key)
        if i==0:
            ax.legend(('Coherence','Phase relation'), 
                      bbox_to_anchor=[0.67,1.9])
            
            ax.text(-0.11, -1.5, 'MSE normal and control',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,   
            rotation='vertical',)
#             ax.legend(bbox_to_anchor=[1.45,1.15])
            
    axs=fig.get_axes()
    ps.shift('upp', axs, 0.2, n_rows=len(axs), n_cols=1)
    ps.shift('down', axs, 0.08, n_rows=len(axs), n_cols=1)
#     ps.shift('left', axs, 0.1, n_rows=len(axs), n_cols=1)
    ps.shift('right', axs, 0.05, n_rows=len(axs), n_cols=1)
    
    labels=['GP vs GP', 'TI vs TI', 'TI vs TA', 'TA vs TA']
    ylims=[[0, 0.2]]*2+[[0, 0.16]]*2
    for i, ax in enumerate(axs):      
        
        ax.my_set_no_ticks(yticks=3)
        ax.set_ylim(ylims[i])
        pos=ax.get_position()#pylab.getp(ax, 'position')
        print pos.height
        ax.text(0.5, 0.85, labels[i],
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)     
        ax.set_ylabel('')
        if i==3:
            ax.set_xlabel('Connection with dopamine modulation removed')
        else:
            ax.my_remove_axis(xaxis=True)
            ax.set_xlabel('')

    pylab.show()
        

def plot_raw(d, d_keys, attr='mean_coherence'):
    
    for key in sorted(d.keys()):
        fig, axs=ps.get_figure(n_rows=1, 
                               n_cols=4, 
                               w=800.0*0.65*2, 
                               h=300.0*0.65*2, 
                               fontsize=8,
                               frame_hight_y=0.5, 
                               frame_hight_x=0.9, 
                               title_fontsize=8)        
        
        for i, model in enumerate(models):
            ax=axs[i]
            ax.plot(d[key]['Net_0'][model][attr], 'b')
            ax.plot(d[key]['Net_1'][model][attr], 'r')
            ax.set_title(d_keys[key]+' '+model)
 
#     pylab.show()    
    
if __name__=='__main__':
    models=['GP_GP', 'GA_GA', 'GI_GA', 'GI_GI']
    nets=['Net_0', 'Net_1']
    attrs=['mean_coherence', 'phases_diff_with_cohere']
    path='/home/mikael/results/papers/inhibition/network/simulate_slow_wave_ZZZ5/'
    home = expanduser("~")
    script_name=__file__.split('/')[-1][0:-3]
    file_name = get_file_name(script_name, home)
    from_disk=0
    
    sd = get_storage(file_name, '')
    d={}
    if not from_disk:
    
        d['raw']=gather(path, nets, models, attrs)
        d['data']=extract_data(d['raw'], nets, models, attrs)
        d['performance']=compute_performance(d['data'], nets, models, attrs)
        d['bar_obj'], d['labels']=generate_plot_data(d['performance'],
                                                     models, 
                                                     attrs)

        save(sd, d)
    else:
        filt=['bar_obj']+models+['labels']
        d = sd.load_dic(*filt)
#         d = sd.load_dic()
    pp(d['bar_obj'])
#     plot_raw(d, d_keys, attr='mean_coherence')
#     plot_raw(d, d_keys, attr='phases_diff_with_cohere')

    plot(d['bar_obj'], d['labels'])
    pp(d)
    
    
    
    
    
    