'''bu
Created on Sep 10, 2014

@author: mikael

'''
import matplotlib.gridspec as gridspec
import numpy
import pylab
import os
import sys
import warnings
import core.plot_settings as ps

from matplotlib import ticker
from matplotlib.font_manager import FontProperties
from core.my_signals import Data_bar
from core import misc
from core.data_to_disk import Storage_dic, text_save
from core.network.manager import get_storage, save
from scripts_inhibition.base_simulate import get_file_name, save_figures
import pprint
from copy import deepcopy

pp=pprint.pprint

# from core.


def create_name(file_name):
    return file_name.split('-')[-1]


def gather(path, nets, models, attrs, **kwargs): 
     
    fs=kwargs.get('file_names')
    dic_keys=kwargs.get('dic_keys')
    name_maker=kwargs.get('name_maker', create_name)
    
    if not fs:
        fs=os.listdir(path)
        fs=[path+s for s in fs]
    
    fs=sorted(fs)
    if not dic_keys:    
        fs=[s for s in fs if name_maker(s) ]
        dic_keys=[name_maker(s) for s in fs if name_maker(s) ]
        
        if len(fs) != len(set(dic_keys)):
            print 'dic_keys do not contain unique names'
        
    d={}
    i=0
    for name0, key in zip(fs, dic_keys):

        dd={}
        
        if key.split('/')[-1] in ['jobbs', 'params', 'std']:
            continue
        
        for net in nets:
            name=name0+'/'+net+'.pkl'
            
            if kwargs.get('ignore_files'):
                if kwargs.get('ignore_files')(name):
                    continue              
            if not os.path.isfile(name):
                warnings.warn('Data missing (no .pkl file) '+name)
                continue

#             slice(0,-4)
            file_name=name[:-4]
            sd = Storage_dic.load(file_name)
            args=nets+models+attrs
            
            ddd=sd.load_dic(*args)
#             pp(ddd)
            if ddd=={}:
                warnings.warn('Data missing '+name)
            
            dd=misc.dict_update(dd, ddd)
        if dd:
            print i, key
#         pp(dd)
#         print key, dd.keys()
        if dd:  
            d = misc.dict_update(d, {key:dd})
        i+=1
    return d

def extract_data(d, nets, models, attrs, **kwargs):
    
    out={}
    for keys, val in misc.dict_iter(d):
        
        if keys[-1]=='phases_diff_with_cohere':
            v=numpy.mean(val.y_val, axis=0)
            args=[[keys],
                  [v]]
            
        if keys[-1]=='mean_coherence': 
                
            v_max=max(val.y[2:20])
            v=numpy.mean(val.y[2:20])
            args=[[keys, keys[0:-1]+['mean_coherence_max']],
                  [val.y, v_max]]        
            
        if keys[-1]=='firing_rate':
            val.y=val.y[100:] #remove transient artifacts at start of sim
            std=numpy.std(val.y)
            v=numpy.mean(val.y)
            
            if numpy.isnan(v):
                print keys
                raise
            
            if keys[0]=='M2_pert_mod7' and keys[2]=='GI':
                print v,std, keys
#                 pylab.plot(val.y)
#                 pylab.show()
            
            if v>0.0001:
                synchrony_index=(std**2)/v
            else:
                synchrony_index=0.0
                
            if keys[0]=='SN' and keys[1]=='Net_1':
#                 print keys, synchrony_index

                print (keys, round(synchrony_index,1), 
                       round(v,1),  round(std,1))#, round(v,1), round(std,1)
            if keys[0]=='Normal' and keys[1]=='Net_1':
#                 print keys, synchrony_index

                print (keys, round(synchrony_index,1), 
                       round(v,1),  round(std,1))            
            
            fs=kwargs.get('oi_fs',1000.)
            x=val.x[kwargs.get('oi_start', fs):]
            y=val.y[kwargs.get('oi_start', fs):]
  
            k=kwargs.get('psd',{'NFFT':128*8*4, 
                                'fs':kwargs.get('oi_fs',1000.), 
                                'noverlap':128*8*4/2}) 

              
            d={'x':x, 'y':y}
            import core.signal_processing as sp
            from core.my_signals import Data_psd
              
              
#             y=y[:256*(len(y)/256)]
#             y=numpy.mean(y.reshape((len(y)/4),4), axis=1).ravel()
  
              
            ypsd,xpsd=sp.psd(deepcopy(y), **k)
            
            
            if numpy.isnan(numpy.min(ypsd)):
                pp(ypsd)
                print keys
                
            d={'x':xpsd,'y':ypsd}
            psd=Data_psd(**d)

            
            bol=(psd.x>kwargs.get('oi_min', 15))*(psd.x<kwargs.get('oi_max', 25))
            integral1=sum(psd.y[bol])
            
            
            idx=numpy.argmax(psd.y)    
                    
#             kwargs['oi_upper']=psd.x[idx]*1.6
#             print psd.x[idx], keys
            
            oi_upper=kwargs.get('oi_upper', psd.x[idx]*1.6)
            
            bol=(psd.x>0)*(psd.x<oi_upper)
            integral2=sum(psd.y[bol])
            oscillation_index=integral1/integral2

            args=[[keys, keys[0:-1]+['synchrony_index'],
                   keys[0:-1]+['oscillation_index'],
                  keys[0:-1]+['psd2']],
                  [v, synchrony_index,
                   oscillation_index,
                   psd]]
        
        if keys[-1]=='psd':
            psd=val
            
            idx=numpy.argmax(psd.y)            
#             print idx
            
            oi_upper=kwargs.get('oi_upper', psd.x[idx]*1.6)
            
                            
            bol=(psd.x>kwargs.get('oi_min', 15))*(psd.x<kwargs.get('oi_max', 25))
            integral1=sum(psd.y[bol])

            
            bol=(psd.x>0)*(psd.x<oi_upper)
            integral2=sum(psd.y[bol])
            oscillation_index=integral1/integral2
            args=[[keys,
                    keys[0:-1]+['psd_oi']],
                  [val, oscillation_index]]   
    
        
        for k, v in zip(*args):
            out=misc.dict_recursive_add(out,  k, v)
    
    attrs=[]
    for keys, val in misc.dict_iter(out):
        if not keys[-1] in attrs:
            attrs.append(keys[-1])
        
    return out, attrs     
      
def compute_mse(v1,v2):
#     if v1<=0.:
#     v_mse=numpy.NAN
#     else:
#     try:
    v_mse=(v1-v2)/v1
#     except:
#         print 'Failed cmp v_mse for'
#         v_mse=1.
    if isinstance(v_mse, numpy.ndarray):
        v_mse[numpy.isnan(v_mse)]=0
    elif numpy.isnan(v_mse):
        v_mse=0
    v_mse=numpy.mean((v_mse)**2)
    
    return v_mse



def add_or_append(results, l, x,  keys, values):
    for keys, val in zip(keys, values):
        if not misc.dict_haskey(results, keys):
            results = misc.dict_recursive_add(results, keys, 
                [(x, val)])
        else:
            l = misc.dict_recursive_get(results, keys)
            l.append((x, val))
    
    return keys, val, l, results

def compute_performance(d, nets, models, attrs, **kwargs):       
    results={}
    
    midpoint=kwargs.get('midpoint',1)
    
    for run in d.keys():
        
        if run=='no_pert' and not kwargs.get('exclude_no_pert',False):
            continue
        for model in models:
            for attr in attrs:
               
                run_key_list=run.split('_')
                
                if kwargs.get('compute_performance_name_and_x'):
                    x, name=kwargs.get('compute_performance_name_and_x')(run_key_list)
                elif len(run_key_list)==4:
                    s,t,_,x=run_key_list
                    x=float(x)
                    name=s+'_'+t
                elif len(run_key_list)==3:
                    name,_,mod=run_key_list
#                     print mod
                    if not mod.isdigit() and not mod[0:3]=='mod':
                        x=1
                        name='_'.join(run_key_list)
                    else:     
                        x=int(mod[3:])
                elif len(run_key_list)==2:
                    s,t=run_key_list
                    name=s+'_'+t
                    x=1
                elif len(run_key_list)==1:
                    name=run_key_list[0]
                    x=0
#                 print name 
                keys1=[run, 'Net_0', model, attr]
                keys2=[run, 'Net_1', model, attr]
                keys3=['control',name, model, attr]
                keys4=['lesion',name, model, attr]
                
                if not misc.dict_haskey(d, keys1 ):
                    continue
                
                print run, model, attr
                
                try:
                    v1=misc.dict_recursive_get(d, keys1)
                except:
                    print keys1
                    pp(misc.dict_recursive_get(d, keys1[:3]))
                    raise
                try:
                    v2=misc.dict_recursive_get(d, keys2)               
                except:
                    print keys2
                    pp(misc.dict_recursive_get(d, keys2[:3]))
                    raise
                

                keys=[keys3, keys4]
                values= [v1, v2]
                add_or_append(results,  run_key_list,x, keys, values)
                
                
                ref_keys=kwargs.get('compute_performance_ref_key', 'no_pert')
                keys_c= [kwargs.get('key_no_pert',ref_keys), 
                         'Net_0', model, attr]
                keys_l= [kwargs.get('key_no_pert',ref_keys), 
                         'Net_1', model, attr]
                
                if not misc.dict_haskey(d, keys_c ) or kwargs.get('skip_mse'):
                    continue
                      
#                 if model=='ST' and keys_c[-1]=='synchrony_index':
#                     print keys_c, v_mse0                
                

                    
                if attr=='psd':
                    continue
                if attr=='psd2':
                    continue
               
                if attr=='firing_rate':
                    s='fr'
                if attr=='mean_coherence':
                    s='mc'
                if attr=='mean_coherence_max':
                    s='mcm'
                if attr=='phases_diff_with_cohere':
                    s='pdwc'
                if attr=='oscillation_index':
                    s='oi'
                if attr=='synchrony_index':
                    s='si'
                if attr=='psd_oi':
                    s='psd_oi'                    


                v_control0=misc.dict_recursive_get(d,keys_c)
                v_lesion0=misc.dict_recursive_get(d,keys_l)
               
#                 if type(v_lesion0)==str:
#                     print v_lesion0
#                 if type(v_control0)==str:
#                     print v_control0
#                 print keys_c
                v_mse0=compute_mse(v_control0, v_lesion0)

                    
                keys5=['control',name, model, 'mse_rel_control_'+s]
                keys6=['lesion', name, model, 'mse_rel_control_'+s]
                        
                

                
                v_mse1=compute_mse(v_control0,v1)
                v_mse2=compute_mse(v_control0,v2)/v_mse0
        

                
                keys=[ keys5, keys6]
                values=[v_mse1, v_mse2]
                add_or_append(results,  run_key_list,x, keys, values)

                
                #Add midpoint
                if kwargs.get('add_midpoint',True):
                    for keys, val in zip([keys3,keys4,keys5,keys6],
                                        [v_control0, v_lesion0, 
                                         0, v_mse0/v_mse0]):         
                        l=misc.dict_recursive_get(results, keys)
#                         print keys, l
#                         print keys_c
#                         print (midpoint, val)
                        if not midpoint in [e[0] for e in l]: 
                            l.append((midpoint, val)) 
                    
    for keys, val in misc.dict_iter(results):
        x, y=zip(*val)
        
        x=numpy.array(x)
        y=numpy.array(y)
        
        idx=numpy.argsort(x)
        x=x[idx]
        y=y[idx]
        d=misc.dict_recursive_add(results, keys, [x,y])
                
    gradients={}
    for keys, val in misc.dict_iter(results):
        a1,a2=9,45
        y=val[1]
        h=0.25
#         if len(y)>6:
#             g=(-y[0]+a1*y[1]-a2*y[2]+a2*y[4]-a1*y[5]+y[6])/(60*h)
#         else:
        g=1
        d=misc.dict_recursive_add(gradients, keys, g)         
    return results, gradients

def nice_labels(version=0):
    
    d={'CTX_M1':r'CTX$\to$$MSN_{D1}$',
       'CTX_M2':r'CTX$\to$$MSN_{D2}$',
       'CTX_ST':r'CTX$\to$STN',
       'FS':r'FSN',
       'MS_MS': r'MSN$\to$MSN',
       'M1_M1':r'$MSN_{D1}$$\to$$MSN_{D1}$',
       'M1_M2':r'$MSN_{D1}$$\to$$MSN_{D2}$',   
       'M2_M1':r'$MSN_{D2}$$\to$$MSN_{D1}$',
       'M2_M2':r'$MSN_{D2}$$\to$$MSN_{D2}$',
       'M1_SN':r'$MSN_{D1}$$\to$SNr',
       'M2_GI':r'$MSN_{D2}$$\to$$GPe_{TI}$',
       'M1':r'$MSN_{D1}$',
       'M2':r'$MSN_{D2}$',
       'GP':r'GPe',
       'GA':r'$GPe_{TA}$',
       'GI':r'$GPe_{TI}$',
       'SN':r'SNr',
       'ST':r'STN',
       'FS_M1':r'FSN$\to$$MSN_{D1}$',
       'FS_M2':r'FSN$\to$$MSN_{D2}$',
       'GA_FS':r'$GPe_{TA}$$\to$FSN',
       'GA_M1':r'$GPe_{TA}$$\to$$MSN_{D1}$',
       'GA_M2':r'$GPe_{TA}$$\to$$MSN_{D2}$',
       'GP_ST':r'GPe$\to$STN',
       'GA_ST':r'$GPe_{TA}$$\to$STN',
       'GI_ST':r'$GPe_{TI}$$\to$STN',
       'GP_FS':r'$GPe_{TA}$$\to$FSN',
       'GP_GP':r'GPe$\to$GPe',
       'GA_GA':r'$GPe_{TA}$$\to$$GPe_{TA}$',
       'GA_GI':r'$GPe_{TA}$$\to$$GPe_{TI}$',
       'GI_GA':r'$GPe_{TI}$$\to$$GPe_{TA}$',
       'GI_GI':r'$GPe_{TI}$$\to$$GPe_{TI}$',
       'GI_SN':r'$GPe_{TI}$$\to$SNr',
       'FS_FS':r'FSN$\to$FSN',
       'FS_MS':r'FSN$\to$MSN',
       'ST_GP':r'STN$\to$GPe',
       'ST_GI':r'STN$\to$$GPe_{TI}$',
       'ST_GA':r'STN$\to$$GPe_{TA}$',
       'ST_SN':r'STN$\to$SNr',}
    
    if version==1:
        d.update({'GP_GP':'GP vs GP',
                  'GI_GI':'TI vs TI', 
                  'GI_GA':'TI vs TA', 
                  'GA_GA':'TA vs TA'})
 
    if version==2:
        d.update({'GA':'TA',
                  'GI':'TI',
                  'M1':r'$MSN_{D1}$',
                  'M2':r'$MSN_{D2}$',})   
    return d

# def nice_labels2():
# 
#     return d
        
def gs_builder_conn(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.02 ), 
              hspace=kwargs.get('hspace', 0.1 ))

    iterator = [[slice(0,1),slice(0,1)],
                [slice(1,4),slice(0,1)],
                [slice(4,6),slice(0,1)],
                [slice(0,1),slice(1,2)],
                [slice(1,4),slice(1,2)],
                [slice(4,6),slice(1,2)]]
    
    return iterator, gs, 

def gs_builder_index(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.02 ), 
              hspace=kwargs.get('hspace', 0.1 ))

    iterator = [
                [slice(0,3),slice(0,1)],
                [slice(3,6),slice(0,1)],
                [slice(0,3),slice(1,2)],
                [slice(3,6),slice(1,2)]]
    
    return iterator, gs, 

def gs_builder_index2(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.02 ), 
              hspace=kwargs.get('hspace', 0.1 ))

    iterator = [
                [slice(0,1),slice(0,2)],
                [slice(2,3),slice(0,2)],
                [slice(1,2),slice(0,2)],
                [slice(3,4),slice(0,2)]]
    
    return iterator, gs, 

# def gs_builder_index(*args, **kwargs):
# 
#     n_rows=kwargs.get('n_rows',2)
#     n_cols=kwargs.get('n_cols',3)
#     order=kwargs.get('order', 'col')
#     
#     gs = gridspec.GridSpec(n_rows, n_cols)
#     gs.update(wspace=kwargs.get('wspace', 0.02 ), 
#               hspace=kwargs.get('hspace', 0.1 ))
# 
#     iterator = [[slice(0,1),slice(0,1)],
#                 [slice(0,1),slice(1,2)]]
#     
#     return iterator, gs, 

# gs_builder_index

def gs_builder_conn2(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.2 ), 
              hspace=kwargs.get('hspace', 0.2 ))

    iterator = [[slice(0,3),slice(1,12)],
                [slice(3,7),slice(1,12)],
                [slice(0,3),slice(12,23)],
                [slice(3,7),slice(12,23)]]
    
    return iterator, gs, 

def gs_builder_coher(*args, **kwargs):
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.05 ), 
              hspace=kwargs.get('hspace', 1. / n_cols ))

    iterator = [[slice(1,10), slice(6,17)],
                [slice(1,10), slice(17,20)]]
    
    return iterator, gs, 

def gs_builder_oi_si(*args, **kwargs):
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.1 ), 
              hspace=kwargs.get('hspace', 0.1 ))



    iterator = [[slice(1,2), slice(0,2) ],
                [slice(2,5), slice(0,2) ],
                [slice(1,5), slice(3,6) ],
                [slice(5,9), slice(0,2) ],
                [slice(5,9), slice(3,6) ]]
    
    return iterator, gs, 

def gs_builder_oi_si_simple(*args, **kwargs):
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.1 ), 
              hspace=kwargs.get('hspace', 0.1 ))

    iterator = [[slice(0,1), slice(0,9) ],

                [slice(1,2), slice(0,9) ]]
    
    return iterator, gs,


def gs_builder_psd(*args, **kwargs):
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.15 ), 
              hspace=kwargs.get('hspace', 0.15 ))

    iterator=[]
    for i in range(n_rows):
        for j in range(n_cols):
            iterator.append([slice(i,i+1), slice(j,j+1)])
                
    return iterator, gs,

def gs_builder_coher2(*args, **kwargs):
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.05 ), 
              hspace=kwargs.get('hspace', 1. / n_cols ))

    iterator = [[slice(2,8), slice(6,17)],
                [slice(2,8), slice(17,20)]]
    
    return iterator, gs, 

def generate_plot_data_raw(d, models, attrs, exclude=[], flag='raw', attr='firing_rate', **kwargs):

    labelsx_meta=[]
    
    if flag=='raw':
        data_keys=['x', 'y']
    if flag=='gradient':
        data_keys=['z']
    
    for k in kwargs.get('key_sort', sorted)(d.keys()):
        if k in exclude:
            continue        
        labelsx_meta.append(k)
    
    out={}
    for k in labelsx_meta:
        for model in sorted(models):
            for attr in attrs:
                
                if not misc.dict_haskey(d, [k, model, attr]):
                    continue
                
                v=d[k][model][attr]    
                if type(v)!=list:
                    v=[v]
                for dk,vv in zip(data_keys, v):
                    if not misc.dict_haskey(out, [attr, model,dk]):
                        out=misc.dict_update(out, {attr:{ model:{dk:[vv]}}})
                    else:
                        out[attr][model][dk].append(vv)
    
    for keys, val in misc.dict_iter(out):
#         print numpy.array(val).shape
        val=numpy.array(val).ravel()
#         print val.shape
        out=misc.dict_recursive_set(out, keys, val)

    l1=[m for m in ['M1', 'M2', 'FS', 'GA', 'GI', 'GP','SN', 'ST'] if m in models]
    l1=[m for m in l1 if not (m in exclude)]
    l2=['GA_GA', 'GI_GA', 'GI_GI', 'GP_GP']
    dd={'firing_rate':{'labelsy':l1, 
                       'labelsx_meta':labelsx_meta},
        'synchrony_index':{'labelsy':l1, 
                              'labelsx_meta':labelsx_meta},
        'oscillation_index':{'labelsy':l1, 
                              'labelsx_meta':labelsx_meta},
        'psd':{'labelsy':l1, 
                              'labelsx_meta':labelsx_meta},
        'psd2':{'labelsy':l1, 
                              'labelsx_meta':labelsx_meta},
        'psd_oi':{'labelsy':l1, 
                              'labelsx_meta':labelsx_meta},
        'mse_rel_control_fr':{'labelsy':l1, 
                              'labelsx_meta':labelsx_meta},
        'mse_rel_control_si':{'labelsy':l1, 
                              'labelsx_meta':labelsx_meta},
        'mse_rel_control_oi':{'labelsy':l1, 
                              'labelsx_meta':labelsx_meta},
#         'mse_rel_control_psd':{'labelsy':l1, 
#                               'labelsx_meta':labelsx_meta},
        'mean_coherence':{'labelsy':l2, 
                          'labelsx_meta':labelsx_meta},
        'mean_coherence_max':{'labelsy':l2, 
                          'labelsx_meta':labelsx_meta},
        'mse_rel_control_mc':{'labelsy':l2, 
                              'labelsx_meta':labelsx_meta},
        'mse_rel_control_mcm':{'labelsy':l2, 
                              'labelsx_meta':labelsx_meta},
        'mse_rel_control_pdwc':{'labelsy':l2, 
                              'labelsx_meta':labelsx_meta},
        
#         'phases_diff_with_cohere':{'labelsy':l2, 
#                               'labelsx_meta':labelsx_meta}
        }
    
    for attr, d in dd.items():
        if not attr in attrs:
            continue
#         print attr
        if flag=='raw':
#             print d
            key=out['firing_rate'].keys()[0]
            d['labelsx']=out['firing_rate'][key]['x']#[0.25,0.5,0.75,1,1.25,1.5,1.75]*len(labelsx_meta)
        if flag=='gradient':
            d['labelsx']=labelsx_meta
            
        for dk in data_keys:
            d[dk]=[]
            
            for key in d['labelsy']:
                try:
                    if not attr in out.keys():
                        continue
                    v=out[attr][key]
                except Exception as e:
                    print attr
                    print out.keys()
                    
                    raise type(e)(e.message), None, sys.exc_info()[2]
                d[dk].append(v[dk])
            d[dk]=numpy.array(d[dk])
            
    return dd




def separate_M1_M2(*args, **kwargs):
    l=[]
    for d in args:
        d0=d.copy()
        d1=d.copy()

        for k in [kwargs.get('z_key'), 'labelsy', 'labelsy_meta']:
            if k in d0:        
                d0[k]=d0[k][0:kwargs.get(k+'_sep_at',2)]
            if k in d1:        
                d1[k]=d1[k][kwargs.get(k+'_sep_at',2):]
        l.extend([d0,d1])
    return l
# 
# def psd(self, bin_w = 5., nmax = 4000,time_range = []):
#     
#     
#     ids = np.unique(spikes[:,0])[:nmax]
#     nr_neurons = len(ids)
#     bins = np.arange(time_range[0],time_range[1],bin_w)
#     a,b = np.histogram(spikes[:,1], bins)
#     ff = abs(np.fft.fft(a- np.mean(a)))**2
#     Fs = 1./(bin_w*0.001)
#     freq2 = np.fft.fftfreq(len(bins))[0:len(bins/2)+1]
#     freq = np.linspace(0,Fs/2,len(ff)/2+1)
#     px = ff[0:len(ff)/2+1]
#     max_px = np.max(px[1:])
#     idx = px == max_px
#     corr_freq = freq[pl.find(idx)]
#     new_px = px/sum(px)
#     return new_px,freq, freq2, corr_freq[0]



def spec_entropy(power, freq, freq_range = []):

    if freq_range!=[]:
        power = power[(freq>freq_range[0]) & (freq < freq_range[1])]
        freq = freq[(freq>freq_range[0]) & (freq < freq_range[1])]
    
    k = len(freq)
    sum_power = 0
    
    
    power/=sum(power)
    for ii in range(k):
        sum_power += (power[ii]*numpy.log(power[ii]))
    spec_ent = -(sum_power/numpy.log(k))
    return spec_ent

def plot_psd(d0,d1, flag='dop', labelsx=[], **k):

    fs=k.get('cohere_fig_fontsize',16)
    tfs=k.get('cohere_fig_title_fontsize',6)
    scale=4
    
    n=int(numpy.sqrt(len(d1['labelsx_meta'])))+1
    
    figs=[]
    for iModel in range(len(d1['labelsy'])):
#         kw= {'n_rows':n,
#              'n_cols':n,  
#              'w':int(72/2.54*11.6)*(4./3)*scale,
#              'h':int(72/2.54*11.6)*scale,
#              'linewidth':1*scale,
#              'fontsize':7*scale,
#              'title_fontsize':7*scale,
#              'gs_builder':gs_builder_psd,
#              'projection':None}
#         fig, axs=ps.get_figure2(**kw
#                                 ) 
        
        
        kw= {'n_rows':n,
             'n_cols':n,  
             'w':int(72/2.54*11.6)*(4./3)*scale,
             'h':int(72/2.54*11.6)*scale,
             'linewidth':1*scale,
             'fontsize':7*scale,
             'title_fontsize':7*scale,
             'gs_builder':gs_builder_psd,
             'projection':'3d'}
        fig, axs=ps.get_figure2(**kw
                                ) 
        
        fig.suptitle(d1['labelsy'][iModel])
        for ax in axs:
            ax.tick_params(direction='out',
                           length=2,
                           width=0.5,
                           pad=0.01,
                            top=False, right=False
                            )
      
        for i, ax in enumerate(axs):
            
            if i>=len(d1['labelsx_meta']):
                continue
            
            
            if k.get('psd_data_set', 'lesion')=='control':
                data=d0['y'][iModel,i]
            
            if k.get('psd_data_set', 'lesion')=='lesion':
                data=d1['y'][iModel,i]
#             data=d1['y'][iModel,i]
            
            if type(data) not in [list, numpy.ndarray]:
                data=[data]
            colors=misc.make_N_colors('copper', len(data))
            Z=[]
            
            oi=[]
            oi2=[]
            for j, psd in enumerate(data):
                
                
#                 idx=numpy.argmax(psd.y)
#                 k['oi_upper']=psd.x[idx]*0.8
                
                
                bol=(psd.x>k.get('oi_min', 15))*(psd.x<k.get('oi_max', 25))
                integral1=sum(psd.y[bol])
                
                bol=(psd.x>0)*(psd.x<k.get('oi_upper', 1000))
                integral2=sum(psd.y[bol])
                oi.append(integral1/integral2)
                oi2.append(spec_entropy(psd.y, psd.x,freq_range = [0, k.get('oi_upper')]))
#                  x=numpy.linspace(-numpy.pi*3,numpy.pi*3,len(trace))
                norm=sum(psd.y)*(psd.x[-1]-psd.x[0])/len(psd.x)
                
                if norm==0.0:
                    print d1['labelsy'][iModel], d1['labelsx_meta'][i]
#                     raise
                else:
                    psd.y/=norm 
                Z.append(psd.y)
#                 psd.plot(ax, color=colors[j])
            
#             ax.plot(oi, marker='o')
#             ax.plot(oi2, marker='o')
            Z=numpy.array(Z)
            if numpy.isnan(numpy.min(Z)):
                pp(Z)
                print d1['labelsy'][iModel], d1['labelsx_meta'][i]
#                 raise
            
            X = numpy.arange(Z.shape[1])
            Y = numpy.arange(Z.shape[0])
            X, Y = numpy.meshgrid(X, Y)
            
            Z[numpy.isnan(Z)]=0
            from matplotlib import cm
            image=axs[i].plot_surface(X, Y, Z, 
                                      cstride=1,
                                      rstride=1,
                                      vmin=0,vmax=1, 
                                       cmap=cm.coolwarm,
                                linewidth=0, antialiased=False
                                 )
            ax.set_zlim([0, .5])
            image.set_clim([0,numpy.max(Z)])
#             pylab.show()
#             pylab.show()  
#             ax.set_ylim([0,1])
            ax.set_title(d1['labelsx_meta'][i])
#             ax.set_xlim([0,80])
#             if i==0:
#                 sm = pylab.cm.ScalarMappable(cmap='copper', 
#                                              norm=pylab.normalize(vmin=0,
#                                                                   vmax=len(data)-1))
#                 sm._A = []
#                     
#                 box = ax.get_position()
#                 pos=[box.x0+1.03*box.width, box.y0+box.height*0.1,
#                      0.01, box.height*0.8]
#                 axColor=pylab.axes(pos)
#                 cbar=pylab.colorbar(sm, cax=axColor, ticks=range(len(data)))
#                 
#                 cbar.ax.tick_params( length=1, )
#     #             cbar.ax.set_yticklabels(ranln)  
#                  
#                 ax.text(1.32,  0.5, r'Perturbation', 
#                         transform=ax.transAxes, va='center', rotation=270) 
#                 
#     #             x.set_xlabel(r'Angle (rad)')
#     #             ax.set_ylabel('Norm. count TI-TA')
#                 ax.my_set_no_ticks(xticks=4)
#                 ax.my_set_no_ticks(yticks=4)
            if i!=0:
                ax.my_remove_axis(xaxis=True, yaxis=True)
            
#             ax.set_ylim([0,0.31])

        figs.append(fig)
#     kw={}
#     images=[]    
#     ax=axs[0]
    return figs

def plot_oi_si_simple(d0,d1,d2,d3, flag='dop', labelsx=[], **k):
    fs=k.get('cohere_fig_fontsize',16)
    tfs=k.get('cohere_fig_title_fontsize',6)
    scale=k.get('scale', 1)
    fig, axs=ps.get_figure2(**k['oi_si_simple_fig_kw']) 

    for ax in axs:
        ax.tick_params(direction='in',
                       length=2, top=False, right=False)  
  
    kw={}
    images=[]    
    ax=axs[0]
    


    bol=[l=='no_pert' for l in d3['labelsx_meta']]    
    y=[]    
    for i, _ in enumerate(d0['labelsy']):
        val=d1['y'][i, numpy.array(bol)][0][0]
        y.append(val)
#         y[1].append(d1['y'][i, numpy.array(bol)][0])
#     y=[
#         d1['y'][0, numpy.array(bol)][0][0],
#         d1['y'][1, numpy.array(bol)][0][0],
#         d1['y'][2, numpy.array(bol)][0][0],
#         d1['y'][3, numpy.array(bol)][0][0]] 
    v=[]
    

    bol=numpy.array([l not in ['no_pert'] 
                     for l in d1['labelsx_meta']])
    for i in range(len(d0['labelsy'])):    
        e=numpy.array(list(d1['y'][i,bol])).ravel()
        v.append(e/y[i])
    
    
    labelsy=[nice_labels(2)[la] for la in d0['labelsy']]
    v=numpy.array(v)
    kw['ax']=ax
    kw['d']={'z':v,
            'labelsx_meta':[e for i, e in enumerate(d0['labelsx_meta']) 
                            if bol[i]],
            'labelsx':range(v.shape[1]),
            'labelsy_meta':labelsy,
#             'labelsy_meta':['TA', 'TI','SNr', 'STN'],
            'labelsy':range(len(v))}
    
    kw['images']=images
    kw['z_key']='z'
    kw['startx']=0
    kw['starty']=0
    kw['flip_axes']=False
    kw['vertical_lines']=True
    kw['horizontal_lines']=False
    kw['fontsize_x']=k.get('fontsize_x',24)
    kw['fontsize_y']=k.get('fontsize_y',24)
    kw['nice_labels_x']=nice_labels(version=0)
    kw['nice_labels_y']=nice_labels(version=0)
    kw['cmap']='coolwarm'
    kw['color_line']='k'
    kw['csv_path']=k.get('data_path')
    _plot_conn(**kw)
    
    box = ax.get_position()
    pos=[box.x0+1.03*box.width, box.y0+box.height*0.1, 
         0.01,  box.height*0.8]
        
    for l in ax.patches:
        pylab.setp(l,**{'edgecolor':'0.5'})
    
    axColor=pylab.axes(pos)
    cbar=pylab.colorbar(images[0], cax = axColor, 
                        orientation="vertical")

    
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params( length=1, ) 

    ax.text(1.18, 0.0, '-/+ relative lesion', 
            transform=ax.transAxes, va='center', rotation=270) 
  
    ax=axs[1]
    

    bol=[l=='no_pert' for l in d3['labelsx_meta']]
    y2=[]    
#     for i, _ in enumerate(d0['labelsy']):
#         
#         y2.append(numpy.mean(d0['y'][i,:]))
#     
    for i, _ in enumerate(d0['labelsy']):
        val=d3['y'][i, numpy.array(bol)][0][0]
        y2.append(val)
#     y2=[
#         d3['y'][0, numpy.array(bol)][0][0],
#         d3['y'][1, numpy.array(bol)][0][0],
#         d3['y'][2, numpy.array(bol)][0][0],
#         d3['y'][3, numpy.array(bol)][0][0] ] 
    v=[]

    bol=numpy.array([l not in ['no_pert'] 
                     for l in d1['labelsx_meta']] )
    for i in range(len(d0['labelsy'])):    
        e=numpy.array(list(d3['y'][i,bol])).ravel()
        
        
    
        if y2[i]>0:
            v.append(e/y2[i])
        else:
            v.append(0.0)
    
    
    v=numpy.array(v)
    kw['ax']=ax
    kw['d']={'z':v,
            'labelsx_meta':[e for i, e in 
                            enumerate(d2['labelsx_meta']) if bol[i]],
            'labelsx':range(v.shape[1]),
            'labelsy_meta':labelsy,
#             'labelsy_meta':['TA', 'TI','SNr', 'STN'],
            'labelsy':range(len(v))}
    _plot_conn(**kw)


    box = ax.get_position()
    pos=[box.x0+1.03*box.width, box.y0+box.height*0.1, 
         0.01,  box.height*0.8]
        
    for l in ax.patches:
        pylab.setp(l,**{'edgecolor':'0.5'})
    
    axColor=pylab.axes(pos)
    cbar=pylab.colorbar(images[1], cax = axColor, 
                        orientation="vertical")

    
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params( length=1, ) 

    images[0].set_clim(k.get('oi_si_simple_clim0', [0,2]))
    images[1].set_clim(k.get('oi_si_simple_clim1', [0,2]))

    for i in [0]:
        axs[i].my_remove_axis(xaxis=True, yaxis=False)
        axs[i].set_ylabel('Synchrony')
    
    for i in [1]:
        axs[i].set_ylabel('Oscillation') 
    
#     pylab.show()
    return fig

def plot_oi_si(d0,d1,d2,d3, flag='dop', labelsx=[], **k):
    linewidth=0.5
    fs=k.get('cohere_fig_fontsize',16)
    tfs=k.get('cohere_fig_title_fontsize',6)
    scale=k.get('scale', 1)
    fig, axs=ps.get_figure2(n_rows=13,
                             n_cols=6,  
                             w=int(72/2.54*11.6)*scale,
                             h=int((72/2.54*11.6)/1.6)*scale,
                             linewidth=linewidth,
                             fontsize=fs*scale,
                             title_fontsize=tfs*scale,
                             gs_builder=k.get('cohere_gs', gs_builder_oi_si)) 
    
#     pylab.show()
    for ax in axs:
        ax.tick_params(direction='in',
                       length=2, top=False, right=False)  
    
    
    ax1=axs[0]
    ax2=axs[1]

    bol=[l=='Normal' for l in d1['labelsx_meta']]
    ax1.set_xlim([-0.2,4.])
    ax2.set_ylim([0,15]) 
    
    ax1.set_ylim([40,55])
    y=[[],[]]
    for i, _ in enumerate(d0['labelsy']):
        y[0].append(numpy.mean(d0['y'][i,:]))
        y[1].append(d1['y'][i, numpy.array(bol)][0])


    for ax in [ax1,ax2]:
        Data_bar(**{'y':y}).bar2(ax,  **{'edgecolor':'k',
                                         'top_lable_rotation':0,
                                         'top_label_round_off':0,
                                         'colors':['k','w'],
                                         'alphas':[1,1],
                                         'color_axis':1,
                                         'top_label':False,
                                         'linewidth':linewidth, 
                                         })
    
#     pylab.show()
    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
        
#     d = .015  # how big to make the diagonal lines in axes coordinates
#     # arguments to pass plot, just so we don't keep repeating them
#     kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#     ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
#     ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#     
#     kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#     ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
#     ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    
    
    ax=axs[3]
    
    ax.text(0.02,  -.35, 'Black=Control',  transform=ax.transAxes,
                va='center', rotation=0)  
    ax.text(0.02, -.58,'White=Lesion',  transform=ax.transAxes,
                va='center', rotation=0) 
    
    bol=[l=='Normal' for l in d3['labelsx_meta']]

    y2=[[],[]]
    for i, _ in enumerate(d0['labelsy']):
        y2[0].append(numpy.mean(d2['y'][i,:]))
        y2[1].append(d3['y'][i, numpy.array(bol)][0])
        
#     y2=[[numpy.mean(d2['y'][0,:]),
#          numpy.mean(d2['y'][1,:]),
#          numpy.mean(d2['y'][2,:]),
#          numpy.mean(d2['y'][3,:])],
#        [d3['y'][0, numpy.array(bol)][0],
#         d3['y'][1, numpy.array(bol)][0],
#         d3['y'][2, numpy.array(bol)][0],
#         d3['y'][3, numpy.array(bol)][0]]]

    Data_bar(**{'y':y2}).bar2(ax,  **{'edgecolor':'k',
                                     'top_lable_rotation':0,
                                     'top_label_round_off':0,
                                     'colors':['k','w'],
                                     'alphas':[1,1],
                                     'color_axis':1,
                                     'top_label':False,
                                         'linewidth':linewidth, })             
    
    labelsy=[nice_labels(2)[la] for la in d0['labelsy']]
    ax.set_xlim([-0.2,len(labelsy)])
    axs[0].set_xlim([-0.2,len(labelsy)])
    axs[1].set_xlim([-0.2,len(labelsy)])
    
    ax.set_xticklabels(labelsy, rotation=70)
#     ax.set_xticklabels(['TA', 'TI','SNr', 'STN']) 
    kw={}
    images=[]    
    ax=axs[2]
    
    bol=numpy.array([l not in ['Normal','all'] 
                     for l in d1['labelsx_meta']])
    v=[]
    for i in range(len(labelsy)):        
        v.append(d1['y'][i,bol]/y[1][i])
        
    kw['ax']=ax
    kw['d']={'z':numpy.array(v),
            'labelsx_meta':[e for i, e in enumerate(d0['labelsx_meta']) 
                            if bol[i]],
            'labelsx':range(len(d0['y'][0,bol])),
#             'labelsy_meta':['TA', 'TI','SNr', 'STN'],
            'labelsy_meta':labelsy,
            'labelsy':range(len(v))}
    kw['images']=images
    kw['z_key']='z'
    kw['startx']=0
    kw['starty']=0
    kw['flip_axes']=False
    kw['vertical_lines']=False
    kw['vertical_lines']=False
    kw['fontsize_x']=k.get('fontsize_x',24)*scale
    kw['fontsize_y']=k.get('fontsize_y',24)*scale
    kw['nice_labels_x']=nice_labels(version=0)
    kw['nice_labels_y']=nice_labels(version=0)
    kw['cmap']='coolwarm'
    kw['color_line']='k'
    kw['csv_path']=k.get('data_path')+'syncrony.csv'
    _plot_conn(**kw)
    
    box = ax.get_position()
    pos=[box.x0+1.03*box.width, box.y0+box.height*0.1, 
         0.01,  box.height*0.8]
    
    
    for l in ax.patches:
        pylab.setp(l,**{'edgecolor':'0.5'})
    
    axColor=pylab.axes(pos)
    cbar=pylab.colorbar(images[0], cax = axColor, 
                        orientation="vertical")

    ax.text(1.18, 0.0, '-/+ relative lesion', 
            transform=ax.transAxes, va='center', rotation=270) 
    
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params( length=1, ) 

    
    ax=axs[4]
    
    bol=numpy.array([l not in ['Normal','all'] 
                     for l in d1['labelsx_meta']])

    v=[]
    for i in range(len(labelsy)):    
        v.append(d3['y'][i,bol]/y2[1][i])

    kw['ax']=ax
    kw['csv_path']=k.get('data_path')+'oscillation.csv'
    kw['d']={'z':numpy.array(v),
            'labelsx_meta':[e for i, e in enumerate(d2['labelsx_meta']) if bol[i]],
            'labelsx':range(len(d2['y'][0,bol])),
#             'labelsy_meta':['TA', 'TI','SNr', 'STN'],
            'labelsy_meta':labelsy,
            'labelsy':range(len(v))}
    _plot_conn(**kw)
    box = ax.get_position()
    pos=[box.x0+1.03*box.width, box.y0+box.height*0.1, 
         0.01,  box.height*0.8]
        
    for l in ax.patches:
        pylab.setp(l,**{'edgecolor':'0.5'})
    
    axColor=pylab.axes(pos)
    cbar=pylab.colorbar(images[1], cax = axColor, 
                        orientation="vertical")

    
    tick_locator = ticker.MaxNLocator(nbins=2)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params( length=1, ) 

    images[0].set_clim(k.get('oi_si_simple_clim0', [0,2]))
    images[1].set_clim(k.get('oi_si_simple_clim1', [0,2]))

#     for image in images:
#         image.set_clim([0,2])

    for i in [1,2]:
        axs[i].my_remove_axis(xaxis=True, yaxis=False)
        axs[i].set_ylabel('Synchrony')
    for i in [1,3]:
        axs[i].my_set_no_ticks(yticks=4)
    for i in [0]:
        axs[i].my_set_no_ticks(yticks=1)    
    for i in [3,4]:
        axs[i].set_ylabel('Oscillation')

    return fig

def plot_coher(d, labelsy, flag='dop', labelsx=[], **k):
    
    
    fs=k.get('cohere_fig_fontsize',16)
    tfs=k.get('cohere_fig_title_fontsize',16)
    fig, axs=ps.get_figure2(n_rows=k.get('cohere_nrows',12),
                             n_cols=20,  
                            w=k.get('w',500), 
                            h=k.get('h',900), 
                            linewidth=1,
                            fontsize=fs,
                            title_fontsize=tfs,
                            gs_builder=k.get('cohere_gs',gs_builder_coher)) 
    
#     pylab.show()

    for ax in axs:
        ax.tick_params(direction='out',
                       length=2,
                       width=0.5,
#                        pad=1,
                        top=False, right=False
                        ) 
        ax.tick_params(pad=2) 
        
    startx=0
    starty=0
    z_key='y'
    
    images=[]
    kwargs={'ax':axs[0],
            'd':d,
            'flip_axes':True,
            'fontsize_x':k.get('cohere_fontsize_x',16),
            'fontsize_y':k.get('cohere_fontsize_y',16),
#             'vertical_lines':True, 
            'horizontal_lines':True, 
            'images':images,
            'z_key':z_key,
            'startx':startx,
            'starty':starty,
            'nice_labels_x':nice_labels(version=1),
            'nice_labels_y':nice_labels(version=0),
            'color_line':'w',
            'cmap':k.get('cmap','jet')}

    _plot_conn(**kwargs)
    
    images[0].set_clim(k.get('cohere_ylim_image',[0,4]))
    
    kwargs['ax']=axs[1]
    _plot_bar(**kwargs)
    axs[1].set_xlim(k.get('cohere_ylim_bar',[0,4]))
    
       
    box = axs[0].get_position()
    pos=[box.x0+0.1*box.width, 
         box.y0+box.height
         +box.height*k.get('cohere_cmap_ypos',0.14), 
         box.width*0.8, 
         0.01]
    
    
    for l in axs[1].patches:
        pylab.setp(l,**{'edgecolor':'0.5'})
    
    axColor=pylab.axes(pos)
    #     axColor = pylab.axes([0.05, 0.9, 1.0, 0.05])
    cbar=pylab.colorbar(images[0], cax = axColor, 
                        orientation="horizontal")
    cbar.ax.set_title('MSE control vs lesion rel. base model')#, rotation=270)
    from matplotlib import ticker
    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params( length=1, ) 
       
    
    axs[0].text(0.05, 
                k.get('cohere_xlabel0_posy', -0.18), 
                k.get('labelx0', "Coherence"), 
                transform=axs[0].transAxes)
    axs[0].text(0.55, 
                k.get('cohere_xlabel0_posy', -0.18),
                 
                k.get('labelx1', "Phase shift"),
                 transform=axs[0].transAxes)
    axs[1].text(0.5,
                 k.get('cohere_xlabel10_posy', -0.065),
                 "Mean", 
                transform=axs[1].transAxes,
                ha='center',
                rotation=0)
    axs[1].text(0.5, k.get('cohere_xlabel11_posy', -0.1), "effect", 
                transform=axs[1].transAxes,
                ha='center',
                rotation=0) 

    font0 = FontProperties()
    font0.set_weight('bold')
    axs[1].text(0.5,k.get('cohere_title_posy',1.02), k.get('title', 'Slow wave'),
#                 fontsize=28,
                va='center',
                ha='center',
                 transform=axs[0].transAxes,
                                rotation=0,
                                fontproperties=font0)
    
    axs[0].text(k.get('cohere_ylabel_ypos', -0.7), 
                0.5, 
                k.get('cohere_ylabel',"Perturbed connection"), 
                transform=axs[0].transAxes,
                verticalalignment='center', 
                rotation=90)        
                    
    axs[1].my_remove_axis(xaxis=False, yaxis=True)
    axs[1].my_set_no_ticks(xticks=2)

    return fig


    
def _plot_bar(**kwargs):
    
    ax=kwargs.get('ax')
    d=kwargs.get('d')
    z_key=kwargs.get('z_key')
    flip_axes=kwargs.get('flip_axes')

    if flip_axes:
        stopy=len(d['labelsx']) 


        
        d[z_key]=numpy.transpose(d[z_key])
    else:
        stopy=len(d['labelsy'])



    ratio=1
    posy=numpy.linspace(.5*ratio,stopy-.5*ratio, stopy)    

    ax.barh(posy,numpy.mean( d[z_key],axis=0)[::-1], align='center', color='0.5',
#             linewidth=0.1
            )
    ax.plot([1,1], [0,stopy], 'k', linewidth=1, linestyle='--')    
    ax.set_ylim([0,stopy])  
     
def _plot_conn(**kwargs):
    
    csv_path= kwargs['csv_path']
    ax=kwargs.get('ax')
    d=kwargs.get('d')
    images=kwargs.get('images')
    z_key=kwargs.get('z_key')
    startx=kwargs.get('startx')
    starty=kwargs.get('starty')
    flip_axes=kwargs.get('flip_axes')
    vertical_lines=kwargs.get('vertical_lines')
    horizontal_lines=kwargs.get('horizontal_lines')
    fontsize_x=kwargs.get('fontsize_x',24)
    fontsize_y=kwargs.get('fontsize_y',24)
    nice_labels_x=kwargs.get('nice_labels_x')
    nice_labels_y=kwargs.get('nice_labels_y')
    cmap=kwargs.get('cmap', 'jet')
    color_line=kwargs.get('color_line','k')
    
    if flip_axes:
        stopx=len(d['labelsy'])
        stopy=len(d['labelsx']) 
        labelsx_meta=d['labelsy']
        if 'labelsy_meta' in d.keys():
            labelsx_meta=d['labelsy_meta']
        else:
            labelsx_meta=d['labelsy']
        labelsy_meta=d['labelsx_meta']
#         labelsx=d['labelsx']
#         labelsy=d['labelsy']
#         nice_labels=nice_labels2()
#         nice_labels2=nice_labels()
        
        d[z_key]=numpy.transpose(d[z_key])
    else:
        stopy=len(d['labelsy'])
        stopx=len(d['labelsx']) 
        
        if 'labelsy_meta' in d.keys():
            labelsy_meta=d['labelsy_meta']
        else:
            labelsy_meta=d['labelsy']
        labelsx_meta=d['labelsx_meta']
#         labelsx=d['labelsy']
#         labelsy=d['labelsx']

  
#         nice_labels2=nice_labels2()
#         nice_labels=nice_labels()
#         
    for i in range(len(labelsx_meta)):         
        if labelsx_meta[i] in nice_labels_y.keys():
            labelsx_meta[i]=nice_labels_x[labelsx_meta[i]]
            
    for i in range(len(labelsy_meta)):         
        if labelsy_meta[i] in nice_labels_y.keys():
            labelsy_meta[i]=nice_labels_y[labelsy_meta[i]]
#     
#     labelsy=d['labelsy']
#     for i in range(len(labelsy_meta)):         
#         if labelsy_meta[i] in nice_labels2().keys():
#             labelsy[i]=nice_labels2()[labelsy[i]]  
#         elif labelsy[i] in nice_labels().keys():
#             labelsy[i]=nice_labels()[labelsy[i]]
#                  

    stopx_meta=len(labelsx_meta)
    stopy_meta=len(labelsy_meta)

    ratio=stopy/stopy_meta
    posy=numpy.linspace(.5*ratio,stopy-.5*ratio, stopy_meta)    
    
    ratio=stopx/stopx_meta
    posx=numpy.linspace(.5*ratio,stopx-.5*ratio, stopx_meta)
    
    
    
    x1,y1=numpy.meshgrid(numpy.linspace(startx, stopx, stopx+1), 
                         numpy.linspace(starty, stopy, stopy+1)) 
    
    
#     Z=[]
#     for i in range(d[z_key].shape[0]):
#         e=numpy.array([list(x) for x in d[z_key][i,:]]).ravel()
#         Z.append(e)
#     Z=numpy.array(Z)
#     
    print x1.shape,y1.shape, d[z_key][::-1,].shape
#     print len(labelsy_meta[::-1])
#     print d[z_key][::-1,]
    
    
    csv=';'.join(['nuclei']+labelsx_meta)+'\n'
    for i, l1 in enumerate(d[z_key][::-1,]):
#         print labelsy_meta[::-1][i],l1
        csv+=';'.join([labelsy_meta[::-1][i]]+map(str,map(lambda x:round((x-1)*100,2),list(l1))))+'\n'
    
    print 'Saving csv to: ', csv_path
    text_save(csv, csv_path)
    
    im = ax.pcolor(x1, y1, d[z_key][::-1,], cmap=cmap)       
#     im = ax.pcolor(x1, y1, Z[::-1,], cmap=cmap)       

    if vertical_lines:
        x=numpy.linspace(0, stopx, stopx_meta+1)
        for xx in x:
            ax.plot([xx,xx],[0,stopy], color_line, 
                    linewidth=kwargs.get('y_sep_linewidth',1.), 
                    linestyle='-')
    if horizontal_lines:
        x=numpy.linspace(0, stopy, stopy_meta+1)
        for xx in x:
            ax.plot([0,stopx],[xx,xx],color_line, 
                    linewidth=kwargs.get('x_sep_linewidth',1.),
                    linestyle='-')
    
    images.append(im)                
    ax.set_yticks(posy)
    ax.set_yticklabels(labelsy_meta[::-1],
                       fontsize=fontsize_y)
    ax.set_xticks(posx)

    ax.set_xticklabels([s.rjust(10) for s in labelsx_meta], rotation=70, 
                           ha='right', 
                           fontsize=fontsize_x)

    ax.set_ylim([0,stopy])
    ax.set_xlim([0,stopx])
    

def set_colormap(ax, im, label, nbins=3, **kwargs):
    box = ax.get_position()
    axColor = pylab.axes([box.x0 + box.width * kwargs.get('x_scale',
                                                          1.03), 
                          box.y0+box.height*0.1, 
                          0.01, 
                          box.height*0.8])
    cbar=pylab.colorbar(im, cax = axColor, orientation="vertical")
    cbar.ax.set_ylabel(label, rotation=270)
    from matplotlib import ticker
    
    tick_locator = ticker.MaxNLocator(nbins=nbins)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params(direction='in',
                       length=1, 
                       width=0.5
#                        top=False, right=False
                        ) 


def plot_conn(d0, d1, d2, d3, **kwargs):
    
    kw={'n_rows':8, 
        'n_cols':2, 
        'w':int(72/2.54*17.6), 
        'h':int(72/2.54*17.6)/3, 
        'fontsize':16,
        'frame_hight_y':0.5,
        'frame_hight_x':0.7,
        'title_fontsize':16,
        'gs_builder':gs_builder_conn}
    kwargs_fig=kwargs.get('kwargs_fig', kw)
    
    fig, axs=ps.get_figure2(**kwargs_fig) 
    
      
    for ax in axs:
        ax.tick_params(direction='in',
                       length=2,
                       width=0.5,
#                        pad=1,
                        top=False, right=False
                        ) 
        ax.tick_params(pad=1)
  
    flag=kwargs.get('flag', 'raw')
    coher_label=kwargs.get('coher_label', 'Coherence')
    fr_label=kwargs.get('fr_label',"Firing rate (Hz)")
    title=kwargs.get('title', "Slow wave") 
    z_key=kwargs.get('z_key',"y")
    cmap=kwargs.get('cmap')
    color_line=kwargs.get('color_line', 'k')
    fontsize_x=kwargs.get('fontsize_x', 16)
    fontsize_y=kwargs.get('fontsize_y', 16)
    
    if kwargs.get('separate_M1_M2', True):
        d00, d01, d20, d21=separate_M1_M2( d0, d2, **{'z_key':z_key})
        args=[ d00, d01, d1, d20, d21, d3]
    else:
        args=[ d0, d1, d2, d3]
        
    
    startx=0
    starty=0
    images=[]
    for ax, d in zip(axs, args):

        k={'ax':ax,
                'd':d,
                'images':images,
                'fontsize_x':fontsize_x,
                'fontsize_y':fontsize_y,
                'z_key':z_key,
                'startx':startx,
                'starty':starty,
                'vertical_lines':True, 
                'nice_labels_x':nice_labels(version=0),
                'nice_labels_y':nice_labels(version=1),
                'cmap':cmap,
                'color_line':color_line,
                'csv_path':kwargs.get('data_path')
                
                }
        k.update(kwargs)
        _plot_conn(**k)
        

    if kwargs.get('separate_M1_M2', True):
        args=[[axs[3], axs[4],axs[5]], 
              ['', '', coher_label],
              [images[3],images[4],images[5]],
              [{'nbins':2},{'nbins':3},{'nbins':3}]]

    else:
        args=[[axs[2], axs[3]], 
              [fr_label, coher_label],
              [images[2],images[3]],
              [{'nbins':3},{'nbins':3}]]
    clim_raw=kwargs.get('clim_raw', [[0,1.4], [0,90], [0,1]])
    clim_gradient=kwargs.get('clim_gradient', [[-2,2], [-50,50], [-0.6,0.6]])
    if flag=='raw':
        if kwargs.get('separate_M1_M2', True):
            axs[4].text(1.12, 1.15, fr_label, 
                        transform=axs[4].transAxes,
                        rotation=270)
        for i, clim in enumerate(clim_raw*2):
            images[i].set_clim(clim)

    
    if flag=='gradient':
        if kwargs.get('separate_M1_M2', True):
            axs[4].text(1.2, 1.1, fr_label, transform=axs[4].transAxes,
                        rotation=270) 
        for i, clim in enumerate(clim_gradient*2):
            images[i].set_clim(clim)
    
    for ax, label, im, k in zip(*args):
        set_colormap(ax, im, label,**k)
    
    axs[0].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)    
    if kwargs.get('top_labels_show',True):   
        axs[0].text(0.35, 1.05, "Control", 
                    fontsize=kwargs.get('top_lables_fontsize',20),
                    transform=axs[0].transAxes)     

    if kwargs.get('separate_M1_M2', True):    
        axs[1].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)   
        axs[3].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True) 
        axs[4].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True) 
        axs[5].my_remove_axis(xaxis=False, yaxis=True,keep_ticks=True) 
        if kwargs.get('top_labels_show',True): 
            axs[3].text(0.35, 1.05, "Lesion", 
                        fontsize=kwargs.get('top_lables_fontsize',20),
                        transform=axs[3].transAxes)  
        title_pos=1.6
    elif kwargs.get('ax_4x1', False):
        if kwargs.get('top_labels_show',True):     
            axs[2].text(0.35, 1.05, "Lesion", 
                        fontsize=kwargs.get('top_lables_fontsize',20),
                        transform=axs[2].transAxes)  
        axs[0].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True) 
        axs[1].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True) 
        axs[2].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True) 
        axs[3].my_remove_axis(xaxis=False, yaxis=False,keep_ticks=True)
        title_pos=1.4
    else:    
        if kwargs.get('top_labels_show',True):     
            axs[2].text(0.35, 1.05, "Lesion", 
                        fontsize=kwargs.get('top_lables_fontsize',20),
                        transform=axs[2].transAxes)  
        axs[2].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True) 
        axs[3].my_remove_axis(xaxis=False, yaxis=True,keep_ticks=True)
        title_pos=1.4

    font0 = FontProperties()
    font0.set_weight('bold')
    title_ax=kwargs.get('title_ax',4)
    title_posy=kwargs.get('title_posy',0.35)
    if kwargs.get('title_flipped'):
        axs[title_ax].text(1.22, 
                           title_posy, title, transform=axs[title_ax].transAxes, 
                fontproperties=font0 , 
                horizontalalignment=  'center',
                fontsize=kwargs.get('conn_fig_title_fontsize',20),
                verticalalignment=  'center',
                ha='center',
                rotation=270)        
    else:
        axs[0].text(1., 
                    title_pos, 
                    title, 
                    transform=axs[0].transAxes, 
                    
                fontsize=k.get('conn_fig_title_fontsize',20),
                fontproperties=font0 , 
                horizontalalignment=  'center',    )        
#     fig.tight_layout()
    return fig



def add(d0,d1):
    d0['labelsy']+=d1['labelsy']
#     d0['labelsx_meta']+=d1['labelsx_meta']
    d0['x']=numpy.concatenate((d0['x'], d1['x']), axis=0)
    d0['y']=numpy.concatenate((d0['y'], d1['y']), axis=0)
    return d0

def get_data(models, nets, attrs, paths, from_disk, attr_add, sd, **kwargs):
    exclude=kwargs.pop('exclude',[])
    
    if type(paths)!=list:
        paths=[paths]
    
    d = {}
    if not from_disk:
        dtmp={}
        for path in paths:
            dtmp.update(gather(path, nets, models, attrs, **kwargs))
        
        d['raw'] = dtmp#gather(path, nets, models, attrs, **kwargs)
        d['data'], attrs = extract_data(d['raw'], nets, models, attrs, **kwargs)
#         if kwargs.get('compute_performance', True):
        out = compute_performance(d['data'], nets, models, attrs, **kwargs)
        
        
        d['change_raw'], d['gradients'] = out
        v = generate_plot_data_raw(d['change_raw']['control'], 
            models, attrs + attr_add, 
            flag='raw', 
            exclude=exclude,
            **kwargs)
        d['d_raw_control'] = v
        v = generate_plot_data_raw(d['change_raw']['lesion'], 
            models, 
            attrs + attr_add, 
            flag='raw', 
            exclude=exclude,
            **kwargs)
        d['d_raw_lesion'] = v
        v = generate_plot_data_raw(d['gradients']['control'], 
            models, attrs + attr_add, 
            flag='gradient', 
            exclude=exclude,
            **kwargs)
        d['d_gradients_control'] = v
        v = generate_plot_data_raw(d['gradients']['lesion'], 
            models, attrs + attr_add, 
            flag='gradient', 
            exclude=exclude,
            **kwargs)
        d['d_gradients_lesion'] = v

        if 'raw' not in kwargs.get('keep', []):
            del d['raw']
        
        if 'data' not in kwargs.get('keep', []):
            del d['data']
            
        del d['change_raw']
        del d['gradients']
        
        save(sd, d)
    else:
        d = sd.load_dic()
    return d

def create_figs(d, **kwargs):
    figs = []
    
    do_plots=kwargs.get('do_plots',['firing_rate',
                                    'mse_rel',
                                    'gradient',
                                    'cohere'])
    
    if 'firing_rate' in do_plots:
        
        kwargs.update({'color_line':'w'})
        fig = plot_conn(d['d_raw_control']['firing_rate'], 
                        d['d_raw_control']['mean_coherence_max'], 
                        d['d_raw_lesion']['firing_rate'], 
                        d['d_raw_lesion']['mean_coherence_max'],
                        **kwargs)
        figs.append(fig)

    if 'index' in do_plots:
        
        kwargs.update({'color_line':'w'})
        fig = plot_conn(d['d_raw_control']['synchrony_index'], 
                        d['d_raw_control']['oscillation_index'], 
                        d['d_raw_lesion']['synchrony_index'], 
                        d['d_raw_lesion']['oscillation_index'],
                        **kwargs)
        figs.append(fig)

    if 'fr_and_oi' in do_plots:
        
        kwargs.update({'color_line':'w'})
        fig = plot_conn(d['d_raw_control']['firing_rate'], 
                        d['d_raw_control']['oscillation_index'], 
                        d['d_raw_lesion']['firing_rate'], 
                        d['d_raw_lesion']['oscillation_index'],
                        **kwargs)
        figs.append(fig)

    if 'mse_rel' in do_plots:
     
        kwargs.update({'color_line':'w'})
        fig = plot_conn(d['d_raw_control']['mse_rel_control_fr'], 
            d['d_raw_control']['mse_rel_control_mc'], 
            d['d_raw_lesion']['mse_rel_control_fr'], 
            d['d_raw_lesion']['mse_rel_control_mc'],
            **kwargs)
        figs.append(fig)
        
    if 'gradient' in do_plots:
     
        k = {'flag':'gradient', 
                   'coher_label':'Coherence/nS', 
                   'fr_label':"Firing rate/nS", 
                   'z_key':"z", 
                   'cmap':'coolwarm',
                   'color_line':'k'}
    
        kwargs.update(k)
        fig = plot_conn(d['d_gradients_control']['firing_rate'], 
            d['d_gradients_control']['mean_coherence_max'], 
            d['d_gradients_lesion']['firing_rate'], 
            d['d_gradients_lesion']['mean_coherence_max'], **kwargs)
        figs.append(fig)

    if 'cohere' in do_plots:
        d0 = d['d_raw_lesion']['mse_rel_control_mc']
        d1 = d['d_raw_lesion']['mse_rel_control_pdwc']
        d2 = add(d0, d1)
        fig=plot_coher(d2, d2['labelsx'], **kwargs)
        figs.append(fig)

    if 'mse_index' in do_plots:
        d0 = d['d_raw_lesion']['mse_rel_control_si']
        d1 = d['d_raw_lesion']['mse_rel_control_oi']
        d2 = add(d0, d1)
        kwargs['labelx0']='Synchronizy'
        kwargs['labelx1']='Oscillation'
        
        fig=plot_coher(d2, d2['labelsx'], **kwargs)
        figs.append(fig)        
        
    if 'si_oi_index' in do_plots:
        d0 = d['d_raw_control']['synchrony_index']
        d1 = d['d_raw_lesion']['synchrony_index']
        d2 = d['d_raw_control']['oscillation_index']
        d3 = d['d_raw_lesion']['oscillation_index']
        
        fig=plot_oi_si(d0,d1,d2,d3, **kwargs)
        figs.append(fig)          

    if 'si_oi_index_simple' in do_plots:
        d0 = d['d_raw_control']['synchrony_index']
        d1 = d['d_raw_lesion']['synchrony_index']
        d2 = d['d_raw_control']['oscillation_index']
        d3 = d['d_raw_lesion']['oscillation_index']

        
        fig=plot_oi_si_simple(d0,d1,d2,d3, **kwargs)
        figs.append(fig)
    if 'si_oi_index_simple2' in do_plots:
        d0 = d['d_raw_control']['synchrony_index']
        d1 = d['d_raw_lesion']['synchrony_index']
        d2 = d['d_raw_control']['psd_oi']
        d3 = d['d_raw_lesion']['psd_oi']

        
        fig=plot_oi_si_simple(d0,d1,d2,d3, **kwargs)
        figs.append(fig)
    if 'psd' in do_plots:
        d0 = d['d_raw_control']['psd']
        d1 = d['d_raw_lesion']['psd']
        
        figs+=plot_psd(d0,d1, **kwargs)
#         figs.append(fig)
    if 'psd2' in do_plots:
        d0 = d['d_raw_control']['psd2']
        d1 = d['d_raw_lesion']['psd2']
        
        figs+=plot_psd(d0,d1, **kwargs)
#         figs.append(fig) 
    return figs


    
def main(**kwargs):
    

    
    exclude=kwargs.get('exclude',[])
    
    models=kwargs.get('models0', 
                      ['M1', 'M2', 'FS', 'GA', 'GI', 'GP', 'ST','SN',
                       'GP_GP', 'GA_GA', 'GI_GA', 'GI_GI'])

    models=[m for m in models if not ( m in exclude)]
    
    nets=['Net_0', 'Net_1']
    attrs=[
           'firing_rate', 
           'mean_coherence', 
           'phases_diff_with_cohere',
           'psd'
           ]
    
    from_disk=kwargs.get('from_diks',1)
    path=('/home/mikael/results/papers/inhibition/network/'
          +'supermicro/simulate_slow_wave_ZZZ_conn_effect_perturb/')
    path=kwargs.get('data_path', path)
    
    script_name=kwargs.get('script_name', (__file__.split('/')[-1][0:-3]
                                           +'/data'))
    file_name = get_file_name(script_name)
    
    attr_add=['mse_rel_control_fr', 'mse_rel_control_mc',
              'mse_rel_control_pdwc', 'mse_rel_control_mcm',
              'mse_rel_control_oi', 'mse_rel_control_si',
#               'mse_rel_control_psd', 
#               'psd_oi'
              ]
    
#     exclude+=['MS_MS', 'FS_MS', 'MS']
    sd = get_storage(file_name, '')
    d = get_data(models, 
                 nets, 
                 attrs, 
                 path, 
                 from_disk, 
                 attr_add, 
                 sd,
                 **kwargs)

#     pp(kwargs)
    figs = create_figs(d, **kwargs)

    save_figures(figs, script_name, dpi=200)

    pylab.show()        
    

#     pp(d)
    
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
    
if __name__=='__main__':
    main()
    
    
    
