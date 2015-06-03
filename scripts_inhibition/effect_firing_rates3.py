'''
Created on Nov 14, 2014

@author: mikael
'''
import matplotlib.gridspec as gridspec
import numpy
import toolbox.plot_settings as ps
import pylab
import pprint
pp=pprint.pprint

from scripts_inhibition import effect_conns
from toolbox import misc
from simulate import save_figures
from toolbox.network.manager import get_storage, save
from simulate import get_file_name, save_figures
from toolbox.data_to_disk import Storage_dic
def create_name_beta(file_name):
    print file_name
    l=file_name.split('_')
    if file_name.split('/')[-1] in ['std', 'params', 'jobbs']:
        return False

#     if file_name.split('/')[5]=='0000':       
#         return False        
    amp=[1.0, 1.05, 1.1, 1.15, 1.2]
    s=[]
    for i in [1, 7, 9, 11, 13]:
        if i ==11:
            s.append(l[i].split('-')[0])
#         elif i==13:
#             i=int(int(l[5])/393)
#             s.append(str(amp[i]))
            
        else:
            s.append(l[i])
    print s
    return '_'.join(s)

def create_name_sw(file_name):
    print file_name
    l=file_name.split('_')
    if file_name.split('/')[-1] in ['std', 'params', 'jobbs']:
        return False


    if l[7]=='control':       
        return False
    
    if len(l)>15:
        return False
    
    s=[]
    for i in [1, 2, 8, 10, 12, 13]:
        if i ==12:
            s.append(l[i].split('-')[0])
        elif i ==13:
            s.append(l[i].split('-')[1])
        else:
            s.append(l[i])
    print s
    return '_'.join(s)



#     tp=file_name.split('_')[1]
#     if len(file_name.split('-'))>=3:
#         
#         l=file_name.split('-')[-3].split('_')
#         return l[-4]+'_'+l[-3]+'_'+l[-2]+'_'+l[-1]
# #         return file_name.split('/')[-1]
#     else:
#         return file_name


def gs_builder(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.02 ), 
              hspace=kwargs.get('hspace', 0.1 ))

    iterator = [
                [slice(0,1), slice(0,1)],
                [slice(0,1), slice(1,2)],
                [slice(0,1), slice(3,4)],
                [slice(0,1), slice(4,5)]]
    return iterator, gs

def postprocess(d):
    out={}
    ylabels0=sorted(d.keys())

    xlabels=sorted(d[ylabels0[0]]['Net_0'].keys())
    i,j=0,0
    for tp in ['beta','sw']:
        ylabels=[]
        for key in ylabels0:
            if key[0]!=tp[0] or len(d[key].keys())<2:
                continue
            ylabels.append(key)
            j+=1
#             if j==396:
#                 pass
            
            for net in d[key].keys():
                mrs=[]
                for model in xlabels:
                    obj=d[key][net][model]['spike_statistic']
                    mrs.append(obj.rates['mean'])
                
                keys=[tp, net, 'mean_rates']    
                if not misc.dict_haskey(out, keys):
                    misc.dict_recursive_add(out, keys, [mrs])
                    if net=='Net_0': i+=1
                else:
                    out[tp][net]['mean_rates'].append(mrs)
                    if net=='Net_0':i+=1
            if not i==j:
                print d[key].keys(), i,j
                raise
#             print j,i
        for net in d[key].keys(): 
            keys=[tp, net, 'xlabels']    
            misc.dict_recursive_add(out, keys, xlabels)
            keys=[tp, net, 'ylabels']  
            misc.dict_recursive_add(out, keys, ylabels)
        
    pp(out)
    for tp in out.keys():
        for net in out[tp].keys():
            out[tp][net]['mean_rates']=numpy.array(out[tp][net]['mean_rates'])
    
    return out

def process_exp(exp, tr):
    out={}
    
    put_at={'STN':3,
            'TA':0,
            'TI':1,
            'all':2}
    for keys, val in misc.dict_iter(exp):
        if keys[-1]=='CV':
            continue
        keys2=[tr[keys[1]],tr[keys[2]], 'mean_rates']
        
        if not misc.dict_haskey(out, keys2):
            a=numpy.zeros(4)
            a[put_at[keys[0]]]=val
            misc.dict_recursive_add(out, keys2,a )
        else:
            out[tr[keys[1]]] [tr[keys[2]]] ['mean_rates'][put_at[keys[0]]]=val

    pp(out)
    return out

def plot_rates(ax, z, xlabels, ylabels):
    
    x1,y1=numpy.meshgrid(numpy.linspace(0, len(xlabels),  len(xlabels)+1), 
                         numpy.linspace(0, len(ylabels),  len(ylabels)+1)) 
        
    im = ax.pcolor(x1, y1, z[::-1,], cmap='coolwarm')       
    
    posx=numpy.linspace(0, len(xlabels),  len(xlabels))
    posy=numpy.linspace(0, len(ylabels),  len(ylabels))
    
    step=10
    ax.set_xticks(posx)
    ax.set_yticks(posy[::step])
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels[::step])
    return im
    
def show_rates(d, **k):

    scale=k.get('scale',2)
    
    kw={'n_rows':1, 
        'n_cols':5, 
        'w':int(72/2.54*17.6)*2*scale, 
        'h':int(72/2.54*17.6)*1.5*scale, 
        'fontsize':7*scale,
        'title_fontsize':7*scale,
        'frame_hight_y':0.5,
        'frame_hight_x':0.7,
        'gs_builder':gs_builder}
    
    fig, axs=ps.get_figure2(**kw) 
    
    i=0
    images=[]
    for tp in ['beta', 'sw']:
        for net in sorted(d[tp].keys()):
            ax=axs[i]
            dd=d[tp][net]
            im=plot_rates(ax, dd['mean_rates'], dd['xlabels'], dd['ylabels'])
            ax.set_title(tp+' '+net)
            im.set_clim([0, 35])
            images.append(im)
            i+=1
    for i in [1,3]:
        axs[i].my_remove_axis(xaxis=False, yaxis=True,keep_ticks=True)
    
    ax=axs[1]
    im=images[1]
    box = ax.get_position()
    axColor = pylab.axes([box.x0 + box.width *1.03, 
                          box.y0+box.height*0.1, 
                          0.01, 
                          box.height*0.8])
    cbar=pylab.colorbar(im, cax = axColor, orientation="vertical")
    cbar.ax.set_ylabel('Rate (Hz)', rotation=270)
    from matplotlib import ticker
    
    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params(direction='in', length=1, width=0.5) 
    
    return fig
   
def fitting(d0, d1):
    idx_list=[]
    pos_list=[]
    for tp in['beta', 'sw']:
        e=0
        for net, sl in zip(['Net_0', 'Net_1'], [slice(2,4), slice(0,4)]):
            z=d0[tp][net]['mean_rates'][:,sl]
            target=d1[tp][net]['mean_rates'][sl]
            target=numpy.array([target]*z.shape[0])
            if e is 0:
                e=z-target
            else:
                e=numpy.concatenate((z-target, e),axis=1)
        
        e**=2
        e=numpy.sqrt(numpy.mean(e, axis=1))
        idx=numpy.argsort(e)
#         idx_list.append(idx)
#         l=[]
#         for i, _id in enumerate(idx_list[-2]):
#             j=list(idx_list[-1]).index(_id)
#             l.append([i,j])
#         l=numpy.array(l)
#         pos_list.append(l)
#         e=numpy.mean(l,axis=1)
        idx=numpy.argsort(e)
        
#         pp(list(l[idx,:]))
            
#         print idx
#         print e[idx]
        print tp
        for _id in idx[:10]:
            print d0[tp]['Net_0']['ylabels'][_id], d1[tp]['Net_0']['mean_rates'][:], numpy.round(d0[tp]['Net_0']['mean_rates'][_id,:],1),e[_id]
            print d0[tp]['Net_1']['ylabels'][_id], d1[tp]['Net_1']['mean_rates'][:], numpy.round(d0[tp]['Net_1']['mean_rates'][_id,:],1)
         
        
from_disk=2

attrs=['spike_statistic']
models=['GP', 'GI', 'GA', 'ST']
nets=['Net_0', 'Net_1']
paths=[]
paths.append('/home/mikael/results/papers/inhibition/network/'
             +'milner/simulate_beta_new_beginning_slow6/')
paths.append('/home/mikael/results/papers/inhibition/network/'
             +'milner/simulate_slow_wave_new_beginning_slow6/')

script_name=(__file__.split('/')[-1][0:-3]+'/data')
file_name = get_file_name(script_name)

sd = Storage_dic.load(file_name, **{'force_update':False})

if from_disk==0:
    d={}
    for path, create_name in zip(paths, [ create_name_beta,create_name_sw]):
        d_tmp=effect_conns.gather(path, nets, models, attrs, **{'name_maker':create_name})
        misc.dict_update(d, d_tmp)
    
    save(sd, d)
    pp(d)
elif from_disk==1:
    d = sd.load_dic(*(nets+models+['spike_statistic']),
                    **{'star':[0]})
    pp(d)
    pp(sorted(d.keys()))
    
    dd=postprocess(d)
    misc.dict_update(d, dd)
    save(sd, d)
    
elif from_disk==2:
    keys_iterator=[]
    for key in ['mean_rates', 'xlabels', 'ylabels']:
        keys_iterator+=[['beta', 'Net_0', key],
                       ['beta', 'Net_1', key],
                       ['sw', 'Net_0', key],
                       ['sw', 'Net_1', key],
                       ]
    d = sd.load_dic(**{'keys_iterator':keys_iterator})
#     d=sd.load_dic(*filt)
    pp(d)
    
from scripts_inhibition.oscillation_common import mallet2008
exp=mallet2008()



pp(exp)
translation={'all':'GP',
             'STN':'ST',
             'TI':'GI',
             'TA':'GA',
             'activation':'beta',
             'slow_wave':'sw',
             'control':'Net_0',
             'lesioned':'Net_1'}
exp=process_exp(exp, translation)



fitting(d, exp)
