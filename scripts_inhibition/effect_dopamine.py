'''
Created on Sep 10, 2014

@author: mikael
'''
import numpy
import pylab
import os
import toolbox.plot_settings as ps

from matplotlib.font_manager import FontProperties
from os.path import expanduser
from toolbox import misc
from toolbox.data_to_disk import Storage_dic
from toolbox.network.manager import get_storage, save
from toolbox.my_signals import Data_bar
from simulate import get_file_name, save_figures
import pprint
pp=pprint.pprint

# from toolbox.

# def gather(path, nets, models, attrs): 
#     
# 
#     fs=os.listdir(path)
#     d={}
#     i=0
#     for name in fs:
#         if name[-4:]!='.pkl':
#             continue
#         file_name=path+name[:-4]
#         sd = Storage_dic.load(file_name)
#         args=nets+models+attrs
#         dd=sd.load_dic(*args)
#         print name[-12:-4]
#         if name[-12:-4]=='GP-ST-SN':
#             d = misc.dict_update(d, {'GP-ST-SN':dd})
#         else:
#             d = misc.dict_update(d, {name[:-4].split('-')[-1]:dd})
#         i+=1
#     return d

def gather2(path, nets, models, attrs): 
    

    fs=os.listdir(path)
    d={}
    i=0
    for name0 in fs:
        dd={}
        for net in nets:
            name=name0+'/'+net+'.pkl'
            
            
            if not os.path.isfile(path+name):
                print name
                continue
            file_name=path+name[:-4]
            sd = Storage_dic.load(file_name)
            args=nets+models+attrs
            misc.dict_update(dd, sd.load_dic(*args))

        if dd:  
            d = misc.dict_update(d, {name0.split('-')[-1]:dd})
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

def compute_performance(d, nets, models, attrs):       
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
    d0={}
    for run in results.keys():
        for model in models:
            for attr in attrs:
                keys0=['Normal', model, attr] 
                v1=misc.dict_recursive_get(results, keys0)
                d0=misc.dict_recursive_add(d0, keys0, v1)

       
    for run in results.keys():
        for model in models:
            for attr in attrs:
                keys0=['Normal', model, attr]  
                keys3=[run, model, attr]               
                
                v0=misc.dict_recursive_get(d0, keys0)
                v1=misc.dict_recursive_get(results, keys3)
                results=misc.dict_recursive_add(results, keys3, v1/v0)
                
    return results

def generate_plot_data(d, models, attrs, exclude=['all',
                                                  'striatum',
                                                  'GP-ST-SN', 
                                                  'Normal']):
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
        obj=Data_bar(**{'y':numpy.array(l)#/numpy.array(lsum)
                        })
        out=misc.dict_recursive_add(out, [model], obj)
    
    return out, labels

# def plot(d, labels):
#     fig, axs=ps.get_figure(n_rows=4, n_cols=1, w=500.0*0.65*2, h=400.0*0.65*2, fontsize=16,
#                            frame_hight_y=0.6, frame_hight_x=0.8, 
#                            title_fontsize=20, text_usetex=False)        
#     
#     nice_labels={'CTX_M1':r'CTX$\to$$MSN_{D1}$',
#                  'CTX_M2':r'CTX$\to$$MSN_{D2}$',
#                  'CTX_ST':r'CTX$\to$STN',
#                  'MS_MS':r'MSN$\to$MSN',
#                  'M1':r'$MSN_{D1}$',
#                  'M2':r'$MSN_{D2}$',
#                  'GP':r'GPe',
#                  'SN':r'SNr',
#                  'FS_M2':r'FSN$\to$$MSN_{D2}$',
#                  'GP_ST':r'GPe$\to$STN',
#                  'GP_FS':r'GPe$\to$FSN',
#                  'GP_GP':r'GPe$\to$GPe',
#                  'FS_FS':r'FSN$\to$FSN',
#                  'M1_SN':r'$MSN_{D1}$$\to$SNr',
#                  'ST_GP':r'STN$\to$GPe',}
#     
#     for i in range(len(labels)):
#         if labels[i] in nice_labels.keys():
#             labels[i]=nice_labels[labels[i]]
#             
#     
#     for i, key in enumerate(d.keys()):
#         ax=axs[i]
#         
#         d[key].bar2(ax, **{'alpha':False,
#                            'color_axis':1, 
#                            'colors':['r', 'b'],
#                            'edgecolor':'k',
#                            'top_label':False})
#         ax.set_xticklabels(labels, rotation=45, ha='right')
#         
#         
#    
#        
# #         ax.set_title(key)
#         if i==0:
#             ax.legend(('Coherence','Phase relation'), 
#                       bbox_to_anchor=[0.67,1.9])
#             
#             ax.text(-0.11, -1.5, 'MSE normal and control',
#             horizontalalignment='center',
#             verticalalignment='center',
#             transform=ax.transAxes,   
#             rotation='vertical',)
# #             ax.legend(bbox_to_anchor=[1.45,1.15])
#             
#     axs=fig.get_axes()
#     ps.shift('upp', axs, 0.2, n_rows=len(axs), n_cols=1)
#     ps.shift('down', axs, 0.08, n_rows=len(axs), n_cols=1)
# #     ps.shift('left', axs, 0.1, n_rows=len(axs), n_cols=1)
#     ps.shift('right', axs, 0.05, n_rows=len(axs), n_cols=1)
#     
#     labels=['GP vs GP', 'TI vs TI', 'TI vs TA', 'TA vs TA']
#     ylims=[[0, 3.5]]*2+[[0, 2.5]]*2
#     for i, ax in enumerate(axs):      
#         
#         ax.my_set_no_ticks(yticks=3)
#         ax.set_ylim(ylims[i])
#         pos=ax.get_position()#pylab.getp(ax, 'position')
#         print pos.height
#         ax.text(0.5, 0.85, labels[i],
#             horizontalalignment='center',
#             verticalalignment='center',
#             transform=ax.transAxes)     
#         ax.set_ylabel('')
#         if i==3:
#             ax.set_xlabel('Connection with dopamine modulation removed')
#         else:
#             ax.my_remove_axis(xaxis=True)
#             ax.set_xlabel('')
# 
# #     pylab.show()
        

def gs_builder(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.05 ), 
              hspace=kwargs.get('hspace', 1. / n_cols ))

    iterator = [[slice(1,7), slice(2,7)],
                [slice(1,7), slice(7,9)]]
    
    return iterator, gs, 

def plot_coher(d, labelsy, labelsx=[], title_name='Slow wave'):
    fig, axs=ps.get_figure2(n_rows=8, n_cols=9, w=700, h=700, fontsize=24,
                            frame_hight_y=0.5, frame_hight_x=0.7, 
                            title_fontsize=24,
                            gs_builder=gs_builder) 
#     fig, axs=ps.get_figure(n_rows=4, n_cols=1, w=500.0*0.65*2, h=400.0*0.65*2, fontsize=16,
#                            frame_hight_y=0.6, frame_hight_x=0.8, 
#                            title_fontsize=20, text_usetex=False)        
    
#     nice_labels={'CTX_M1':r'CTX$\to$$MSN_{D1}$ ',
#                  'CTX_M2':r'CTX$\to$$MSN_{D2}$ ',
#                  'CTX_ST':r'CTX$\to$STN ',
#                  'MS_MS':r'MSN$\to$MSN ',
#                  'M1':r'$MSN_{D1}$ ',
#                  'M2':r'$MSN_{D2}$ ',
#                  'GP':r'GPe ',
#                  'SN':r'SNr ',
#                  'FS_M2':r'FSN$\to$$MSN_{D2}$ ',
#                  'GP_ST':r'GPe$\to$STN ',
#                  'GP_FS':r'GPe$\to$FSN ',
#                  'GP_GP':r'GPe$\to$GPe ',
#                  'FS_FS':r'FSN$\to$FSN ',
#                  'M1_SN':r'$MSN_{D1}$$\to$SNr ',
#                  'ST_GP':r'STN$\to$GPe ',}

    from effect_conns import nice_labels
    groupings=['Coherence','Phase relation']
# 
#     nice_labels2={'GA_GA':r'TA vs TA',
#                   'GI_GA':r'TI vs TA',
#                   'GI_GI':r'TI vs TI',
#                   'GP_GP':r'GP vs GP'}

    for i in range(len(labelsy)):
        if labelsy[i] in nice_labels(version=0).keys():
            labelsy[i]=nice_labels(version=0)[labelsy[i]]
            
    l0=[]
    l2=[]
    for key in sorted(d['bar_obj'].keys()):
        
#         if key in ['GA_GA', 'GI_GA', 'GP_GP']:
#             d['bar_obj'][key].y[1,:]+=1
        l0.append(d['bar_obj'][key].y[0,:])
        l2.append(d['bar_obj'][key].y[1,:])
        labelsx.append(key)

    z=numpy.transpose(numpy.array(l0+l2))
    
    for i in range(len(labelsx)):
        if labelsx[i] in nice_labels(version=1).keys():
            labelsx[i]=nice_labels(version=1)[labelsx[i]]
            
     
    _vmin=0
    _vmax=4
    stepx=1
    stepy=1
    startx=0
    starty=0
    stopy=14
    stopx=8
    maxy=14
    maxx=8
    
    posy=numpy.linspace(0.5,maxy-0.5, maxy)
    posx=numpy.linspace(0.5,maxx-0.5, maxx)
    axs[1].barh(posy,numpy.mean(z,axis=1)[::-1], align='center', color='0.5')
    axs[1].plot([1,1], [0,stopy], 'k', linewidth=3, linestyle='--')
    x1,y1=numpy.meshgrid(numpy.linspace(startx, stopx, maxx+1),
                   numpy.linspace(stopy, starty, maxy+1))
    
    im = axs[0].pcolor(x1, y1, z, cmap='jet', 
                        vmin=_vmin, vmax=_vmax
                       )
    axs[0].set_yticks(posy)
    axs[0].set_yticklabels(labelsy[::-1])
    axs[0].set_xticks(posx)
    axs[0].set_xticklabels(labelsx*2, rotation=70, ha='right')
    axs[0].set_ylim([0,maxy])
    axs[0].text(0.05, -0.31, "Coherence", transform=axs[0].transAxes)
    axs[0].text(0.55, -0.31, "Phase shift", transform=axs[0].transAxes)
    axs[1].text(0.5, -0.12, "Mean", 
                transform=axs[1].transAxes,
                ha='center',
                rotation=0)
    axs[1].text(0.5, -0.19, "effect", 
                transform=axs[1].transAxes,
                ha='center',
                rotation=0)    
    font0 = FontProperties()
    font0.set_weight('bold')
    axs[1].text(1.45, 0.5, title_name,
                fontsize=28,
                va='center',
                 transform=axs[0].transAxes,
                                rotation=270,
                                fontproperties=font0)
    axs[0].text(-0.6, 0.9, "Connection without dop. effect", transform=axs[0].transAxes,
                rotation=90)
        
    axs[1].my_remove_axis(xaxis=False, yaxis=True)
    axs[1].my_set_no_ticks(xticks=2)
    axs[1].set_xlim([0,maxy])
    axs[1].set_xlim([0,2])
#     axs[1].set_xticks([0.04, 0.12])


    box = axs[0].get_position()
    axColor=pylab.axes([box.x0+0.1*box.width, 
                        box.y0+box.height+box.height*0.08, 
                        box.width*0.8, 
                        0.02])
    #     axColor = pylab.axes([0.05, 0.9, 1.0, 0.05])
    cbar=pylab.colorbar(im, cax = axColor, orientation="horizontal")
    cbar.ax.set_title('MSE control vs lesion rel. base model')#, rotation=270)
    from matplotlib import ticker
    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()

    
    return fig
#     pylab.show()
    
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
    
# if __name__=='__main__':
def main(**kwargs):
    
    exclude=kwargs.get('exclude',[])
    models=['GP_GP', 'GA_GA', 'GI_GA', 'GI_GI']
    models=[m for m in models if not ( m in exclude)]
    
    nets=['Net_0', 'Net_1']
    attrs=['mean_coherence', 'phases_diff_with_cohere']
    
    from_disk=kwargs.get('from_diks',0)
    path=('/home/mikael/results/papers/inhibition/network/'
          +'supermicro/simulate_slow_wave_ZZZ_dop_effect_perturb/')
    path=kwargs.get('data_path', path)
        
    script_name=kwargs.get('script_name', (__file__.split('/')[-1][0:-3]
                                           +'/data'))

    file_name = get_file_name(script_name)
 
    sd = get_storage(file_name, '')
    d={}
    if not from_disk:
    
#         d['raw']=gather(path, nets, models, attrs)
        d['raw']=gather2(path, nets, models, attrs)
        d['data']=extract_data(d['raw'], nets, models, attrs)
        d['performance']=compute_performance(d['data'], nets, models, attrs)
        d['bar_obj'], d['labels']=generate_plot_data(d['performance'],
                                                     models, 
                                                     attrs)

        save(sd, d)
    else:
        filt=['bar_obj']+models+['labels']
        d = sd.load_dic(*filt)

    figs=[]
    figs.append(plot_coher(d, d['labels'], title_name=kwargs.get('title')))
    pylab.show()
    
    save_figures(figs, script_name)
    
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
        
    
    