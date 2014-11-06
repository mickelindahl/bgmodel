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
from matplotlib.font_manager import FontProperties
# from toolbox.

def gather(path, nets, models, attrs): 
    

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
            v=max(val.y[2:20])
        if keys[-1]=='firing_rate':
            v=numpy.mean(val.y)
            
        out=misc.dict_recursive_add(out,  keys, v)
    return out             
def compute_performance(d, nets, models, attr):       
    results={}
    
    for run in d.keys():
        if run=='no_pert':
            continue
        for model in models:
            for attr in attrs:
                
                keys_c= ['no_pert', 'Net_0', model, attr]
                keys_l= ['no_pert', 'Net_1', model, attr]
                try:
                    v_control0=misc.dict_recursive_get(d,keys_c)
                except:
                    continue
                v_lesion0=misc.dict_recursive_get(d,keys_l)
                
                
                s,t,_,x=run.split('_')
                x=float(x)
                conn=s+'_'+t
                keys1=[run, 'Net_0', model, attr]
                keys2=[run, 'Net_1', model, attr]
                keys3=['control',conn, model, attr]
                keys4=['lesion',conn, model, attr]
                
                add=0
                if model=='M1':
                    add=100
                    
                if model=='GI':
                    add=-100
                    
                if model=='GP':
                    add=-200 
              
                if model=='SN':
                    add=200                
                v1=misc.dict_recursive_get(d, keys1)
#                 try:
                v2=misc.dict_recursive_get(d, keys2)
#                 except:
#                     v2=0

                v1=v1#-v_control0+add
                v2=v2#-v_lesion0+add
                
                if not misc.dict_haskey(results, keys3):
                    results=misc.dict_recursive_add(results,  keys3, 
                                                [(x, v1)])
                    results=misc.dict_recursive_add(results,  keys4, 
                                                [(x, v2)])
                else:
                    l=misc.dict_recursive_get(results, keys3)
                    l.append((x, v1))

                    l=misc.dict_recursive_get(results, keys4)
                    l.append((x, v2))
                
                l=misc.dict_recursive_get(results, keys3)
                if not (1, v_control0) in l:             
                    l.append((1, v_control0))
                    
                    l=misc.dict_recursive_get(results, keys4)
                    l.append((1, v_lesion0)) 
                
                
    for keys, val in misc.dict_iter(results):
        x,y=zip(*val)
        
#         # Add midpoint
#         x.append(1.)
#         y.append(100.)
        
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
        g=(-y[0]+a1*y[1]-a2*y[2]+a2*y[4]-a1*y[5]+y[6])/(60*h)
        d=misc.dict_recursive_add(gradients, keys, g)         
    return results, gradients

def generate_plot_data_raw(d, models, attrs, exclude=[]):
    out={}
    
    labelsx=[]
    labelsy=[]
    for k in sorted(d.keys()):
        if k in exclude:
            continue        
        labelsy.append(k)
    res={}
    for keys, val in misc.dict_iter(d):
        res=misc.dict_recursive_add(res, keys[0:2], val)
    
    out={}
    for i, k in enumerate(labelsy):
        for model in sorted(models):
            v=res[k][model]
            if model not in out.keys():
                out.update({model:{'x':[v[0]],
                                   'y':[v[1]]}})
            else:
                out[model]['x'].append(v[0])
                out[model]['y'].append(v[1])
    
    for keys, val in misc.dict_iter(out):
        print numpy.array(val).shape
        val=numpy.array(val).ravel()
        print val.shape
        out=misc.dict_recursive_set(out, keys, val)

    dd0={'labelsy':['M1', 'M2', 'FS', 'GA', 'GI', 'GP','SN', 'ST'], 
         'labelsx_meta':labelsy,
         'labelsx':[0.25,0.5,0.75,1,1.25,1.5,1.75]*16,
         'x':[],
         'y':[]}
    dd1={'labelsy':['GA_GA', 'GI_GA', 'GI_GI', 'GP_GP'], 
         'labelsx_meta':labelsy,
         'labelsx':[0.25,0.5,0.75,1,1.25,1.5,1.75]*16,
         'x':[],
         'y':[]}      
      
    for key in dd0['labelsy']:
        v=out[key]
        dd0['y'].append(v['y'])
        dd0['x'].append(v['x'])

    for key in dd1['labelsy']:
        v=out[key]
        dd1['y'].append(v['y'])
        dd1['x'].append(v['x'])
    
    for out in [dd0,dd1]:   
        out['x']=numpy.array(out['x'])
        out['y']=numpy.array(out['y'])
        
    return dd0,dd1

def generate_plot_data(d, models, attrs, exclude=[]):
    out={}
    
    labelsx=[]
    labelsy=[]
    for k in sorted(d.keys()):
        if k in exclude:
            continue        
        labelsy.append(k)
    res={}
    for keys, val in misc.dict_iter(d):
        res=misc.dict_recursive_add(res, keys[0:2], val)
    
    out={}
    for i, k in enumerate(labelsy):
        for model in sorted(models):
            
            
            v=res[k][model]
            if model not in out.keys():
                out[model]=[v]
            else:
                out[model].append(v)
    
    dd0={'labelsy':labelsy, 
         'labelsx':['FS', 'M1', 'M2', 'GA', 'GI', 'GP','SN', 'ST'],
         'z':[]}
    dd1={'labelsy':labelsy, 
         'labelsx':['GA_GA', 'GI_GA', 'GI_GI', 'GP_GP'],
         'z':[]}      
      
    for key in sorted(out.keys()):
        v=out[key]
        if key in dd0['labelsx']:
            dd0['z'].append(v)
        else:
            dd1['z'].append(v)
       
    return dd0,dd1

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

#     pylab.show()
        
def gs_builder2(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.05 ), 
              hspace=kwargs.get('hspace', 0.05 ))

    iterator = [[slice(1,2),slice(0,4)],
                [slice(2,4),slice(0,4)],
                [slice(4,6),slice(0,4)],
                [slice(1,2),slice(4,8)],
                [slice(2,4),slice(4,8)],
                [slice(4,6),slice(4,8)]]
    
    return iterator, gs, 

def gs_builder(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.05 ), 
              hspace=kwargs.get('hspace', 1. / n_cols ))

    iterator = [[slice(1,8),slice(0,4)],
                [slice(1,8),slice(4,8)],
                ]
    
    return iterator, gs, 

def plot4(d0, d1, d2, d3):
    fig, axs=ps.get_figure2(n_rows=8, n_cols=9, w=1000, h=600, fontsize=24,
                            frame_hight_y=0.5, frame_hight_x=0.7, 
                            title_fontsize=24,
                            gs_builder=gs_builder2) 

    
    d00={'labelsx':d0['labelsx'],
         'labelsx_meta':d2['labelsx_meta'],
         'labelsy':['M1','M2'],
         'x':d0['x'],
         'y':d0['y'][0:2,]}
    d01={'labelsx':d0['labelsx'],
         'labelsx_meta':d2['labelsx_meta'],
         'labelsy':['FS', 'GA', 'GI', 'GP','SN', 'ST'],
         'x':d0['x'],
         'y':d0['y'][2:,]}
    
    d20={'labelsx':d2['labelsx'],
         'labelsx_meta':d2['labelsx_meta'],
         'labelsy':['M1','M2'],
         'x':d2['x'],
         'y':d2['y'][0:2,]}
    d21={'labelsx':d2['labelsx'],
         'labelsx_meta':d2['labelsx_meta'],
         'labelsy':['FS', 'GA', 'GI', 'GP','SN', 'ST'],
         'x':d2['x'],
         'y':d2['y'][2:,]}    
    
    
    startx=0
    starty=0
    images=[]
    for ax, d in zip(axs,[ d00, d01, d1, d20, d21, d3]):
        stopy=len(d['labelsy'])
        stopx=len(d['labelsx'])
        
        
        
        stopx_meta=len(d['labelsx_meta'])
        posy=numpy.linspace(.5,stopy-.5, stopy)
        posx=numpy.linspace(3.5,stopx-3.5, stopx_meta)
        x=numpy.linspace(0, stopx, stopx_meta+1)
        x1,y1=numpy.meshgrid(numpy.linspace(startx, stopx, stopx+1), 
                             numpy.linspace(starty, stopy, stopy+1)) 
        
        
        im = ax.pcolor(x1, y1, d['y'][::-1,], cmap='jet')       
        for xx in x:
            ax.plot([xx,xx],[0,stopy], 'k', linewidth=1.)
        images.append(im)                
        ax.set_yticks(posy)
        ax.set_yticklabels(d['labelsy'][::-1])
        ax.set_xticks(posx)
        ax.set_xticklabels(d['labelsx_meta'], rotation=70, 
                           ha='center', 
                           fontsize=20)
        ax.set_ylim([0,stopy])
        ax.set_xlim([0,stopx])
        
    for ax, label, im in zip([axs[3], axs[4],axs[5]], 
                             ['', 'Firing rate (Hz)', 'Coherence'],
                             [images[3],images[4],images[5]]):
        
        box = ax.get_position()
        axColor = pylab.axes([box.x0 + box.width * 1.1, 
                              box.y0+box.height*0.1, 
                              0.02, 
                              box.height*0.8])
        cbar=pylab.colorbar(im, cax = axColor, orientation="vertical")
        cbar.ax.set_ylabel(label, rotation=270)
        from matplotlib import ticker
        
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()
    
    axs[0].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)   
    axs[1].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)   
    axs[3].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True) 
    axs[4].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True) 
    axs[5].my_remove_axis(xaxis=False, yaxis=True,keep_ticks=True) 
    axs[0].text(0.35, 1.05, "Control", transform=axs[0].transAxes)     
    axs[3].text(0.35, 1.05, "Lesion", transform=axs[3].transAxes)  
       
    font0 = FontProperties()
    font0.set_weight('bold')
    axs[0].text(0.8, 1.2, "Slow wave", transform=axs[0].transAxes, 
                fontproperties=font0         )
def plot3(d0, d1, d2, d3):
    fig, axs=ps.get_figure2(n_rows=9, n_cols=9, w=1000, h=600, fontsize=24,
                            frame_hight_y=0.5, frame_hight_x=0.7, 
                            title_fontsize=24,
                            gs_builder=gs_builder) 

#     d0['z'][0][0]=1000.0
    z=numpy.transpose(numpy.array(d0['z']+d1['z']))
    z2=numpy.transpose(numpy.array(d2['z']+d3['z']))
    
    labelsy=d0['labelsy']+d1['labelsy']
    labelsx=d0['labelsx']+d1['labelsx']
    _vmin=-200
    _vmax=200
    stepx=1
    stepy=1
    startx=0
    starty=0
    stopy=16
    stopx=12
    maxy=16
    maxx=12
    
    posy=numpy.linspace(0.5,maxy-0.5, maxy)
    posx=numpy.linspace(0.5,maxx-0.5, maxx)
#     axs[1].barh(posy,numpy.mean(z,axis=1)[::-1], align='center', color='k')
    
    x1,y1=numpy.meshgrid(numpy.linspace(startx, stopx, maxx+1),
                         numpy.linspace(starty, stopy, maxy+1))
    
    im = axs[0].pcolor(x1, y1, z, cmap='coolwarm', 
                        vmin=_vmin, vmax=_vmax
                       )
    
    axs[0].set_yticks(posy)
    axs[0].set_yticklabels(labelsy[::-1])
    axs[0].set_xticks(posx)
    axs[0].set_xticklabels(labelsx, rotation=70, ha='right')
    axs[0].set_ylim([0,maxy])
#     axs[0].set_zlim([-200,200])
    axs[0].text(0.1, 1.02, "Firing rate", transform=axs[0].transAxes)
    axs[0].text(0.6, 1.02, "Coherence", transform=axs[0].transAxes)

    font0 = FontProperties()
    font0.set_weight('bold')
    axs[0].text(0.35, 1.12, "Control", transform=axs[0].transAxes,
                fontproperties=font0)
#     axs[0].text(1.3, 0.65, "Mean effect", transform=axs[0].transAxes,
#                 rotation=270)

    im = axs[1].pcolor(x1, y1, z2, cmap='coolwarm', 
                        vmin=_vmin, vmax=_vmax
                       )
    
    axs[1].set_yticks(posy)
#     axs[0].set_yticklabels(labelsy[::-1])
    axs[1].set_xticks(posx)
    axs[1].set_xticklabels(labelsx, rotation=70, ha='right')
    axs[1].set_ylim([0,maxy])
#     axs[0].set_zlim([-200,200])
    axs[1].text(0.1, 1.01, "Firing rate", transform=axs[1].transAxes)
    axs[1].text(0.6, 1.01, "Coherence", transform=axs[1].transAxes)
    from matplotlib.font_manager import FontProperties
    font0 = FontProperties()
    font0.set_weight('bold')
    axs[1].text(0.35, 1.12, "Lesion", transform=axs[1].transAxes,
                fontproperties=font0)
#     axs[1].text(1.3, 0.65, "Mean effect", transform=axs[0].transAxes,
#                 rotation=270)
        
    axs[1].my_remove_axis(xaxis=False, yaxis=True)
#     axs[1].my_set_no_ticks(xticks=2)
#     axs[1].set_ylim([0,maxy])
#     axs[1].set_xticks([0.04, 0.12])
    
    
    box = axs[1].get_position()
    axColor = pylab.axes([box.x0 + box.width * 1.1, 
                          box.y0+box.height*0.1, 
                          0.02, 
                          box.height*0.8])
    cbar=pylab.colorbar(im, cax = axColor, orientation="vertical")
    cbar.ax.set_ylabel('Gradient', rotation=270)
    from matplotlib import ticker
    
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    
def plot2(d, labels):
    fig, axs=ps.get_figure2(n_rows=8, n_cols=6, w=700, h=700, fontsize=24,
                            frame_hight_y=0.5, frame_hight_x=0.7, 
                            title_fontsize=24,
                            gs_builder=gs_builder) 
#     fig, axs=ps.get_figure(n_rows=4, n_cols=1, w=500.0*0.65*2, h=400.0*0.65*2, fontsize=16,
#                            frame_hight_y=0.6, frame_hight_x=0.8, 
#                            title_fontsize=20, text_usetex=False)        
    
    nice_labels={'CTX_M1':r'CTX$\to$$MSN_{D1}$ ',
                 'CTX_M2':r'CTX$\to$$MSN_{D2}$ ',
                 'CTX_ST':r'CTX$\to$STN ',
                 'MS_MS':r'MSN$\to$MSN ',
                 'M1':r'$MSN_{D1}$ ',
                 'M2':r'$MSN_{D2}$ ',
                 'GP':r'GPe ',
                 'SN':r'SNr ',
                 'FS_M2':r'FSN$\to$$MSN_{D2}$ ',
                 'GP_ST':r'GPe$\to$STN ',
                 'GP_FS':r'GPe$\to$FSN ',
                 'GP_GP':r'GPe$\to$GPe ',
                 'FS_FS':r'FSN$\to$FSN ',
                 'M1_SN':r'$MSN_{D1}$$\to$SNr ',
                 'ST_GP':r'STN$\to$GPe ',}
    groupings=['Coherence','Phase relation']

    nice_labels2={'GA_GA':r'TA vs TA',
                  'GI_GA':r'TI vs TA',
                  'GI_GI':r'TI vs TI',
                  'GP_GP':r'GP vs GP'}

    for i in range(len(labels)):
        if labels[i] in nice_labels.keys():
            labels[i]=nice_labels[labels[i]]
            
    
    l0=[]
    l2=[]
    labels2=[]
    for key in sorted(d['bar_obj'].keys()):
        
#         if key in ['GA_GA', 'GI_GA', 'GP_GP']:
#             d['bar_obj'][key].y[1,:]+=1
        l0.append(d['bar_obj'][key].y[0,:])
        l2.append(d['bar_obj'][key].y[1,:])
        labels2.append(nice_labels2[key])

    z=numpy.transpose(numpy.array(l0+l2))
    
    _vmin=0
    _vmax=0.16
    stepx=1
    stepy=1
    startx=0
    starty=0
    stopy=15
    stopx=8
    maxy=15
    maxx=8
    
    posy=numpy.linspace(0.5,maxy-0.5, maxy)
    posx=numpy.linspace(0.5,maxx-0.5, maxx)
    axs[1].barh(posy,numpy.mean(z,axis=1)[::-1], align='center', color='k')
    
    x1,y1=numpy.meshgrid(numpy.linspace(startx, stopx, maxx+1),
                   numpy.linspace(stopy, starty, maxy+1))
    
    im = axs[0].pcolor(x1, y1, z, cmap='Greys', 
                        vmin=_vmin, vmax=_vmax
                       )
    axs[0].set_yticks(posy)
    axs[0].set_yticklabels(labels[::-1])
    axs[0].set_xticks(posx)
    axs[0].set_xticklabels(labels2*2, rotation=70, ha='right')
    axs[0].set_ylim([0,maxy])
    axs[0].text(0.1, 1.02, "Coherence", transform=axs[0].transAxes)
    axs[0].text(0.6, 1.02, "Phase shift", transform=axs[0].transAxes)
    axs[0].text(1.3, 0.65, "Mean effect", transform=axs[0].transAxes,
                rotation=270)
        
    axs[1].my_remove_axis(xaxis=False, yaxis=True)
    axs[1].my_set_no_ticks(xticks=2)
    axs[1].set_ylim([0,maxy])
    axs[1].set_xticks([0.04, 0.12])


    box = axs[0].get_position()
    axColor=pylab.axes([box.x0+0.1*box.width, 
                        box.y0+box.height+box.height*0.15, 
                        box.width*0.8, 
                        0.05])
    #     axColor = pylab.axes([0.05, 0.9, 1.0, 0.05])
    cbar=pylab.colorbar(im, cax = axColor, orientation="horizontal")
    cbar.ax.set_title('Normalized MSE control vs lesion')#, rotation=270)
    from matplotlib import ticker
    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()

    
    
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
    
if __name__=='__main__':
    models=['M1', 'M2', 'FS', 'GA', 'GI', 'GP', 'ST','SN',
            'GP_GP', 'GA_GA', 'GI_GA', 'GI_GI']
    nets=['Net_0', 'Net_1']
    attrs=[
           'firing_rate', 
           'mean_coherence', 
#            'phases_diff_with_cohere'
           ]
    path=('/home/mikael/results/papers/inhibition/network/'
          +'supermicro/simulate_slow_wave_ZZZ_conn_effect_perturb/')
 
    home = expanduser("~")
    script_name=__file__.split('/')[-1][0:-3]
    file_name = get_file_name(script_name)
    from_disk=0
    
    sd = get_storage(file_name, '')
    d={}
    if not from_disk:
    
        d['raw']=gather(path, nets, models, attrs)
        d['data']=extract_data(d['raw'], nets, models, attrs)
        out=compute_performance(d['data'], nets, models, attrs)
        d['change_raw'], d['gradients']=out

        d['d0_raw'], d['d1_raw']=generate_plot_data_raw(d['change_raw']['control'],
                                                        models, 
                                                        attrs)
        d['d2_raw'], d['d3_raw']=generate_plot_data_raw(d['change_raw']['lesion'],
                                                        models, 
                                                        attrs)
        d['d0'], d['d1']=generate_plot_data(d['gradients']['control'],
                                                    models, 
                                                    attrs)
        d['d2'], d['d3']=generate_plot_data(d['gradients']['lesion'],
                                                    models, 
                                                    attrs)
        save(sd, d)
    else:
#         filt=['performance']+models
        d = sd.load_dic()
#         d2=sd.load_dic()
#         d = sd.load_dic()
    pp(d['d0'])
    pp(d['d1'])
#     plot_raw(d, d_keys, attr='mean_coherence')
#     plot_raw(d, d_keys, attr='phases_diff_with_cohere')
    plot3(d['d0'], d['d1'], d['d2'], d['d3'])
    plot4(d['d0_raw'], d['d1_raw'], d['d2_raw'], d['d3_raw'])
#     plot(d['bar_obj'], d['labels'])
    pylab.show()
#     pp(d)
    
    
    
    
    
    