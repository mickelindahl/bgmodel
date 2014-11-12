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
from toolbox.network.manager import get_storage, save, load
from toolbox.my_signals import Data_bar
from simulate import get_file_name
import pprint
pp=pprint.pprint

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
            args=[[keys],
                  [v]]
            
        if keys[-1]=='mean_coherence':
            v_max=max(val.y[2:20])
            v=numpy.mean(val.y[2:20])
            args=[[keys, keys[0:-1]+['mean_coherence_max']],
                  [v,v_max]]        
            
        if keys[-1]=='firing_rate':
            v=numpy.mean(val.y)
            args=[[keys],
                  [v]]
        
        for k, v in zip(*args):
            out=misc.dict_recursive_add(out,  k, v)
    
    attrs=[]
    for keys, val in misc.dict_iter(out):
        if not keys[-1] in attrs:
            attrs.append(keys[-1])
        
    return out, attrs     
      
def compute_mse(v1,v2):
    v_mse=(v1-v2)/v1
    
    if isinstance(v_mse, numpy.ndarray):
        v_mse[numpy.isnan(v_mse)]=0
    elif numpy.isnan(v_mse):
        v_mse=0
    v_mse=numpy.mean((v_mse)**2)
    
    return v_mse

def compute_performance(d, nets, models, attr):       
    results={}
    
    for run in d.keys():
        if run=='no_pert':
            continue
        for model in models:
            for attr in attrs:
                
                keys_c= ['no_pert', 'Net_0', model, attr]
                keys_l= ['no_pert', 'Net_1', model, attr]
                
                if not misc.dict_haskey(d,keys_c ):
                    continue
                
                v_control0=misc.dict_recursive_get(d,keys_c)
                v_lesion0=misc.dict_recursive_get(d,keys_l)
                if type(v_lesion0)==str:
                    print v_lesion0
                if type(v_control0)==str:
                    print v_control0
                v_mse0=compute_mse(v_control0, v_lesion0)
                
                s,t,_,x=run.split('_')
                x=float(x)
                conn=s+'_'+t
                
                keys1=[run, 'Net_0', model, attr]
                keys2=[run, 'Net_1', model, attr]
                keys3=['control',conn, model, attr]
                keys4=['lesion',conn, model, attr]
               
                if attr=='firing_rate':
                    s='fr'
                if attr=='mean_coherence':
                    s='mc'
                if attr=='mean_coherence_max':
                    s='mcm'
                if attr=='phases_diff_with_cohere':
                    s='pdwc'
                    
                keys5=['control',conn, model, 'mse_rel_control_'+s]
                keys6=['lesion',conn, model, 'mse_rel_control_'+s]
                        
                
                v1=misc.dict_recursive_get(d, keys1)
                v2=misc.dict_recursive_get(d, keys2)
                
                v_mse1=compute_mse(v_control0,v1)/v_mse0
                v_mse2=compute_mse(v_control0,v2)/v_mse0
        
                for keys, val in zip([keys3,keys4,keys5,keys6],
                                     [v1, v2, v_mse1, v_mse2]): 
                    if not misc.dict_haskey(results, keys):
                        results=misc.dict_recursive_add(results,  keys, 
                                                    [(x, val)])                
                    else:
                        l=misc.dict_recursive_get(results, keys)
                        l.append((x, val))

                
                #Add midpoint
                for keys, val in zip([keys3,keys4,keys5,keys6],
                                    [v_control0, v_lesion0, 
                                     0, v_mse0/v_mse0]):         
                    l=misc.dict_recursive_get(results, keys)
                    if not (1, val) in l: 
                        l.append((1, val)) 
                    
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
        g=(-y[0]+a1*y[1]-a2*y[2]+a2*y[4]-a1*y[5]+y[6])/(60*h)
        d=misc.dict_recursive_add(gradients, keys, g)         
    return results, gradients

def nice_labels():
    
    d={'CTX_M1':r'CTX$\to$$MSN_{D1}$',
       'CTX_M2':r'CTX$\to$$MSN_{D2}$',
       'CTX_ST':r'CTX$\to$STN',
       'FS':r'FSN',
       'MS_MS': r'MSN$\to$MSN',
       'M1_M1':r'$MSN_{D1}$$\to$$MSN_{D1}$',
       'M1_M2':r'$MSN_{D1}$$\to$$MSN_{D2}$',   
       'M2_M1':r'$MSN_{D2}$$\to$$MSN_{D2}$',
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
    return d

def nice_labels2():
    d={'GP_GP':'GP vs GP',
       'GI_GI':'TI vs TI', 
       'GI_GA':'TI vs TA', 
       'GA_GA':'TA vs TA'}
    return d
        
def gs_builder_conn(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.05 ), 
              hspace=kwargs.get('hspace', 0.05 ))

    iterator = [[slice(1,2),slice(0,4)],
                [slice(2,5),slice(0,4)],
                [slice(5,7),slice(0,4)],
                [slice(1,2),slice(4,8)],
                [slice(2,5),slice(4,8)],
                [slice(5,7),slice(4,8)]]
    
    return iterator, gs, 



# def gs_builder_coher(*args, **kwargs):
#     import matplotlib.gridspec as gridspec
#     n_rows=kwargs.get('n_rows',2)
#     n_cols=kwargs.get('n_cols',3)
#     order=kwargs.get('order', 'col')
#     
#     gs = gridspec.GridSpec(n_rows, n_cols)
#     gs.update(wspace=kwargs.get('wspace', 0.05 ), 
#               hspace=kwargs.get('hspace', 1. / n_cols ))
# 
#     iterator = [[slice(0,6), slice(1,9)],
# #                 [slice(1,7), slice(7,9)]
#                 ]
#     
    return iterator, gs, 

def gs_builder_coher(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.05 ), 
              hspace=kwargs.get('hspace', 1. / n_cols ))

    iterator = [[slice(1,8), slice(2,7)],
                [slice(1,8), slice(7,9)]]
    
    return iterator, gs, 

def generate_plot_data_raw(d, models, attrs, exclude=[], flag='raw', attr='firing_rate'):

    labelsx_meta=[]
    
    if flag=='raw':
        data_keys=['x', 'y']
    if flag=='gradient':
        data_keys=['z']
    
    for k in sorted(d.keys()):
        if k in exclude:
            continue        
        labelsx_meta.append(k)
    
#     res={}
#     for keys, val in misc.dict_iter(d):
#         if attr==keys[2]:
#             res=misc.dict_recursive_add(res, keys[0:2], val)
    
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
        print numpy.array(val).shape
        val=numpy.array(val).ravel()
        print val.shape
        out=misc.dict_recursive_set(out, keys, val)

    
    dd={'firing_rate':{'labelsy':['M1', 'M2', 'FS', 'GA', 'GI', 'GP','SN', 'ST'], 
                       'labelsx_meta':labelsx_meta},
        'mse_rel_control_fr':{'labelsy':['M1', 'M2', 'FS', 'GA', 'GI', 'GP','SN', 'ST'], 
                              'labelsx_meta':labelsx_meta},
        'mean_coherence':{'labelsy':['GA_GA', 'GI_GA', 'GI_GI', 'GP_GP'], 
                          'labelsx_meta':labelsx_meta},
        'mean_coherence_max':{'labelsy':['GA_GA', 'GI_GA', 'GI_GI', 'GP_GP'], 
                          'labelsx_meta':labelsx_meta},
        'mse_rel_control_mc':{'labelsy':['GA_GA', 'GI_GA', 'GI_GI', 'GP_GP'], 
                              'labelsx_meta':labelsx_meta},
        'mse_rel_control_mcm':{'labelsy':['GA_GA', 'GI_GA', 'GI_GI', 'GP_GP'], 
                              'labelsx_meta':labelsx_meta},
        'mse_rel_control_pdwc':{'labelsy':['GA_GA', 'GI_GA', 'GI_GI', 'GP_GP'], 
                              'labelsx_meta':labelsx_meta}}
    
    for attr, d in dd.items():
        print attr
        if flag=='raw':
#             print d
            d['labelsx']=[0.25,0.5,0.75,1,1.25,1.5,1.75]*len(labelsx_meta)
        if flag=='gradient':
            d['labelsx']=labelsx_meta
            
        for dk in data_keys:
            d[dk]=[]
            
            for key in d['labelsy']:
                v=out[attr][key]
                d[dk].append(v[dk])
            d[dk]=numpy.array(d[dk])
            
    return dd

def separate_M1_M2(*args, **kwargs):
    l=[]
    for d in args:
        d0=d.copy()
        d1=d.copy()

        for k in [kwargs.get('z_key'), 'labelsy']:        
            d0[k]=d0[k][0:2]
            d1[k]=d1[k][2:]
        l.extend([d0,d1])
    return l

def plot_coher(d, labelsy, flag='dop', labelsx=[]):
    fig, axs=ps.get_figure2(n_rows=8, n_cols=9,  w=700, h=900, fontsize=24,
                            frame_hight_y=0.5, frame_hight_x=0.7, 
                            title_fontsize=24,
                            gs_builder=gs_builder_coher) 
#     fig, axs=ps.get_figure(n_rows=4, n_cols=1, w=500.0*0.65*2, h=400.0*0.65*2, fontsize=16,
#                            frame_hight_y=0.6, frame_hight_x=0.8, 
#                            title_fontsize=20, text_usetex=False)        
    
# 
#     for i in range(len(labelsy)):
#         if labelsy[i] in nice_labels().keys():
#             labelsy[i]=nice_labels()[labelsy[i]]
#             
#     
#     for i in range(len(labelsx)):
#         if labelsx[i] in nice_labels2().keys():
#             labelsx[i]=nice_labels2()[labelsx[i]]
#             
     
    startx=0
    starty=0
    z_key='y'
    
    images=[]
    kwargs={'ax':axs[0],
            'd':d,
            'flip_axes':True,
            'fontsize_x':24,
#             'vertical_lines':True, 
            'horizontal_lines':True, 
            'images':images,
            'z_key':z_key,
            'startx':startx,
            'starty':starty}

    _plot_conn(**kwargs)
    
    kwargs['ax']=axs[1]
    _plot_bar(**kwargs)
    
    
    
    box = axs[0].get_position()
    pos=[box.x0+0.1*box.width, 
         box.y0+box.height+box.height*0.15, 
         box.width*0.8, 
         0.05]
    
    
    for l in axs[1].patches:
        pylab.setp(l,**{'edgecolor':'0.5'})
    
    axColor=pylab.axes(pos)
    #     axColor = pylab.axes([0.05, 0.9, 1.0, 0.05])
    cbar=pylab.colorbar(images[0], cax = axColor, orientation="horizontal")
    cbar.ax.set_title('MSE control vs lesion rel. base model')#, rotation=270)
    from matplotlib import ticker
    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()
    axs[0].text(0.05, 1.02, "Coherence", transform=axs[0].transAxes)
    axs[0].text(0.55, 1.02, "Phase shift", transform=axs[0].transAxes)
    axs[1].text(1.45, 0.65, "Mean effect", transform=axs[0].transAxes,
                    rotation=270)
    axs[0].text(-0.6, 0.9, "Perturbed connection", transform=axs[0].transAxes,
                rotation=90)                        
    axs[1].my_remove_axis(xaxis=False, yaxis=True)
    axs[1].my_set_no_ticks(xticks=2)
#     set_colormap(axs[0], images[0], 'MSE coherence change')
    
    
#     stopx_meta=len(labelsx_meta)
#     
#     posy=numpy.linspace(.5,stopy-.5, stopy)
#     ratio=stopx/stopx_meta
#     posx=numpy.linspace(.5*ratio,stopx-.5*ratio, stopx_meta)
#     x=numpy.linspace(0, stopx, stopx_meta+1)
#     x1,y1=numpy.meshgrid(numpy.linspace(startx, stopx, stopx+1), 
#                          numpy.linspace(starty, stopy, stopy+1)) 
#     
#     
#     _vmin=0
#     _vmax=4
#     stepx=1
#     stepy=1
#     startx=0
#     starty=0
#     stopy=14
#     stopx=8
#     maxy=len(labels)
#     maxx=8
#     
#     posy=numpy.linspace(0.5,maxy-0.5, maxy)
#     posx=numpy.linspace(0.5,maxx-0.5, maxx)
#     axs[1].barh(posy,numpy.mean(z,axis=1)[::-1], align='center', color='0.5')
#     axs[1].plot([1,1], [0,stopy], 'k', linewidth=3, linestyle='--')
#     x1,y1=numpy.meshgrid(numpy.linspace(startx, stopx, maxx+1),
#                    numpy.linspace(stopy, starty, maxy+1))
#     
#     im = axs[0].pcolor(x1, y1, z, cmap='jet', 
#                         vmin=_vmin, vmax=_vmax
#                        )
#     axs[0].set_yticks(posy)
#     axs[0].set_yticklabels(labels[::-1])
#     axs[0].set_xticks(posx)
#     axs[0].set_xticklabels(labels2*2, rotation=70, ha='right')
#     axs[0].set_ylim([0,maxy])
#     axs[0].text(0.05, 1.02, "Coherence", transform=axs[0].transAxes)
#     axs[0].text(0.55, 1.02, "Phase shift", transform=axs[0].transAxes)
#     axs[1].text(1.45, 0.65, "Mean effect", transform=axs[0].transAxes,
#                                 rotation=270)
#     axs[0].text(-0.6, 0.9, "Connection without dop. effect", transform=axs[0].transAxes,
#                 rotation=90)
#         
#     axs[1].my_remove_axis(xaxis=False, yaxis=True)
#     axs[1].my_set_no_ticks(xticks=2)
#     axs[1].set_xlim([0,maxy])
#     axs[1].set_xlim([0,2])
# #     axs[1].set_xticks([0.04, 0.12])
# 
# 
#     box = axs[0].get_position()
#     axColor=pylab.axes([box.x0+0.1*box.width, 
#                         box.y0+box.height+box.height*0.15, 
#                         box.width*0.8, 
#                         0.05])
#     #     axColor = pylab.axes([0.05, 0.9, 1.0, 0.05])
#     cbar=pylab.colorbar(im, cax = axColor, orientation="horizontal")
#     cbar.ax.set_title('MSE control vs lesion rel. base model')#, rotation=270)
#     from matplotlib import ticker
#     tick_locator = ticker.MaxNLocator(nbins=4)
#     cbar.locator = tick_locator
#     cbar.update_ticks()
    
    return fig
#     pylab.show()
    
    
def _plot_bar(**kwargs):
    
    ax=kwargs.get('ax')
    d=kwargs.get('d')
#     images=kwargs.get('images')
    z_key=kwargs.get('z_key')
#     startx=kwargs.get('startx')
#     starty=kwargs.get('starty')
    flip_axes=kwargs.get('flip_axes')
#     vertical_lines=kwargs.get('vertical_lines')
#     horizontal_lines=kwargs.get('horizontal_lines')
#     fontsize_x=kwargs.get('fontsize_x',24)
    if flip_axes:
        stopx=len(d['labelsy'])
        stopy=len(d['labelsx']) 
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
        labelsy_meta=d['labelsy']
        labelsx_meta=d['labelsx_meta']
#         labelsx=d['labelsy']
#         labelsy=d['labelsx']
        
#         nice_labels2=nice_labels2()
#         nice_labels=nice_labels()
#         
#     for i in range(len(labelsx_meta)):         
#         if labelsx_meta[i] in nice_labels().keys():
#             labelsx_meta[i]=nicex_labels()[labels_meta[i]]
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

    ratio=1
    posy=numpy.linspace(.5*ratio,stopy-.5*ratio, stopy)    
    
#     ratio=stopx/stopx_meta
#     posx=numpy.linspace(.5*ratio,stopx-.5*ratio, stopx_meta)
    
    ax.barh(posy,numpy.mean( d[z_key],axis=0)[::-1], align='center', color='0.5',
#             linewidth=0.1
            )
    ax.plot([1,1], [0,stopy], 'k', linewidth=3, linestyle='--')    
    ax.set_ylim([0,stopy])  
     
def _plot_conn(**kwargs):
    
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
    if flip_axes:
        stopx=len(d['labelsy'])
        stopy=len(d['labelsx']) 
        labelsx_meta=d['labelsy']
        labelsy_meta=d['labelsx_meta']
        labelsx=d['labelsx']
        labelsy=d['labelsy']
#         nice_labels=nice_labels2()
#         nice_labels2=nice_labels()
        
        d[z_key]=numpy.transpose(d[z_key])
    else:
        stopy=len(d['labelsy'])
        stopx=len(d['labelsx']) 
        labelsy_meta=d['labelsy']
        labelsx_meta=d['labelsx_meta']
        labelsx=d['labelsy']
        labelsy=d['labelsx']
        
#         nice_labels2=nice_labels2()
#         nice_labels=nice_labels()
#         
#     for i in range(len(labelsx_meta)):         
#         if labelsx_meta[i] in nice_labels().keys():
#             labelsx_meta[i]=nicex_labels()[labels_meta[i]]
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
    
        
    im = ax.pcolor(x1, y1, d[z_key][::-1,], cmap='jet')       

    if vertical_lines:
        x=numpy.linspace(0, stopx, stopx_meta+1)
        for xx in x:
            ax.plot([xx,xx],[0,stopy], 'k', linewidth=1.)
    if horizontal_lines:
        x=numpy.linspace(0, stopy, stopy_meta+1)
        for xx in x:
            ax.plot([0,stopx],[xx,xx],'k', linewidth=1.)
    
    images.append(im)                
    ax.set_yticks(posy)
    ax.set_yticklabels(labelsy_meta[::-1])
    ax.set_xticks(posx)

    ax.set_xticklabels([s.rjust(10) for s in labelsx_meta], rotation=70, 
                           ha='right', 
                           fontsize=fontsize_x)

    ax.set_ylim([0,stopy])
    ax.set_xlim([0,stopx])
    

def set_colormap(ax, im, label):
        box = ax.get_position()
        axColor = pylab.axes([box.x0 + box.width * 1.03, 
                              box.y0+box.height*0.1, 
                              0.01, 
                              box.height*0.8])
        cbar=pylab.colorbar(im, cax = axColor, orientation="vertical")
        cbar.ax.set_ylabel(label, rotation=270)
        from matplotlib import ticker
        
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()

def plot_conn(d0, d1, d2, d3, **kwargs):
    fig, axs=ps.get_figure2(n_rows=8, n_cols=9, w=1000, h=600, fontsize=24,
                            frame_hight_y=0.5, frame_hight_x=0.7, 
                            title_fontsize=24,
                            gs_builder=gs_builder_conn) 
    
    
    flag=kwargs.get('flag', 'raw')
    coher_label=kwargs.get('coher_label', 'Coherence')
    fr_label=kwargs.get('fr_label',"Firing rate (Hz)")
    title=kwargs.get('title', "Slow wave") 
    z_key=kwargs.get('z_key',"y")

    d00, d01, d20, d21=separate_M1_M2( d0, d2, **{'z_key':z_key})
    args=[ d00, d01, d1, d20, d21, d3]

    startx=0
    starty=0
    images=[]
    for ax, d in zip(axs,args):

        kwargs={'ax':ax,
                'd':d,
                'images':images,
                'z_key':z_key,
                'startx':startx,
                'starty':starty}

        _plot_conn(**kwargs)
        

        
    args=[[axs[3], axs[4],axs[5]], 
          ['', '', coher_label],
          [images[3],images[4],images[5]]]

    for ax, label, im in zip(*args):
        set_colormap(ax, im, label)
    
#     if flag=='raw':
    axs[4].text(1.18, 1.15, fr_label, transform=axs[4].transAxes,
                rotation=270)

    axs[0].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)   
    axs[1].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)   
    axs[3].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True) 
    axs[4].my_remove_axis(xaxis=True, yaxis=True,keep_ticks=True) 
    axs[5].my_remove_axis(xaxis=False, yaxis=True,keep_ticks=True) 
    axs[0].text(0.35, 1.05, "Control", transform=axs[0].transAxes)     
    axs[3].text(0.35, 1.05, "Lesion", transform=axs[3].transAxes)  
           
    font0 = FontProperties()
    font0.set_weight('bold')
    axs[0].text(0.8, 1.4, title, transform=axs[0].transAxes, 
                fontproperties=font0         )
    return fig

def add(d0,d1):
    d0['labelsy']+=d1['labelsy']
#     d0['labelsx_meta']+=d1['labelsx_meta']
    d0['x']=numpy.concatenate((d0['x'], d1['x']), axis=0)
    d0['y']=numpy.concatenate((d0['y'], d1['y']), axis=0)
    return d0

if __name__=='__main__':
    models=['M1', 'M2', 'FS', 'GA', 'GI', 'GP', 'ST','SN',
            'GP_GP', 'GA_GA', 'GI_GA', 'GI_GI']
    nets=['Net_0', 'Net_1']
    attrs=[
           'firing_rate', 
           'mean_coherence', 
           'phases_diff_with_cohere'
           ]
    path=('/home/mikael/results/papers/inhibition/network/'
          +'supermicro/simulate_slow_wave_ZZZ_conn_effect_perturb/')
 
    home = expanduser("~")
    script_name=__file__.split('/')[-1][0:-3]
    file_name = get_file_name(script_name)
    from_disk=1
    
    attr_add=['mse_rel_control_fr', 'mse_rel_control_mc',
              'mse_rel_control_pdwc', 'mse_rel_control_mcm']
    
    sd = get_storage(file_name, '')
    d={}
    if not from_disk:
    
        d['raw']=gather(path, nets, models, attrs)
        d['data'], attrs=extract_data(d['raw'], nets, models, attrs)
        out=compute_performance(d['data'], nets, models, attrs)
        d['change_raw'], d['gradients']=out

        v=generate_plot_data_raw(d['change_raw']['control'],
                                  models, attrs+attr_add, flag='raw')
        d['d_raw_control']=v
        v=generate_plot_data_raw(d['change_raw']['lesion'],
                                  models, attrs+attr_add, flag='raw')
        d['d_raw_lesion']=v
        v=generate_plot_data_raw(d['gradients']['control'],
                                  models, attrs+attr_add, flag='gradient')
        
        d['d_gradients_control']=v 
        v=generate_plot_data_raw(d['gradients']['lesion'],
                                  models, attrs+attr_add, flag='gradient')
        d['d_gradients_lesion']=v
        
        save(sd, d)
    else:
#         filt=['performance']+models
        d = sd.load_dic()
#         d2=sd.load_dic()
#         d = sd.load_dic()
#     pp(d['d0'])
#     pp(d['d1'])
#     plot_raw(d, d_keys, attr='mean_coherence')
#     plot_raw(d, d_keys, attr='phases_diff_with_cohere')
#     plot3(d['d0'], d['d1'], d['d2'], d['d3'])
#     figs=[]
#     fig=plot_conn(d['d_raw_control']['firing_rate'], 
#                   d['d_raw_control']['mean_coherence_max'], 
#                   d['d_raw_lesion']['firing_rate'], 
#                   d['d_raw_lesion']['mean_coherence_max'])
#     figs.append(fig)
# 
#     fig=plot_conn(d['d_raw_control']['mse_rel_control_fr'], 
#                   d['d_raw_control']['mse_rel_control_mc'], 
#                   d['d_raw_lesion']['mse_rel_control_fr'], 
#                   d['d_raw_lesion']['mse_rel_control_mc'])
# 
#     figs.append(fig)
#     kwargs={'flag':'gradient',
#             'coher_label':'Coher./weight',
#             'fr_label':"Hz/nS",
#             'z_key':"z"}
#     
#     fig=plot_conn(d['d_gradients_control']['firing_rate'], 
#               d['d_gradients_control']['mean_coherence_max'], 
#               d['d_gradients_lesion']['firing_rate'], 
#               d['d_gradients_lesion']['mean_coherence_max'],**kwargs)
#     figs.append(fig)    

#     from effect_dopamine import plot_coher
    
    d0=d['d_raw_lesion']['mse_rel_control_mc']
    d1=d['d_raw_lesion']['mse_rel_control_pdwc']
    
    d=add(d0,d1)

    plot_coher(d, d['labelsx'])

#     plot_conn(d['d0'], d['d1'], d['d2'], d['d3'], **kwargs)
#     plot(d['bar_obj'], d['labels'])
    pylab.show()
#     pp(d)
    
    
    
    
    
    v