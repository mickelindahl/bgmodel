'''
Created on 25 mar 2014

@author: mikael
'''

import numpy
import pylab
import toolbox.plot_settings as ps


from os.path import expanduser
from simulate import (main_loop, show_fr, get_file_name, 
                      get_file_name_figs)

from toolbox.data_to_disk import Storage_dic
from toolbox import misc
from toolbox.network import manager
from toolbox.network.manager import (add_perturbations, get_storage)
from toolbox.network.manager import Builder_Go_NoGo_with_lesion as Builder
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
    return {'duration':duration,
            'print_time':False,
            'proportion_connected':prop_conn,
            'save_conn':{'overwrite':True},
            'resolution':res, 
            'repetition':rep,
            'sim_time':laptime*res*res*rep, 
            'sim_stop':laptime*res*res*rep, 
            'size':4000.0, 
            'start_rec':0.0,  
            'stop_rec':numpy.inf,
            'sub_sampling':sub,
            'threads':THREADS,}   
    
def get_kwargs_engine():
    return {'verbose':True}

def get_networks(builder, **k_in):
    info, nets, builder=manager.get_networks(builder,
                                             get_kwargs_builder(**k_in),
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
        max_set=0
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
                if max_set<=i:
                    max_set+=1
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


def show_heat_map(d,attr,**k):
    models=['SN']
    res=k.get('resolution')
    titles=k.get('titles')
    n=len(models)
    m=len(d.keys())
#     attr='mean_rate_slices'
    fig, axs=ps.get_figure(n_rows=3, n_cols=2, w=600.0*0.65*2, h=700.0*0.65*2, fontsize=24,
                           frame_hight_y=0.5, frame_hight_x=0.6, title_fontsize=20)        
     
    i=0
    performance={}
    
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
            x0,y0,z0=get_optimal(z.shape[0])
            performance[key]=numpy.round(numpy.sum(numpy.abs(z)) ,2)
             
             
             
            im=axs[i].pcolor(x, y, z, cmap='coolwarm',  vmin=-40, vmax=40)
#           
            box = axs[i].get_position()
            
            if key in ['Net_0', 'Net_2', 'Net_4']:
                v=1.0
            else:
                v=0.9
            axs[i].set_position([box.x0*v, box.y0, box.width*0.85, box.height])
            
            axs[i].set_xlabel('Action 1')
            axs[i].set_ylabel('Action 2')
            # create color bar
            
#             pylab.colorbar(im, cax = axColor, orientation="vertical")

  
#             axs[i].set_zlim([-40, 40])
            if key=='Net_1':
#                 from mpl_toolkits.axes_grid1 import make_axes_locatable
#                 divider = make_axes_locatable(pylab.gca())
#                 cax = divider.append_axes("right", "5%", pad="3%")
                axColor = pylab.axes([box.x0*v + box.width * 0.9, box.y0, 0.05, box.height])
                cbar=pylab.colorbar(im, cax = axColor, orientation="vertical")
                cbar.ax.set_ylabel('Contrast (spike/s)', rotation=270)
                from matplotlib import ticker
                tick_locator = ticker.MaxNLocator(nbins=5)
                cbar.locator = tick_locator
                cbar.update_ticks()
#                 pylab.colorbar(im)
            
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
#     pylab.tight_layout()       
#     for ax in axs:
#         ax.view_init(elev=15)
    
    ax=axs[5]
    import pprint
    ax.text( 0.1, 0.1, pprint.pformat(performance), 
                   transform=ax.transAxes, 
            fontsize=12)
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
    fig, axs=ps.get_figure(n_rows=1, n_cols=1, w=400.0, h=250.0, fontsize=16)        
 
    x,y,z=get_optimal(size)
       
    im=axs[0].pcolor(x, y, z, cmap='coolwarm',  vmin=-40, vmax=40)
#           
    box = axs[0].get_position()
    axs[0].set_position([box.x0*1.05, box.y0, box.width*0.85, box.height])
 
    axs[0].set_xlabel('Action 1')
    axs[0].set_ylabel('Action 2')
    axs[0].my_set_no_ticks(xticks=5,yticks=5, )
            
    return fig
    

class Setup(object):

    def __init__(self, **k):
        self.threads=k.get('threads',1)
        self.res=k.get('resolution',2)
        self.rep=k.get('repetition',2)
        self.laptime=k.get('laptime',1500.0)
        self.duration=k.get('duration',[1000.,500.])
        self.l_mean_rate_slices=k.get('l_mean_rate_slices', ['mean_rate_slices', 
                                                             'mean_rate_slices_0',
                                                             'mean_rate_slices_1'])
    def builder(self):
        d= {'repetition':self.rep,
            'resolution':self.res,
            'input_lists': [['C1'],
                            ['C1', 'C2']],
            'sub_sampling':6.25,
            'laptime':self.laptime,                
            'duration':self.duration}
        return d

    def firing_rate(self):
        d={'average':False, 
           'sets':[0,1],
           'time_bin':5,
           'threads':self.threads}
        return d
    
    def mean_rate_slices(self):
        d={'threads':self.threads}
        return d

    def plot_fr(self):
        d={'win':10.,
           'by_sets':True,
           't_start':0.0,
           't_stop':10000.0}
        return d
    def plot_fr2(self):
        d={'win':10.,
           'by_sets':True,
           't_start':0.0,
           't_stop':30000.0,
           'labels':['Action 1', 'action 2'],
'fig_and_axes':{'n_rows':3, 
                                        'n_cols':1, 
                                        'w':800.0*0.55*2, 
                                        'h':600.0*0.55*2, 
                                        'fontsize':11*2,
                                        'frame_hight_y':0.8,
                                        'frame_hight_x':0.78,
                                        'linewidth':3.
                                        }
    }
        return d    
    
        
 
    def plot_3d(self):
        d={'resolution':self.res,
           'titles':['Only D1', 
                     'D1,D2',
                     'MSN lesioned (D1, D2)',
                     'FSN lesioned (D1, D2)',
                     'GPe TA lesioned (D1,D2)']}
        return d

        
    def plot_bulkt(self):
        d={'labels':[
                    'Only D1', 
                     'D1 and D2',
                    'MSN lesioned (D1 and D2)',
                     'FSN lesioned (D1 and D2)',
                     'GPe TA lesioned (D1 and D2)']}
        return d
    def get_l_mean_rate_slices(self):
        return self.l_mean_rate_slices

def simulate(builder, from_disk, perturbation_list, script_name, setup):
    home = expanduser("~")
    
    file_name = get_file_name(script_name, home)
    file_name_figs = get_file_name_figs(script_name, home)
    
    d_firing_rate = setup.firing_rate()
    d_mrs = setup.mean_rate_slices()
    
    attr = ['firing_rate']
    models = ['M1', 'M2', 'FS', 'ST', 'GA', 'GI', 'SN']
    sets = ['set_0', 'set_1']
    
    info, nets, intervals, rep, x_set, y_set = get_networks(builder, 
                                                            **setup.builder())

    x=range(len(x_set))
    xticklabels=[str(a)+'*'+str(b) for a, b in zip(x_set, y_set) ]
    d_mrs.update({'intervals':intervals[1], 
                  'repetition':rep, 
                  'set_0':{'x':x, 'xticklabels':xticklabels, 'x_set':x_set}, 
                  'set_1':{'x':x, 'xticklabels':xticklabels, 'x_set':y_set}, 
                  'sets':[0, 1]})
    
    d_firing_rate['mean_rate_slices']=d_mrs
    
    d=d_mrs.copy()
    d['intervals']=[[v[0], v[0]+100] for v in intervals[1]]
    d_firing_rate['mean_rate_slices_0']=d
 
    d=d_mrs.copy()
    d['intervals']=[[v[1]-100, v[1]] for v in intervals[1]]
 
    d_firing_rate['mean_rate_slices_1']=d
    
    kwargs_dic = {'firing_rate':d_firing_rate, 
#                   'mean_rate_slices2': d_mrs
                  }


    add_perturbations(perturbation_list, nets)    
    sd = get_storage(file_name, info)
    
#     net=nets['Net_0']
#     for key in sorted(net.par['nest']): 
#         if 'weight' in net.par['nest'][key]: 
#             print net.par['nest'][key]['weight']
#     
#     for key in sorted(net.par['nest']): 
#         if 'weight' in net.par['nest'][key]: 
#             print key

# l=[]
# for key in sorted(net.par['conn']):
#      
#     if 'weight' in net.par['conn'][key] and key not in l: 
# #         print net.par['conn'][key]['fan_in']
#         print key
#         l.append(key)
# for key in sorted(net.par['conn']): 
#     if 'weight' in net.par['conn'][key]: 
#         print key
#      
    from_disks, d = main_loop(from_disk, attr, models, 
                              sets, nets, kwargs_dic, sd)
    
    return file_name, file_name_figs, from_disks, d, models, sd


def create_figs(setup, file_name_figs, d, models,sd):
    
    sd_figs = Storage_dic.load(file_name_figs)
    figs = []
    d_plot_3d=setup.plot_3d()

    
    d_plot_fr = setup.plot_fr()
    d_plot_fr2 = setup.plot_fr2()
    
    d_plot_bulk=setup.plot_bulkt()
    l_mean_rate_slices=setup.get_l_mean_rate_slices()
    
    
    figs.append(plot_optimal(setup.res))
#     for name in l_mean_rate_slices:
#         figs.append(show_3d(d,name,  **d_plot_3d))
# 
    for name in l_mean_rate_slices:
        fig, perf=show_heat_map(d, name,  **d_plot_3d)
        figs.append(fig)
 

# 
#     for name in l_mean_rate_slices:   
#         figs.append(show_bulk(d, models, name, **d_plot_bulk))
#      
#     figs.append(show_neuron_numbers(d, models))
#      
#     for i in range(len(d.keys())):
#         figs.append(show_fr(d['Net_'+str(i)], models, **d_plot_fr))
     
    for i in range(len(d.keys())):
        
        if i!=1:
            continue
        
        figs.append(show_fr(d['Net_'+str(i)], ['M1', 'M2', 'SN'], **d_plot_fr2))
        axs=figs[-1].get_axes()
        ps.shift('upp', axs, 0.1, n_rows=len(axs), n_cols=1)
        ps.shift('left', axs, 0.25, n_rows=len(axs), n_cols=1)
        for i, ax in enumerate(axs):      
            
            ax.my_set_no_ticks(xticks=5, yticks=4)
            ax.legend(bbox_to_anchor=[1.45,1.])
        axs=figs[-1].get_axes()
        for i, ax in enumerate(axs):
            ax.my_set_no_ticks(xticks=5)
#             ax.set_ylabel('Ra')
            if i==1:
                pass
            else:
                ax.set_ylabel('')
            if i >1:
                pass
            else:
#                 ax.set_ylabel('')
                ax.my_remove_axis(xaxis=True)
            if i==2:
                pass
            else:
                ax.set_xlabel('')
            

    sd_figs.save_figs(figs, format='png')
#     sd_figs.save_figs(figs, format='svg')
def main(builder=Builder,
         from_disk=2,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3],
         setup=Setup(**{'threads':THREADS,
                        'resolution':5,
                        'repetition':5})):
    
    
    v=simulate(builder, from_disk, perturbation_list, script_name, setup)
    file_name, file_name_figs, from_disks, d, models, sd = v
    
    if numpy.all(numpy.array(from_disks) > 0):
        create_figs(setup, file_name_figs, d, models, sd)
    
    
#     pylab.show()
 
import unittest
class TestMethods(unittest.TestCase):     
    def setUp(self):
        from toolbox.network.default_params import Perturbation_list as pl
        from_disk=2
        
        import oscillation_perturbations4 as op
        
        rep, res=1, 5
        
        sim_time=rep*res*res*1500.0
      
        threads=2
        
        l=op.get()
        
        p=pl({'simu':{'sim_time':sim_time,
                      'sim_stop':sim_time,
                      'threads':threads}},
                  '=')
#         p+=l[4+3] #data2
#         p+=l[4] #data        
        p+=l[4+5] #data3
        
        self.setup=Setup(**{'threads':threads,
                        'resolution':res,
                        'repetition':rep})
        
        v=simulate(builder=Builder,
                            from_disk=from_disk,
                            perturbation_list=p,
                            script_name=(__file__.split('/')[-1][0:-3]
                                         +'/data3'),
                            setup=self.setup)
        
        file_name, file_name_figs, from_disks, d, models, sd= v
        
        self.res=res
        self.file_name_figs=file_name_figs
        self.from_disks=from_disks
        self.d=d
        self.models=models
        self.sd=sd
        

    def test_create_figs(self):
        create_figs(
                    self.setup,
                    self.file_name_figs, 
                    self.d, 
                    self.models,
                    self.sd)
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
    
    
    
    



    