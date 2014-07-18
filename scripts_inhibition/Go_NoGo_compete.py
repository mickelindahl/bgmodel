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
    sub=k_in.get('sub_sampling',50)
    
    return {'print_time':False,
            'save_conn':{'overwrite':True},
            'resolution':res, 
            'repetition':rep,
            'sim_time':1000.*res*res*rep, 
            'sim_stop':1000.*res*res*rep, 
            'size':750.0, 
            'start_rec':0.0,  
            'stop_rec':1000.*(res*res*rep+1),
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
def show_bulk(d, models, **k):
    
    attr='mean_rate_slices'    
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
                                'label':labels[j]})
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
                
     

def show_3d(d,**k):
    models=['SN']
    res=k.get('resolution')
    titles=k.get('titles')
    n=len(models)
    m=len(d.keys())
    attr='mean_rate_slices'
    fig, axs=ps.get_figure(n_rows=m, n_cols=1, w=500.0, h=800.0, fontsize=10, 
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
                                antialiased=False)
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


class Setup(object):

    def __init__(self, **k):
        self.threads=k.get('threads',1)
        self.res=k.get('resolution',2)
        self.rep=k.get('repetition',2)
                
    def builder(self):
        d= {'repetition':self.rep,
            'resolution':self.res,
            'input_lists': [['C1'],
                            ['C1', 'C2']],
            'sub_sampling':50}
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
           't_stop':30000.0}
        return d
    
    def plot_3d(self):
        d={'resolution':self.res,
           'titles':['Only D1', 
                     'D1 and D2',
                     'MSN lesioned (D1 and D2)',
                     'FSN lesioned (D1 and D2)',
                     'GPe TA lesioned (D1 and D2)']}
        return d

        
    def plot_bulkt(self):
        d={'labels':['Only D1', 
                     'D1 and D2',
                     'MSN lesioned (D1 and D2)',
                     'FSN lesioned (D1 and D2)',
                     'GPe TA lesioned (D1 and D2)']}
        return d

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
    
    return file_name, file_name_figs, from_disks, d, models


def create_figs(setup, file_name_figs, d, models):
    
    sd_figs = Storage_dic.load(file_name_figs)
    figs = []
    
    d_plot_fr = setup.plot_fr()
    d_plot_3d=setup.plot_3d()
    d_plot_bulk=setup.plot_bulkt()
    figs.append(show_neuron_numbers(d, models))
   
    figs.append(show_bulk(d, models, **d_plot_bulk))
    figs.append(show_3d(d, **d_plot_3d))
#     for i in range(5):
#         figs.append(show_fr(d['Net_'+str(i)], models, **d_plot_fr))
#     
#     for i in range(5):
#         figs.append(show_fr(d['Net_'+str(i)], ['M1', 'M2', 'SN'], **d_plot_fr))
#         

    sd_figs.save_figs(figs, format='png')

def main(builder=Builder,
         from_disk=2,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3],
         setup=Setup(**{'threads':THREADS,
                        'resolution':5,
                        'repetition':5})):
    
    
    v=simulate(builder, from_disk, perturbation_list, script_name, setup)
    file_name, file_name_figs, from_disks, d, models = v
    
    if numpy.all(numpy.array(from_disks) > 0):
        create_figs(setup, file_name_figs, d, models)
    
    
#     pylab.show()
 
import unittest
class TestMethods(unittest.TestCase):     
    def setUp(self):
        from toolbox.network.default_params import Perturbation_list as pl
        from_disk=0
        
        import oscillation_perturbations as op
        
        rep, res=1, 5
        
        sim_time=rep*res*res*1000.0
      
        threads=12
        
        l=op.get()
        
        p=pl({'simu':{'sim_time':sim_time,
                      'sim_stop':sim_time,
                      'threads':threads}},
                  '=')
        p+=l[1]
        self.setup=Setup(**{'threads':threads,
                        'resolution':res,
                        'repetition':rep})
        
        v=simulate(builder=Builder,
                            from_disk=from_disk,
                            perturbation_list=p,
                            script_name=(__file__.split('/')[-1][0:-3]
                                         +'/data'),
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
    
    
    
    



    