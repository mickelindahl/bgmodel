'''
Created on 25 mar 2014

@author: mikael
'''

import numpy
import pylab
import toolbox.plot_settings as ps

from os.path import expanduser
from simulate import show_fr, get_file_name, get_file_name_figs
from toolbox import misc
from toolbox.data_to_disk import Storage_dic
from toolbox.network import manager
from toolbox.network.manager import add_perturbations, compute, run, save, load
from toolbox.network.manager import Builder_Go_NoGo_with_lesion as Builder

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
            'sim_time':1000.0, # need to sum to the durationof silence + burst in burst3 setup
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

# def show_fr(d):
#     _, axs=ps.get_figure(n_rows=7, n_cols=1, w=1000.0, h=800.0, fontsize=10)  
#     for model, i in [['M1',0], ['M2', 1], ['GI',2], ['SN',3]]:
#         d[model]['firing_rate'].plot(ax=axs[i],  **{'label':model})
        

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


def show_3d(d,**k):
    models=['SN']
    res=k.get('resolution')
    n=len(models)
    m=len(d.keys())
    attr='mean_rate_slices'
    fig, axs=ps.get_figure(n_rows=m, n_cols=2, w=1000.0, h=800.0, fontsize=10, 
                           projection='3d')        
     
    i=0
    for model in models:
        alpha=0.8
        dcm={'Net_0':'jet',
             'Net_1':'coolwarm',}
        for key in sorted(d.keys()):
            obj0=d[key]['set_0'][model][attr]
            obj1=d[key]['set_1'][model][attr]
            args=[obj0.x, obj1.x,
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
            axs[i+1].plot_surface(x, y, z_std, cmap='coolwarm', rstride=1, cstride=1, 
                                linewidth=0, 
                                shade=True,
                                alpha=alpha,
                                antialiased=False)
             
             
    #                 alpha-=0.3
            i+=2
    #                 pylab.show()
    #                 print v
               
    for ax in axs:
        ax.view_init(elev=5)
    
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

    def plot_fr(self):
        d={'win':10.,
           'by_sets':True,
           't_start':0.0,
           't_stop':10000.0}
        return d
    
    def plot_3d(self):
        d={'resolution':self.res}
        return d

        
def simulate(builder, from_disk, perturbation_list, script_name, setup):
    home = expanduser("~")
    
    file_name = get_file_name(script_name, home)
    file_name_figs = get_file_name_figs(script_name, home)
    
    d_firing_rate = setup.firing_rate()
    
    models = ['M1', 'M2', 'SN']
    sets = ['set_0', 'set_1']
    
    info, nets, intervals, rep, x, y = get_networks(builder, 
                                                  **setup.builder())
    d_mr_slices = {'intervals':intervals[1], 
        'repetition':rep, 
        'set_0':{'x':x}, 
        'set_1':{'x':y}, 
        'sets':[0, 1]}
    add_perturbations(perturbation_list, nets)    
    sd = Storage_dic.load(file_name)
    sd.add_info(info)
    sd.garbage_collect()
    d = {}
    from_disks = [from_disk] * len(nets.keys())
    for net, fd in zip(nets.values(), from_disks):
        if fd == 0:
            dd = run(net)
#             save(sd, dd)
#             dd = compute(dd, models, ['mean_rate_slices'], **{'mean_rate_slices':d_mean_rate_slices})
#             dd = compute(dd, models, ['firing_rate'], **{'firing_rate':d_firing_rate})
            save(sd, dd)
        
        elif fd == 1:
            filt = [net.get_name()] + models + ['spike_signal']
            dd = load(sd, *filt)
            dd = compute(dd, models, ['mean_rate_slices'], **{'mean_rate_slices':d_mr_slices})
            dd = compute(dd, models, ['firing_rate'], **{'firing_rate':d_firing_rate})
            save(sd, dd)
        
        elif fd == 2:
            filt = [net.get_name()] + sets + models + ['spike_signal', 
                'firing_rate', 
                'mean_rate_slices']
            dd = load(sd, *filt)
#             dd=cmp_mean_rates_intervals(dd, intervals[1], x, rep, **{'sets':[0,1]})
        d = misc.dict_update(d, dd)
    
    return file_name, file_name_figs, from_disks, d, models


def create_figs(setup, file_name_figs, d, models):
    
    sd_figs = Storage_dic.load(file_name_figs)
    figs = []
    
    d_plot_fr = setup.plot_fr()
    d_plot_3d=setup.plot_3d()
    for i in range(5):
        figs.append(show_fr(d['Net_'+str(i)], models, **d_plot_fr))
    figs.append(show_3d(d, **d_plot_3d))
    
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
class TestOcsillation(unittest.TestCase):     
    def setUp(self):
        from toolbox.network.default_params import Perturbation_list as pl
        from_disk=2
        
        import oscillation_perturbations as op
        
        rep, res=1, 5
        
        sim_time=rep*res*1000.0
      
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
        
        file_name_figs, from_disks, d, models= v
        
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
                         TestOcsillation
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)
    
    
    
    



    