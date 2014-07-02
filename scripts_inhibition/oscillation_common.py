'''
Created on Aug 12, 2013

@author: lindahlm
'''
import numpy
import os

from copy import deepcopy
from os.path import expanduser
from toolbox import misc, pylab
from toolbox.data_to_disk import Storage_dic
from toolbox.network import manager
from toolbox.network.data_processing import Data_units_relation
from toolbox.network.manager import (add_perturbations, compute, 
                                     run, save, load, get_storage)
from toolbox.network.manager import Builder_slow_wave as Builder
from toolbox.my_signals import Data_bar
from simulate import (cmp_psd, show_fr, show_hr, show_psd,
                      show_coherence, show_phase_diff,
                      get_file_name, get_file_name_figs)
import toolbox.plot_settings as ps
import pprint
pp=pprint.pprint
    
DISPLAY=os.environ.get('DISPLAY')
THREADS=4

def mallet2008():
    d={'all':{'slow_wave':{'control':{'rate':25.8,
                                    'CV':0.49},
                         'lesioned':{'rate':20.6,
                                    'CV':1.29}},
            'activation':{'control':{'rate':33.6,
                                    'CV':0.43},
                         'lesioned':{'rate':14.8,
                                    'CV':0.74}}},
     'TI':{'slow_wave':{'lesioned':{'rate':24.4,
                                    'CV':1.28}},
           'activation':{'lesioned':{'rate':14.3,
                                     'CV':0.77}}},
     'TA':{'slow_wave':{'lesioned':{'rate':11.5,
                                    'CV':1.58}},
           'activation':{'lesioned':{'rate':19.3,
                                     'CV':0.61}}},
       'STN':{'slow_wave':{'control':{'rate':12.6,
                                    'CV':1.75},
                         'lesioned':{'rate':20.5,
                                    'CV':2.02}},
            'activation':{'control':{'rate':14.9,
                                    'CV':0.84},
                         'lesioned':{'rate':31.1,
                                    'CV':0.64}}},}
    return d


def get_all_slow_wave_rate(d):
    y = [d['all']['slow_wave']['control']['rate'], d['all']['slow_wave']['lesioned']['rate']]
    return y


def get_all_activation_rate(d):
    y = [d['all']['activation']['control']['rate'], d['all']['activation']['lesioned']['rate']]
    return y


def get_all_slow_wave_CV(d):
    y = [d['all']['slow_wave']['control']['CV'], d['all']['slow_wave']['lesioned']['CV']]
    return y


def get_all_activation_CV(d):
    y = [d['all']['activation']['control']['CV'], d['all']['activation']['lesioned']['CV']]
    return y


def get_TI_TA_slow_wave_rate(d):
    y = [d['TI']['slow_wave']['lesioned']['rate'], d['TA']['slow_wave']['lesioned']['rate']]
    return y


def get_TI_TA_slow_wave_CV(d):
    y = [d['TI']['slow_wave']['lesioned']['CV'], d['TA']['slow_wave']['lesioned']['CV']]
    return y


def get_TI_TA_activation_rate(d):
    y = [d['TI']['activation']['lesioned']['rate'], d['TA']['activation']['lesioned']['rate']]
    return y


def get_TI_TA_activation_CV(d):
    y = [d['TI']['activation']['lesioned']['CV'], d['TA']['activation']['lesioned']['CV']]
    return y

def get_STN_slow_wave_rate(d):
    y = [d['STN']['slow_wave']['control']['rate'], d['STN']['slow_wave']['lesioned']['rate']]
    return y


def get_STN_activation_rate(d):
    y = [d['STN']['activation']['control']['rate'], d['STN']['activation']['lesioned']['rate']]
    return y


def get_STN_slow_wave_CV(d):
    y = [d['STN']['slow_wave']['control']['CV'], d['STN']['slow_wave']['lesioned']['CV']]
    return y


def get_STN_activation_CV(d):
    y = [d['STN']['activation']['control']['CV'], d['STN']['activation']['lesioned']['CV']]
    return y



def plot_mallet2008():
    d=mallet2008()
    fig, axs=ps.get_figure(n_rows=3, 
                           n_cols=4, 
                           w=1200.0, 
                           h=400.0, 
                           fontsize=10) 
  
    i=0
    y = get_all_slow_wave_rate(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['GP control', 'GP lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,40])
    i += 1
    
    y = get_all_activation_rate(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['GP control', 'GP lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,40])
    i += 1
    
    y = get_all_slow_wave_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['GP control', 'GP lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,1.4])
    i += 1
    
    y = get_all_activation_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['GP control', 'GP lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,1.4])
    i += 1 
    
    y = get_TI_TA_slow_wave_rate(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['TI lesioned', 'TA lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,30])
    i += 1
    
    y = get_TI_TA_slow_wave_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['TI lesioned', 'TA lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,1.8])
    i += 1    

    y = get_TI_TA_activation_rate(d)
    Data_bar(**{'y':y}).bar(axs[i], alpha=0.5)
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['TI lesioned', 'TA lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,30])
    i += 1
    
    y = get_TI_TA_activation_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['TI lesioned', 'TA lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,1.8])
    i += 1     
    
    i=0
    y = get_all_slow_wave_rate(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['STN control', 'STN lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,40])
    i += 1
    
    y = get_STN_activation_rate(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['STN control', 'STN lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,40])
    i += 1
    
    y = get_STN_slow_wave_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['STN control', 'STN lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,1.4])
    i += 1
    
    y = get_STN_activation_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['STN control', 'STN lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,1.4])
    i += 1 
    
    return fig, axs
def add_GPe(d):
    for key in d.keys():
        if not 'GA' in d[key].keys():
            continue
        if not 'GI' in d[key].keys():
            continue
        s1=d[key]['GA']['spike_signal']
        s2=d[key]['GI']['spike_signal']
        s3=s1.merge(s2)
        d[key]['GP']={'spike_signal':s3}
        

# def cmp_statistical_test():

def get_kwargs_builder(**k_in):
    return {'print_time':False, 
            'threads':THREADS, 
            'save_conn':{'overwrite':False},
            'sim_time':10000.0, 
            'sim_stop':10000.0, 
            'size':10000.0, 
            'start_rec':0.0, 
            'sub_sampling':1}

def get_kwargs_engine():
    return {'verbose':True}

def get_networks(builder, **k_in):
    return manager.get_networks(builder, 
                                get_kwargs_builder(**k_in), 
                                get_kwargs_engine())

def create_relations(models_coher, dd):
    for key in dd.keys():
        for model in models_coher:
            k1, k2 = model.split('_')
            dd[key][model] = {}
            obj = Data_units_relation(model, dd[key][k1]['spike_signal'], 
                dd[key][k2]['spike_signal'])
            dd[key][model]['spike_signal'] = obj



def plot_spk_stats(d, axs, i, **k):
    y_mean, y_mean_SEM = [], []
    y_CV, y_CV_SEM = [], []
 
    for key in sorted(d.keys()):
        v = d[key]
        for model in ['GP']:
            st = v[model]['spike_statistic']
            y_mean.append(st.rates['mean'])
            y_mean_SEM.append(st.rates['SEM'])
            y_CV.append(st.cv_isi['mean'])
            y_CV_SEM.append(st.cv_isi['SEM'])
    
    print y_mean
    print y_mean_SEM      
    
    dm=mallet2008()
    
    mode=k.get('statistics_mode', 'activation')
    if mode=='slow_wave':
        Y=[get_all_slow_wave_rate(dm),
           get_all_slow_wave_CV(dm),
           get_TI_TA_slow_wave_rate(dm),
           get_TI_TA_slow_wave_CV(dm)]
    if mode=='activation':
        Y=[get_all_activation_rate(dm),
           get_all_activation_CV(dm),
           get_TI_TA_activation_rate(dm),
           get_TI_TA_activation_CV(dm)]
            
            
    Data_bar(**{'y':[[y_mean[0], y_mean[1]],
                     [Y[0][0],  Y[0][1]]],
                'y_std':[[y_mean_SEM[0], y_mean_SEM[1]],
                         [0.,  0.]]}).bar2(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['GP control', 'GP lesioned'])
    i += 1

    print y_CV
    print Y
    print y_CV_SEM
    Data_bar(**{'y':[[y_CV[0], y_CV[1]],
                     [Y[1][0], Y[1][1]]],
                'y_std':[[y_CV_SEM[0], y_CV_SEM[1]],
                         [0., 0.]]}).bar2(axs[i])
    
#     Data_bar(**{'y':y_CV}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['GP control', 'GP lesioned'])
    i += 1            
#     pylab.show()
    y_mean = []
    y_CV = []    
    
    for key in sorted(d.keys()):
        for model in ['GI', 'GA']:
            v = d[key]
            
            st = v[model]['spike_statistic']
            y_mean.append(st.rates['mean'])
            y_mean_SEM.append(st.rates['SEM'])
            y_CV.append(st.cv_isi['mean'])
            y_CV_SEM.append(st.cv_isi['SEM'])
        
        

    Data_bar(**{'y':y_mean[0:2]}).bar(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['TI control', 'TA control'])
    i += 1
#     pylab.show()
    
    Data_bar(**{'y':[[y_mean[2],y_mean[3]],
                     [Y[2][0], Y[2][1]]],
                'y_std':[[y_mean_SEM[2],y_mean_SEM[3]],
                         [0., 0.]]}).bar2(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['TI lesioned', 'TA lesioned'])
    
    i += 1

    Data_bar(**{'y':y_CV[0:2]}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['TI control', 'TA control'])
    i += 1

    Data_bar(**{'y':[[y_CV[2], y_CV[3]],
                     [Y[3][0], Y[3][1]]],
                'y_std':[[y_CV_SEM[2], y_CV_SEM[3]],
                         [0., 0.]]}).bar2(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['TI lesioned', 'TA lesioned'])
    
    i += 1
    
    
#     Data_bar(**{'y':y_CV}).bar(axs[i])
#     axs[i].set_ylabel('Coefficient of variation')
#     axs[i].set_xticklabels(['TA control', 'TA lesioned',
#                             'TI control', 'TI lesioned'])
#     i += 1
#     
    return i


def plot_activity_hist(d, axs, i):
    for key in sorted(d.keys()):
        d[key]['GP']['activity_histogram'].plot(axs[i])
        axs[i].set_ylim([0,50])
        
        axs[i].set_title(key)
        d[key]['GP']['activity_histogram_stat'].hist(axs[i+1])
    
    i += 2
    return i


def plot_coherence(d, axs, i, **k):
    ax = axs[i]
    for key in sorted(d.keys()):
        v = d[key]
        for j, model in enumerate(['GP_GP', 'GI_GI', 'GI_GA', 'GA_GA']):
            ax = axs[i+j]
            ch = v[model]['mean_coherence']
            ch.plot(ax)
            ax.set_title(model)
            ax.set_xlim(k.get('xlim_cohere',[0,2]))
    i+=4
    
    return i

def plot_phases_diff_with_cohere(d, axs, i, xmax=5):
    
    for key in sorted(d.keys()):
        v = d[key]
        for j, model in enumerate(['GP_GP', 'GI_GI', 'GI_GA', 'GA_GA', 
                                   'ST_ST', 'GP_ST', 'GA_ST', 'GI_ST',]):
            ax = axs[i+j]
            ch = v[model]['phases_diff_with_cohere']
            ch.hist(ax)
            
            ax.set_title(model)
    i+=8
    return i

def plot_isi(d, axs, i):
    
    for key in sorted(d.keys()):
        v = d[key]
        for j, model in enumerate(['GA']):
            ax = axs[j+i]
            obj = v[model]['spike_statistic']
            m=numpy.mean(obj.isi['raw'][0])
            std=numpy.std(obj.isi['raw'][0])
            CV=std/m
            CV2=obj.cv_isi['raw'][0]
            CV3=obj.cv_isi['mean']
            l='m:{0:.4} s:{1:.4} CV:{2:.4} CV2:{3:.4} CV3:{4:.4}'.format(m,std,
                                                                         CV, 
                                                                         CV2,
                                                                         CV3)
            ax.hist(obj.isi['raw'][0], **{'label':l})
            ax.legend()
#             ax.title()
            
    
    i+=1
        
   


def show_summed(d, **k):
#     import toolbox.plot_settings as ps  
    fig, axs=ps.get_figure(n_rows=5, 
                           n_cols=4, 
                           w=1400.0, 
                           h=800.0, 
                           fontsize=10)   
    


    i=0
    i = plot_spk_stats(d, axs, i, **k)

    i = plot_activity_hist(d, axs, i)
    i = plot_coherence(d, axs, i, **k)
    i = plot_phases_diff_with_cohere(d, axs, i)
#     pylab.show()    
    return fig
        

class Setup(object):
    
    def __init__(self, period, threads):
        self.period=period
        self.threads=threads
   
   
   
    def activity_histogram(self):
        d = {'average':False,
             'period':self.period}
        
        return d
    
    def pds(self):
        d = {'NFFT':1024 * 4, 
             'fs':1000., 
             'noverlap':1024 * 2, 
             'threads':self.threads}
        return d
    
   
    def coherence(self):
        d = {'fs':1000.0, 'NFFT':1024 * 4, 
            'noverlap':int(1024 * 2), 
            'sample':30.,  
             'threads':self.threads}
        return d
    

    def phase_diff(self):
        d = {'inspect':False, 
             'lowcut':0.5, 
             'highcut':2., 
             'order':3, 
             'fs':1000.0, 
             'bin_extent':250., 
             'kernel_type':'gaussian', 
             'params':{'std_ms':125., 
                       'fs':1000.0}, 
             'threads':self.threads}
        
        return d
    
    def phases_diff_with_cohere(self):
        kwargs={
                'NFFT':1024 * 4, 
                'fs':100., 
                'noverlap':1024 * 2, 
                'sample':10.,   
                
               'lowcut':0.5, 
               'highcut':2., 
               'order':3, 

             'bin_extent':250., 
             'kernel_type':'gaussian', 
             'params':{'std_ms':125., 
                       'fs':100.0}, 
      
                'threads':self.threads}
        return kwargs
    
    def firing_rate(self):
        d={'average':False, 
           'threads':self.threads, 
           'win':100.0}
        return d
    
    def plot_fr(self):
        d={'win':100.,
           't_start':0.0,
           't_stop':20000.0}
        return d

    def plot_coherence(self):
        d={'xlim':[0, 10]}
        return d
    
    def plot_summed(self):
        d={'xlim_cohere':[0, 10],
           'statistics_mode':'activation'}
        return d
    
def simulate(builder=Builder,
         from_disk=0,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3],
         setup=Setup(1000.0, THREADS)):
    
    k = get_kwargs_builder()

    home = expanduser("~")
    d_pds = setup.pds()
    d_cohere = setup.coherence()
    d_phase_diff = setup.phase_diff()
    d_phases_diff_with_cohere = setup.phases_diff_with_cohere()
    d_firing_rate = setup.firing_rate()
    d_activity_hist=setup.activity_histogram()
    
    attr = [
        'firing_rate', 
        'mean_rates', 
        'spike_statistic']
    
    attr2=['psd',
           'activity_histogram',
           'activity_histogram_stat']
    
    attr_coher = [
        'phase_diff', 
        'phases_diff_with_cohere',
        'mean_coherence'
        ]
    
    kwargs_dic = {'firing_rate':d_firing_rate, 
                  'mean_rates':{'t_start':k['start_rec'] + 1000.}, 
                  'mean_coherence':d_cohere, 
                  'phase_diff':d_phase_diff, 
                  'phases_diff_with_cohere':d_phases_diff_with_cohere,
                  'spike_statistic':{'t_start':k['start_rec'] + 1000.}}
    
    file_name = get_file_name(script_name, home)
    file_name_figs = get_file_name_figs(script_name, home)
    
    models = ['M1', 'M2', 'FS', 'GI', 'GA', 'ST', 'SN', 'GP']
    models_coher = ['GI_GA', 'GI_GI', 'GA_GA', 'GA_ST', 'GI_ST', 'GP_GP',
                     'ST_ST', 'GP_ST',]
    
    info, nets, _ = get_networks(builder)
    add_perturbations(perturbation_list, nets)
    sd = get_storage(file_name, info)
    
    d = {}
    
    from_disks = [from_disk] * 2
    for net, fd in zip(nets.values(), from_disks):
        if fd == 0:
            dd = run(net)
            add_GPe(dd)
            dd = compute(dd, models, attr, **kwargs_dic)
            save(sd, dd)
        elif fd == 1:
            filt = [net.get_name()] + models + ['spike_signal']
             
            dd = load(sd, *filt)    
            
                        
#             dd = compute(dd, models, ['firing_rate'], **kwargs_dic)     
#             pp(dd)
            dd = compute(dd, models, attr, **kwargs_dic)
            save(sd, dd)
            
#             filt = [net.get_name()] + models + ['spike_signal']+attr
            
#             dd = load(sd, *filt)
            
            for keys, val in misc.dict_iter(dd):
                if keys[-1]=='spike_signal':
                    val.wrap.allowed.append('get_phases_diff_with_cohere')
            
            create_relations(models_coher, dd)
            dd = compute(dd, models_coher, attr_coher, **kwargs_dic)
            
            cmp_psd(d_pds, models, dd) 
            cmp_activity_hist(models, dd, **d_activity_hist )
            cmp_activity_hist_stat(models, dd, **{'average':False})          
            
            save(sd, dd)
        elif fd == 2:
            filt = ([net.get_name()] 
                    + models + models_coher 
                    + attr + attr2 + attr_coher)
            dd = load(sd, *filt)

            #             cmp_statistical_test(models, dd)
        d = misc.dict_update(d, dd)
    
    return file_name_figs, from_disks, d, models, models_coher


def cmp_activity_hist(models, d, **kwargs):
    for key1 in d.keys():
        for model in models:
            obj=d[key1][model]['firing_rate'].get_activity_histogram(**kwargs)
            d[key1][model]['activity_histogram'] = obj

def cmp_activity_hist_stat(models, d, **kwargs):       

    for key1 in d.keys():
        for model in models:
            v=d[key1][model]['activity_histogram']
            obj=v.get_activity_histogram_stat(**kwargs)
            d[key1][model]['activity_histogram_stat'] = obj
            

def create_figs(file_name_figs, from_disks, d, models, models_coher, setup):

    d_plot_fr=setup.plot_fr()
    d_plot_coherence=setup.plot_coherence()
    d_plot_summed=setup.plot_summed()

    sd_figs = Storage_dic.load(file_name_figs)
    if numpy.all(numpy.array(from_disks) == 2):
        figs = []
        figs.append(show_fr(d, models, **d_plot_fr))
        figs.append(show_hr(d, models))
        figs.append(show_psd(d, models=models))
        figs.append(show_coherence(d, models=models_coher, **d_plot_coherence))
        figs.append(show_phase_diff(d, models=models_coher))
        
        figs.append(show_summed(d, **d_plot_summed))

        sd_figs.save_figs(figs, format='png')

def main(*args, **kwargs):
    
    v=simulate(*args, **kwargs)
    file_name_figs, from_disks, d, models, models_coher = v

    
    setup=args[-1]
    create_figs(file_name_figs, from_disks, d, models, models_coher, setup)
    
#     if DISPLAY: pylab.show()  
    
    return d

import unittest
class TestOcsillation(unittest.TestCase):     
    def setUp(self):
        from toolbox.network.default_params import Perturbation_list as pl
        from_disk=2
        sim_time=40000.0
        size=5000.0
        threads=12
        s=''
        p=pl({'simu':{'sim_time':sim_time,
                      'sim_stop':sim_time,
                      'threads':threads},
                  'netw':{'size':size}},
                  '=')
        
        for amp in [
                0.9, 
                ]: 
            d={'type':'oscillation', 
               'params':{
                         'p_amplitude_mod':amp,
                         'freq': 1.,
                         'freq_min':0.5,
                         'freq_max':1.5, 
                         'period':'uniform'}} 
            dd={}
            for key in ['C1', 'C2', 'CF', 'CS']: 
                dd=misc.dict_update(dd, {'netw': {'input': {key:d} } })     
                         
            p+=pl(dd,'=',**{'name':'amp_'+str(amp)})
        s='1'
                  
#      
        
        self.setup=Setup(1000.0, THREADS)
        v=simulate(builder=Builder,
                            from_disk=from_disk,
                            perturbation_list=p,
                            script_name=(__file__.split('/')[-1][0:-3]
                                         +'/data'+s),
                            setup=self.setup)
        
        file_name_figs, from_disks, d, models, models_coher = v
        
        
        self.file_name_figs=file_name_figs
        self.from_disks=from_disks
        self.d=d
        self.models=models
        self.models_coher=models_coher
        



#     def test_isi(self):
# #         import toolbox.plot_settings as ps
#         fig, axs=ps.get_figure(n_rows=1, 
#                            n_cols=1, 
#                            w=1000.0, h=800.0, fontsize=10)  
#           
#         plot_isi(self.d, axs, 0)
# #         pylab.show()
#         print self.d

#         
#     def test_empty(self):
#         pass
    
#     def test_plot_mallet2008(self):
#         plot_mallet2008()
#         pylab.show()
#         
#     def testPlotStats(self):
# #         import toolbox.plot_settings as ps
#         fig, axs=ps.get_figure(n_rows=3, 
#                            n_cols=2, 
#                            w=1000.0, h=800.0, fontsize=10)  
#           
#         plot_spk_stats(self.d, axs, 0, **{'statistics_mode': 'slow_wave'})
# #         pylab.show()
#         print self.d

#     def testPlotActivityHist(self):
#         import toolbox.plot_settings as ps
#         fig, axs=ps.get_figure(n_rows=2, 
#                            n_cols=2, 
#                            w=1000.0, h=800.0, fontsize=10)  
#         
#         plot_activity_hist(self.d, axs, 0)
# #         plot_spk_stats(self.d, axs, 0)
#         pylab.show()
#         print self.d

#     def testPlotCoherence(self):
#         import toolbox.plot_settings as ps
#         fig, axs=ps.get_figure(n_rows=1, 
#                            n_cols=1, 
#                            w=1000.0, h=800.0, fontsize=10)  
#         
#         plot_coherence(self.d, axs, 0)
# #         plot_spk_stats(self.d, axs, 0)
#         pylab.show()
#         print self.d
# 
#     def testPlotPhasesDffWithCohere(self):
#         import toolbox.plot_settings as ps
#         fig, axs=ps.get_figure(n_rows=3, 
#                            n_cols=2, 
#                            w=1000.0, h=800.0, fontsize=10)  
#         
#         plot_phases_diff_with_cohere(self.d, axs, 0)
# #         plot_spk_stats(self.d, axs, 0)
#         pylab.show()
#         print self.d   
        
#          
#     def testShowSummed(self):
#         show_summed(self.d, **{'xlim_cohere':[0,7], 
#                                'statistics_mode':'slow_wave'})
#         pylab.show()
             


    def test_create_figs(self):
        create_figs(self.file_name_figs, 
                    self.from_disks, 
                    self.d, 
                    self.models, 
                    self.models_coher,
                    self.setup)
        pylab.show()
    
#     def test_show_fr(self):
#         show_fr(self.d, self.models, **{'win':20.,
#                                         't_start':4000.0,
#                                         't_stop':5000.0})
#         pylab.show()
 
#     def test_show_coherence(self):
#         show_coherence(self.d, self.models_coher, **{'xlim':[0,10]})
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
    





    