'''
Created on Aug 12, 2013

@author: lindahlm
'''
print 'before oscillation_common import'
import oscillation_common
print 'after oscillation_common import'
import os

from toolbox.network.manager import Builder_beta as Builder


import pprint
pp=pprint.pprint
    
DISPLAY=os.environ.get('DISPLAY')
THREADS=10

class Setup(object):

    def __init__(self, period, local_num_threads, **kwargs):
        self.period=period
        self.local_num_threads=local_num_threads

        self.nets_to_run=kwargs.get('nets_to_run', ['Net_0',
                                                    'Net_1' ])
        
   
  
    def builder(self):
        return {}
    
    def director(self):
        return {'nets_to_run':self.nets_to_run}  
    
    def activity_histogram(self):
        d = {'average':False,
             'period':self.period}
        
        return d

    def pds(self):
        d = {'NFFT':256, 'fs':1000., 
             'noverlap':256/2, 
             'local_num_threads':self.local_num_threads}
        return d
    
      
    def coherence(self):
        d = {'fs':1000.0, 'NFFT':256, 
            'noverlap':int(256/2), 
            'sample':30., 
             'local_num_threads':self.local_num_threads}
        return d
    

    def phase_diff(self):
        d = {'inspect':False, 
             'lowcut':15, 
             'highcut':25., 
             'order':3, 
             'fs':250.0, 
             'bin_extent':20., 
             'kernel_type':'gaussian', 
             'params':{'std_ms':10., 
                       'fs':250.0}, 
             'local_num_threads':self.local_num_threads}
        
        return d

    def phases_diff_with_cohere(self):
        d={
                'fs':1000.0, 
                'NFFT':256, 
            'noverlap':int(256/2), 
            'sample':30.,  
            
             'lowcut':15, 
             'highcut':25., 
               'order':3, 

             'bin_extent':20., 
             'kernel_type':'gaussian', 
             'params':{'std_ms':10., 
                       'fs':1000.0}, 
      
                'local_num_threads':self.local_num_threads}
        return d

    def firing_rate(self):
        d={'average':False, 
           'local_num_threads':self.local_num_threads,
           'win':100.0}
        return d

    def plot_fr(self):
        d={'win':10.,
           't_start':5000.0,
           't_stop':6000.0,
           'labels':['Control', 'Lesion'],
           
            'fig_and_axes':{'n_rows':8, 
                                        'n_cols':1, 
                                        'w':800.0*0.55*2, 
                                        'h':600.0*0.55*2, 
                                        'fontsize':11*2,
                                        'frame_hight_y':0.8,
                                        'frame_hight_x':0.78,
                                        'linewidth':3.}}
        return d


    def plot_coherence(self):
        d={'xlim':[0, 50],
           'statistics_mode':'activation'}
        return d
    
    def plot_summed(self):
        d={'xlim_cohere':[0, 50]}
        return d
    
    def plot_summed2(self):
        d={'xlim_cohere':[0, 50],
           'rest':True,
           'p_95':False,
           'leave_out':['control_fr', 'control_cv'],
           'statistics_mode':'activation',
           'models_pdwc': ['GP_GP', 'GI_GI', 'GI_GA', 'GA_GA'],
           }
        return d
    
class Main():    
    def __init__(self, **kwargs):
        self.kwargs=kwargs
    
    def __repr__(self):
        return self.kwargs['script_name']

    
    def do(self):
        oscillation_common.main(**self.kwargs)
        
# def main(builder=Builder,
#          from_disk=2,
#          perturbation_list=None,
#          script_name=__file__.split('/')[-1][0:-3],
#          threads=10):
#     
# 
#  
# if __name__ == "__main__":
#     # stuff only to run when not called via 'import' here
#     main()



    