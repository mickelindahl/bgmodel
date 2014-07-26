'''
Created on Aug 12, 2013

@author: lindahlm
'''


import oscillation_common
import os

from toolbox.network.manager import Builder_slow_wave as Builder


import pprint
pp=pprint.pprint
    
# DISPLAY=os.environ.get('DISPLAY')
THREADS=10

class Setup(object):

    def __init__(self, period, threads):
        self.period=period
        self.threads=threads


    def activity_histogram(self):
        d = {'average':False,
             'period':self.period}
        
        return d

    def pds(self):
        d = {'NFFT':1024 * 4, 'fs':1000., 
             'noverlap':1024 * 2, 
             'threads':THREADS}
        return d
    
      
    def coherence(self):
        d = {'fs':1000.0, 'NFFT':1024 * 4, 
            'noverlap':int(1024 * 2), 
            'sample':30.}
        return d
    

    def phase_diff(self):
        d = {'inspect':False, 
             'lowcut':0.5, 
             'highcut':2., 
             'order':3, 
             'fs':1000.0, 
             'bin_extent':500., 
             'kernel_type':'gaussian', 
             'params':{'std_ms':250., 
                       'fs':1000.0}}
        
        return d    

    def phases_diff_with_cohere(self):
        d={
               'fs':100.0, 
               'NFFT':1024 * 4, 
            'noverlap':int(1024 * 2), 
            'sample':30.,
            
             'lowcut':0.5, 
             'highcut':2., 
             'order':3, 

             'bin_extent':500., 
             'kernel_type':'gaussian', 
             'params':{'std_ms':250., 
                       'fs':100.0},
      
            'threads':self.threads}
        return d
    
    def firing_rate(self):
        d={'average':False, 
           'threads':THREADS,
           'win':100.0}
        return d

    
    def plot_fr(self):
        d={'win':100.,
           't_start':10000.0,
           't_stop':20000.0,
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
        d={'xlim':[0, 5]}
        return d
    
    def plot_summed(self):
        d={'xlim_cohere':[0, 5],
           'statistics_mode':'slow_wave'}
        return d
    
    def plot_summed2(self):
        d={'xlim_cohere':[0, 5],
           'rest':False,
           'p_95':True,
           'leave_out':['control_fr', 'control_cv'],
           'statistics_mode':'slow_wave',
           'models_pdwc': ['GP_GP', 'GI_GI', 'GI_GA', 'GA_GA'],
           }
        return d
def main(builder=Builder,
         from_disk=1,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3],
         threads=10):
    
    oscillation_common.main(builder, 
                            from_disk, 
                            perturbation_list, 
                            script_name, 
                            Setup(1000.0, threads))
 
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()



    