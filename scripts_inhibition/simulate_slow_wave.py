'''
Created on Aug 12, 2013

@author: lindahlm
'''
import numpy

import oscillation_common
import os

from toolbox.network.manager import Builder_slow_wave as Builder


import pprint
pp=pprint.pprint
    
DISPLAY=os.environ.get('DISPLAY')
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
        kwargs={
               'fs':1000.0, 
               'NFFT':1024 * 4, 
            'noverlap':int(1024 * 2), 
            'sample':30.,
            
             'lowcut':0.5, 
             'highcut':2., 
             'order':3, 

             'bin_extent':500., 
             'kernel_type':'gaussian', 
             'params':{'std_ms':250., 
                       'fs':1000.0},
      
            'threads':self.threads}
    
    def firing_rate(self):
        d={'average':False, 
           'threads':THREADS,
           'win':100.0}
        return d

    
    def plot_fr(self):
        d={'win':20.,
           't_start':0.0,
           't_stop':numpy.Inf}
        return d

    def plot_coherence(self):
        d={'xlim':[0, 5]}
        return d
    
    def plot_summed(self):
        d={'xlim_cohere':[0, 5]}
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



    