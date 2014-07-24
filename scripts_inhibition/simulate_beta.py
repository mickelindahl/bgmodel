'''
Created on Aug 12, 2013

@author: lindahlm
'''
import oscillation_common
import os

from toolbox.network.manager import Builder_beta as Builder


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
        d = {'NFFT':256, 'fs':1000., 
             'noverlap':256/2, 
             'threads':self.threads}
        return d
    
      
    def coherence(self):
        d = {'fs':1000.0, 'NFFT':256, 
            'noverlap':int(256/2), 
            'sample':30., 
             'threads':self.threads}
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
             'threads':self.threads}
        
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
      
                'threads':self.threads}
        return d

    def firing_rate(self):
        d={'average':False, 
           'threads':THREADS,
           'win':100.0}
        return d

    def plot_fr(self):
        d={'win':100.,
           't_start':5000.0,
           't_stop':10000.0,
           'labels':['Control', 'Lesion'],
           
            'fig_and_axes':{'n_rows':8, 
                                        'n_cols':1, 
                                        'w':800.0, 
                                        'h':600.0, 
                                        'fontsize':12,
                                        'frame_hight_y':0.9}}
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
def main(builder=Builder,
         from_disk=2,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3],
         threads=10):
    
    oscillation_common.main(builder, 
                            from_disk, 
                            perturbation_list, 
                            script_name, 
                            Setup(1000.0/20.0, threads))
 
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()



    