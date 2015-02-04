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

    def __init__(self, period, local_num_threads, **kwargs):
        self.period=period
        self.local_num_threads=local_num_threads

        self.nets_to_run=kwargs.get('nets_to_run', ['Net_0',
                                                    'Net_1' ])
        
        self.fs=256 #Same as Mallet 2008
        
    def builder(self):
        return {}
    
    def director(self):
        return {'nets_to_run':self.nets_to_run}  

    def activity_histogram(self):
        d = {'average':False,
             'period':self.period}
        
        return d

    def pds(self):
        d = {'NFFT':128*8, 'fs':self.fs, 
             'noverlap':128*8/2, 
             'local_num_threads':THREADS}
        return d
    
      
    def coherence(self):
        d = {'fs':self.fs, 'NFFT':128 * 8, 
            'noverlap':int(128 * 8)/2, 
            'sample':100.}
        return d
    

    def phase_diff(self):
        d = {'inspect':False, 
             'lowcut':0.5, 
             'highcut':2., 
             'order':3, 
             'fs':self.fs, 
             
              #Skip convolving when calculating phase shif
             #5000 of fs=10000
#              'bin_extent':self.fs/2, #size of gaussian window for convulution of firing rates 
#              'kernel_type':'gaussian', 
#              'params':{'std_ms':250., # standard deviaion of gaussian convulution
#                        'fs':self.fs}
            }
        return d    

    def phases_diff_with_cohere(self):
        d={
            'fs':self.fs,#100.0, 
            'NFFT':128*8 , 
            'noverlap':128*8/2, 
            'sample':30.,
            
            'lowcut':0.5, 
            'highcut':2., 
            'order':3, 

             #Skip convolving when calculating phase shif

#              'bin_extent':self.fs/2,#500., 
#              'kernel_type':'gaussian', 
#              'params':{'std_ms':250., 
#                        'fs':self.fs,#100.0
#                        },
      
            'local_num_threads':self.local_num_threads}
        return d
    
    def firing_rate(self):
        d={'average':False, 
           'local_num_threads':THREADS,
#            'win':20.0,
           'time_bin':1000./self.fs}
        return d

    
    def plot_fr(self):
        d={'win':20.,
           't_start':10000.0,
           't_stop':20000.0,
           'labels':['Control', 'Lesion'],
           
           'fig_and_axes':{'n_rows':8, 
                            'n_cols':1, 
                            'w':800.0*0.55*2*0.3, 
                            'h':600.0*0.55*2*0.3, 
                            'fontsize':7,
                            'frame_hight_y':0.8,
                            'frame_hight_x':0.78,
                            'linewidth':1.}}
        return d

    def plot_coherence(self):
        d={'xlim':[0, 5]}
        return d
    
    def plot_summed(self):
        d={'xlim_cohere':[0, 5],
           'statistics_mode':'slow_wave'}
        return d
    
    def plot_summed2(self):
        d={
           'xlim_cohere':[0, 5],
           'all':True,
           'p_95':False,
           'leave_out':['control_fr', 'control_cv'],
           'statistics_mode':'slow_wave',
           'models_pdwc': ['GP_GP', 'GI_GI', 'GI_GA', 'GA_GA'],
           }
        return d
def main(builder=Builder,
         from_disk=1,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3],
         local_num_threads=10):
    
    oscillation_common.main(builder, 
                            from_disk, 
                            perturbation_list, 
                            script_name, 
                            Setup(1000.0, local_num_threads))
 
class Main():    
    def __init__(self, **kwargs):
        self.kwargs=kwargs
    
    def __repr__(self):
        return self.kwargs['script_name']
    
    
    def do(self):
        oscillation_common.main(**self.kwargs)
       
    def get_nets(self):
        return self.kwargs['setup'].nets_to_run

    def get_script_name(self):
        return self.kwargs['script_name']

    def get_name(self):
        nets='_'.join(self.get_nets()) 
        script_name=self.kwargs['script_name']
        script_name=script_name.split('/')[1].split('_')[0:2]
        script_name='_'.join(script_name)+'_'+nets
        return script_name+'_'+str(self.kwargs['from_disk']) 
# if __name__ == "__main__":
#     # stuff only to run when not called via 'import' here
#     main()



    