'''
Created on Apr 6, 2015

@author: mikael
'''

from scripts_inhibition import effect_conns, oscillation_common, simulate_slow_wave, simulate_beta
from toolbox.network.manager import get_storage, save
from scripts_inhibition.simulate import get_file_name, save_figures
from toolbox import misc
from matplotlib import ticker
from toolbox.network.manager import get_storage_list,load

import toolbox.plot_settings as ps
import matplotlib.gridspec as gridspec
import pylab
import numpy
import pprint

pp=pprint.pprint


def plot(file_name, figs, setup, flag,  **k):
    nets=['Net_0', 'Net_1']
    
    attr = [
        'firing_rate', 
        'mean_rates', 
        'spike_statistic'
        ]
    
    attr2=['psd',
           'activity_histogram',
           'activity_histogram_stat']
    
    attr_coher = [
                  'phase_diff', 
                  'phases_diff_with_cohere',
                  'mean_coherence'
                 ]
    
    models = ['M1', 'M2', 'FS', 'GI', 'GA', 'ST', 'SN', 'GP']
    models_coher = ['GI_GA', 'GI_GI', 'GA_GA', 'GA_ST', 'GI_ST', 'GP_GP',
                     'ST_ST', 'GP_ST',]
    
    # Adding nets no file name
    sd_list=get_storage_list(nets, file_name, '')
    
    d = {}
    
    for sd, net in zip(sd_list, nets):
        
    
        filt = ([net] 
                + models + models_coher 
                + attr + attr2 + attr_coher)
        dd = load(sd, *filt)
    
            #             cmp_statistical_test(models, dd)
        d = misc.dict_update(d, dd)
    
     
    kw=setup.plot_summed2()
    kw['alphas']=[1.,1.]
    kw['coherence_xcut']=[0,50]
    kw['coherence_color']=['grey', 'k']
    kw['coherence_p_conf95_linestyle']='--'
    kw['hatchs_ti_ti']=['','']
    kw['label_model']='Black=Model'
    kw['label_exp']='White=Exp., Mallet et al 2008'
    kw['linewidth']=1.
    kw['set_text_on_bars']=False
    kw['top_label']=False
    kw['phases_diff_with_cohere_colors']=['grey','k']
    kw['phases_diff_with_cohere_xcut']=[-numpy.pi*0.97, numpy.pi**0.97]
    kw['phases_diff_with_cohere_remove_peaks']=True
    kw['scale']=1
    kw['spk_stats_color_axis']=1
    kw['spk_stats_colors']=['k', 'w']
    kw['spk_stats_colors_ti_ta']=['k','w']
    kw['xlim_cohere']=[-1, 51]
    

    kw.update(k)
    figs.append(oscillation_common.show_summed2(d, **kw))
        
    kw=setup.plot_summed_STN()
    kw['alphas']=[1.,1.]
    kw['coherence_color']=['grey','k']
    kw['coherence_p_conf95_linestyle']='--'
    kw['hatchs_ti_ti']=['','']
    kw['label_model']='Black=Model'
    kw['label_exp']='White=Exp., Mallet et al 2008'
    kw['linewidth']=1.
    kw['phases_diff_with_cohere_xcut']=[-numpy.pi*0.97, numpy.pi**0.97]
    kw['phases_diff_with_cohere_remove_peaks']=True
    kw['phases_diff_with_cohere_colors']=['grey','k']
    kw['scale']=1
    kw['set_text_on_bars']=False
    kw['spk_stats_colors']=['k', 'w']
    kw['spk_stats_colors_ti_ta']=['k','w']
    kw['spk_stats_color_axis']=1
    kw['top_label']=False    
    kw['xlim_cohere']=[-1, 51]
    if flag=='slow_wave':
        kw['ylim_cohere']=[0, 1.0]
    elif flag=='beta':
        kw['ylim_cohere']=[0, 0.5]
        
    kw['coherence_xcut']=[0,50]
    kw.update(k)
    figs.append(oscillation_common.show_summed_STN(d, **kw))
    
    return figs


figs=[]
file_name=('/home/mikael/results/papers/inhibition/network/'
           +'milner/simulate_beta_ZZZ34_slow/'
           +'script_0016_GAGA_20.0_GIGA_10.0-amp_0.16_1.05_stn_3.0')
setup=simulate_beta.Setup(20,1)
plot(file_name, figs, setup, 'beta')


file_name=('/home/mikael/results/papers/inhibition/network/'
           +'milner/simulate_slow_wave_ZZZ34_slow_sw/'
           +'script_0007_GAGA_20.0_GIGA_10.0-amp_0.11-0.85')
setup=simulate_slow_wave.Setup(1000,1)
plot(file_name, figs, setup, 'slow_wave', **{'xlim_cohere':[-0.1, 5.1],
                                             'coherence_xcut':[0, 5],
                                             'phases_diff_with_cohere_remove_peaks':True})


save_figures(figs, __file__.split('/')[-1][0:-3]+'/data', dpi=200)
pylab.show()