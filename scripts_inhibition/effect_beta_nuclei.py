'''
Created on Nov 13, 2014

@author: mikael
'''
from scripts_inhibition import effect_conns

kwargs={'data_path':('/home/mikael/results/papers/inhibition/network/'
                    +'milner/simulate_beta_ZZZ_nuclei_effect_perturb/'),
        'from_diks':1,
        'midpoint':3.5,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data'),
        'h':500,
        'cohere_ylim':[0,4],
        'cohere_gs':effect_conns.gs_builder_coher2,
        'cohere_nrows':5,
        'title':'Activation (beta)',
        'cohere_ylabel':'Perturbed neuron type',
        'cohere_ylabel_ypos': -0.35,
        'cohere_xlabel0_posy':-0.4,
        'cohere_xlabel10_posy':-0.12,
        'cohere_xlabel11_posy':-0.20,
        'cohere_title_posy':1.04,
        'cohere_cmap_ypos':0.25,
        'fontsize_x':16,
        
        }

obj=effect_conns.Main(**kwargs)
obj.do()