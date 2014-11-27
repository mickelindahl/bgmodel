'''
Created on Nov 13, 2014

@author: mikael
'''
from scripts_inhibition import effect_conns

kwargs={'data_path':('/home/mikael/results/papers/inhibition/network/'
                    +'supermicro/simulate_beta_ZZZ_conn_effect_perturb/'),
        'from_diks':1,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data'),
        'title':'Activation (beta)',
        'cohere_ylabel_ypos': -0.35,
        'cohere_xlabel0_posy':-0.4,
        'cohere_xlabel10_posy':-0.12,
        'cohere_xlabel11_posy':-0.20,
        'cohere_title_posy':1.04,
        'cohere_cmap_ypos':0.25,
        'fontsize_x':16,
        'cohere_fig_fontsize':16,
        'cohere_fig_title_fontsize':20,
        'conn_fig_title_fontsize':20,
        'title_flipped':True,
        'top_lables_fontsize':20,
        'clim_raw': [[0,4], [0,90], [0,1]]}

obj=effect_conns.Main(**kwargs)
obj.do()