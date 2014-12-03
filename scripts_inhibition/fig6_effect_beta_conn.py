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
        'w':15/37.*72/2.54*17.6,
        'h':15/37.*72/2.54*17.6*9/5.,
        'cohere_ylabel_ypos': -0.75,
        'cohere_xlabel0_posy':-0.2,
        'cohere_xlabel10_posy':-0.08,
        'cohere_xlabel11_posy':-0.12,
        'cohere_title_posy':1.04,
        'cohere_cmap_ypos':0.15,
        'cohere_fontsize_x':7,
        'cohere_fontsize_y':7,
        'fontsize_x':7,
        'cohere_fig_fontsize':7,
        'cohere_fig_title_fontsize':7,
        'conn_fig_title_fontsize':7,
        'title_flipped':True,
        'do_plots':['cohere'],
        'top_lables_fontsize':7,
        'clim_raw': [[0,4], [0,90], [0,1]]}

obj=effect_conns.Main(**kwargs)
obj.do()