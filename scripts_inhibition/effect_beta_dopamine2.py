'''
Created on Nov 13, 2014

@author: mikael

'''

from scripts_inhibition import effect_conns
from effect_conns import gs_builder_conn
d=kw={'n_rows':8, 
        'n_cols':2, 
        'w':int(72/2.54*18), 
        'h':int(72/2.54*18)/3, 
        'fontsize':7,
        'title_fontsize':7,
        'gs_builder':gs_builder_conn}

kwargs={'add_midpoint':False,
#         'data_path':('/home/mikael/results/papers/inhibition/network/'
#                      +'milner/simulate_beta_ZZZ_dop_effect_perturb2/'),

        'data_path':('/home/mikael/results/papers/inhibition/network/'
                     +'milner/simulate_beta_ZZZ_dop_effect_perturb_final/'),
#        
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
        'cohere_fig_fontsize':7,
        'cohere_fig_title_fontsize':7,
        'conn_fig_title_fontsize':7,
        'cohere_ylim_bar':[0,2],
        'cohere_ylim_image':[0,4],
        'clim_raw': [[0,4], [0,90], [0,1]],
        'do_plots':['cohere', 'mse_index', 'si_oi_index'],
#         'do_plots':['si_oi_index'],
        'exclude':[ 'striatum'],
        'fontsize_x':7,
        'fontsize_y':7,
        
#         'key_no_pert':'Normal',
        'key_no_pert':'Normal',
        'si_oi_index_fig':kw,

        'models0': ['GA','GI', 'ST', 'SN',
                    'GP_GP', 'GA_GA', 'GI_GA', 'GI_GI'],
        'oi_fs':256,
        'title_flipped':True,
        'top_lables_fontsize':7,
}

obj=effect_conns.Main(**kwargs)
obj.do()