'''
Created on Nov 13, 2014

@author: mikael

'''

from scripts_inhibition import effect_conns

scale=4
kw={
         'n_rows':3,
         'n_cols':15,  
         'w':int(72/2.54*(18-11.6))*scale,
         'h':int((72/2.54*11.6)/3*(1+1/2.))*scale,
        'linewidth':1,
        'fontsize':7*scale,
        'title_fontsize':7*scale,
        'gs_builder':effect_conns.gs_builder_oi_si_simple}

kwargs={'add_midpoint':True,
        'midpoint':3.5,
#         'data_path':('/home/mikael/results/papers/inhibition/network/'
#                      +'milner/simulate_beta_ZZZ_dop_effect_perturb2/'),

#         'add_midpoint':False,
        'data_path':('/home/mikael/results/papers/inhibition/network/'
                     +'milner/simulate_beta_ZZZ_nuclei_effect_perturb_final/'),
#         'ax_4x1':True,               s
        'from_diks':0,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data'),
        'title':'Activation (beta)',
        'w':15/37.*72/2.54*17.6,
        'h':15/37.*72/2.54*17.6*9/5.,
#         'separate_M1_M2':False,
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
         'clim_raw': [[0,50], [0,1]],
#         'do_plots':['cohere', 'mse_index', 'si_oi_index'],
        'do_plots':['si_oi_index_simple'],
#         'do_plots':['psd','psd2','si_oi_index_simple', 'si_oi_index_simple2'],
#         'do_plots':['fr_and_oi'],#, 'psd','si_oi_index_simple'],
        'exclude':[ 'striatum'],
        'fontsize_x':7*scale,
        'fontsize_y':7*scale,
        'exclude_no_pert':True,
#         'key_no_pert':'Normal',
        'key_no_pert':'no_pert',

        'models0': ['GA','GI', 'ST', 'SN',
                    'GP_GP', 'GA_GA', 'GI_GA', 'GI_GI'],
        
        'psd':{'NFFT':128*2, 
                'fs':256., 
                'noverlap':128/2},
        'oi_si_simple_fig_kw':kw,
        'oi_si_simple_clim0':[0.5,1.5],
        'oi_si_simple_clim1':[0.5,1.5],

        'oi_fs':256,
#         'oi_upper':39.,
        'oi_min':19,
        'oi_max':21,
        'title_flipped':True,
        'si_oi_index_fig':kw,
        'top_lables_fontsize':7,}

obj=effect_conns.Main(**kwargs)
obj.do()

