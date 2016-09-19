'''
Created on Nov 13, 2014

@author: mikael

'''

from scripts_inhibition import base_effect_conns
d=kw={'n_rows':8, 
        'n_cols':2, 
        'w':int(72/2.54*18), 
        'h':int(72/2.54*18)/2.5, 
        'fontsize':7,
        'title_fontsize':7,
        'gs_builder':base_effect_conns.gs_builder_conn}



kwargs={'add_midpoint':False,

        'data_path':('/home/mikael/results/papers/inhibition/network/'
                     +'milner/fig_03_sim_dop_variation_7/'),

        'from_diks':0,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data'),
        'title':'Activation (beta)',
        'w':72/2.54*17.6,
        'h':72/2.54*17.6*2.5,
        
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
#         'do_plots':['cohere', 'mse_index', 'si_oi_index'],
        'do_plots':['si_oi_index'],
        'exclude':[ 'striatum', 'GP_ST_SN', 'GA_FS'],
        'fontsize_x':7,
        'fontsize_y':7,
        
#         'key_no_pert':'Normal',
        'key_no_pert':'Normal',
#         'si_oi_index_fig':kw,
        'psd':{'NFFT':128*2, 
                'fs':256., 
                'noverlap':128/2*2},
        'models0': ['M1','M2','FS','GA','GI', 'ST', 'SN',
                    'GP_GP', 'GA_GA', 'GI_GA', 'GI_GI'],
        'oi_si_simple_clim0':[0.5,1.5],
        'oi_si_simple_clim1':[0.5,1.5],
        'oi_fs':256,
        'oi_upper':40*0.8,
        'oi_min':19,
        'oi_max':21,
        'title_flipped':True,
        'top_lables_fontsize':7,
}

obj=base_effect_conns.Main(**kwargs)
obj.do()