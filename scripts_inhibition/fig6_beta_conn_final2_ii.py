'''
Created on Nov 13, 2014

@author: mikael

'''

from scripts_inhibition import effect_conns

scale=1
kw={
         'n_rows':3,
         'n_cols':10,  
         'w':int(72/2.54*17.6)*scale,
         'h':int((72/2.54*9)/3*(1+1/2.))*scale,
        'linewidth':1,
        'fontsize':7*scale,
        'title_fontsize':7*scale,
        'gs_builder':effect_conns.gs_builder_oi_si_simple}


def ignore_files(name):
    s=name.split('/')[-2].split('_')[-1]
#     print s, s not in ['16']
    return s in ['8', '16']
    

kwargs={'add_midpoint':True,
#         'data_path':('/home/mikael/results/papers/inhibition/network/'
#                      +'milner/simulate_beta_ZZZ_dop_effect_perturb2/'),

        'data_path':('/home/mikael/results/papers/inhibition/network/'
                     +'milner/simulate_beta_ZZZ_conn_effect_perturb_final2_ii/'),
        'midpoinst':1.0,
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
        'ignore_files':ignore_files,
        'do_plots':['si_oi_index_simple'],
        'exclude':[ 'striatum'],
        'fontsize_x':7*scale,
        'fontsize_y':7*scale,
        'exclude_no_pert':True,
        'key_no_pert':'no_pert',
        'si_oi_index_fig':kw,
        'models0': ['GA','GI', 'ST', 'SN',
                    'GP_GP', 'GA_GA', 'GI_GA', 'GI_GI'],
        'psd':{'NFFT':128*2, 
                'fs':256., 
                'noverlap':128/2},
        'oi_si_simple_clim0':[0.,2],
        'oi_si_simple_clim1':[0.5,1.5],
        'oi_si_simple_fig_kw':kw,
        'oi_fs':256,
#         'oi_upper':40*0.8,
        'oi_min':15,
        'oi_max':25,
        'title_flipped':True,
        'top_lables_fontsize':7,
}
obj=effect_conns.Main(**kwargs)
obj.do()

