'''
Created on Nov 13, 2014

@author: mikael
'''
from scripts_inhibition import effect_conns
from effect_conns import gs_builder_index2
scale=2.5
d=kw={'n_rows':5, 
      'n_cols':2, 
      'w':int(72/2.54*18)*scale, 
      'h':int(72/2.54*18)/3*scale, 
      'fontsize':7*scale,
      'title_fontsize':7*scale,
      'gs_builder':gs_builder_index2}

kwargs={'data_path':('/home/mikael/results/papers/inhibition/network/'
                    +'milner/simulate_beta_ZZZ_nuclei_effect_perturb2/'),
        'from_diks':1,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data'),
        'title':'Activation (beta)',
        'ax_4x1':True,
        'add_midpoint':False,
        'fontsize_x':7*scale,
        'fontsize_y':7*scale,
        'conn_fig_title_fontsize':7*scale,

#         'title_posy':0.2,
        'do_plots':['fr_and_oi'],
        'top_lables_fontsize':7*scale,
        'clim_raw': [[0,50], [0,1]],
        'kwargs_fig':d,
        'oi_min':15.,
        'oi_max':25,
        'oi_fs':256,
        'psd':{'NFFT':128, 
                'fs':256., 
                'noverlap':128/2},

        'separate_M1_M2':False,
        'title_flipped':True,
        'title_ax':2,}

obj=effect_conns.Main(**kwargs)
obj.do()