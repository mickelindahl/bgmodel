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

kwargs={'data_path':('/home/mikael/results/papers/inhibition/network/'
                    +'supermicro/simulate_slow_wave_ZZZ_conn_effect_perturb/'),
        'from_diks':0,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data'),
        'title':'Activation (beta)',
        'fontsize_x':7,
        'fontsize_y':7,
        'conn_fig_title_fontsize':7,
        'title_flipped':True,
#         'title_posy':0.2,
        'do_plots':['index'],
        'top_lables_fontsize':7,
        'clim_raw': [[0,5], [0,50], [0,1]],
        'kwargs_fig':d,
        'oi_min':0.5,
        'oi_max':1.5,
        'psd':{'NFFT':128*8*4, 
                'fs':1000., 
                'noverlap':128*8*4/2}}

obj=effect_conns.Main(**kwargs)
obj.do()