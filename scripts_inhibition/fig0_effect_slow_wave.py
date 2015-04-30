'''
Created on Nov 13, 2014

@author: mikael
'''
from scripts_inhibition import effect_conns

scale=1
kw={'n_rows':11, 
        'n_cols':24, 
        'w':72*17.6*2/3./2.54*scale, 
        'h':150*scale, 
        'fontsize':7*scale,
        'frame_hight_y':0.5,
        'frame_hight_x':0.7,
        'title_fontsize':7*scale,
        'gs_builder':effect_conns.gs_builder_conn2}

kwargs={'data_path':('/home/mikael/results/papers/inhibition/network/'
                    +'milner/simulate_slow_wave_ZZZ_conn_effect_perturb_final/'),
        'from_diks':0,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data'),
        'exclude':['M1', 'M2', 'FS', 'ST','SN', 
                   'M1_SN', 'M2_GI','ST_SN', 'FS_FS', 'FS_M1', 'FS_M2', 
                   'GI_SN', 'M1_M1', 'M1_M2', 'M2_M1', 'M2_M2',
                   'GI_ST'],
        
        'fontsize_x':7*scale,
        'fontsize_y':7*scale,
        'separate_M1_M2':False,
        'clim_raw':[[0,40], [0, 1]],
        'clim_gradient':[[-25, 25], [-0.6, 0.6]],
        'kwargs_fig':kw,
        'fr_label':'Hz',
        'coher_label':'Coherence',
        'title':'Slow wave',
        'title_flipped':True,
        'title_ax':3, 
        'title_posy':0.9,
        'do_plots':['firing_rate'],
        'conn_fig_title_fontsize':7*scale,
        'top_lables_fontsize':7*scale,
        'top_labels_show':False}


obj=effect_conns.Main(**kwargs)
obj.do()