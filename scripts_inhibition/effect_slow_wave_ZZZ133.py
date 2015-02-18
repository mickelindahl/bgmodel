'''
Created on Nov 13, 2014

@author: mikael
'''
from scripts_inhibition import effect_conns
from effect_conns import gs_builder_index2

def create_name(file_name):
    if len(file_name.split('-'))>=2:
          
        return file_name.split('-')[-1]
    else:
        return file_name

def create_name_and_x(l):
    return 1, l[0]+'_'+l[1]+'_'+l[2]+'_'+l[4]

scale=2
d=kw={'n_rows':8, 
      'n_cols':2, 
      'w':int(72/2.54*18)*scale, 
      'h':int(72/2.54*18)/3*scale, 
      'fontsize':7*scale,
      'title_fontsize':7*scale,
      'gs_builder':gs_builder_index2}

kwargs={'data_path':('/home/mikael/results/papers/inhibition/network/'
                    +'milner/simulate_slow_wave_ZZZ133/'),
        'from_diks':0,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data'),
        'title':'Slow wave',
 
        'add_midpoint':False,
        'conn_fig_title_fontsize':7*scale,
        'clim_raw': [[0,50], [0,1]],
        'compute_performance_name_and_x': create_name_and_x,
        'compute_performance_ref_key':'amp_0.1_0.95_stn_2',
        'do_plots':['index'],
        'fontsize_x':9,
        'fontsize_y':7,
        'kwargs_fig':d,
        'name_maker':create_name,
        'psd':{'NFFT':128, 
                'fs':256., 
                'noverlap':128/2},
        'oi_min':15.,
        'oi_max':25,
        'oi_fs':256,
        'separate_M1_M2':False,
        'title_flipped':True,
        'title_ax':2,
        'top_lables_fontsize':7*scale,
 
#         'title_posy':0.2,
        }

obj=effect_conns.Main(**kwargs)
obj.do()