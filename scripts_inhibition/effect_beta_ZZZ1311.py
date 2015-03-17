'''
Created on Nov 13, 2014

@author: mikael
'''
from scripts_inhibition import effect_conns
from effect_conns import gs_builder_index2

def create_name(file_name):
    if len(file_name.split('-'))>=3:
        s=file_name.split('-')[-3].split('/')[-1][7:]
        return s
    else:
        return file_name

def create_name_and_x(l):
    return 1, '_'.join(l)

scale=2
d=kw={'n_rows':6, 
      'n_cols':2, 
      'w':int(72/2.54*18)*scale, 
      'h':int(72/2.54*18)/1.5*scale, 
      'fontsize':7*scale,
      'title_fontsize':7*scale,
      'gs_builder':gs_builder_index2}

kwargs={'data_path':('/home/mikael/results/papers/inhibition/network/'
                    +'milner/simulate_beta_ZZZ1311/'),
        'from_diks':1,
        'script_name':(__file__.split('/')[-1][0:-3]+'/data'),
        'title':'Activation (beta)',
         'ax_4x1':True,
        'add_midpoint':False,
        'conn_fig_title_fontsize':7*scale,
        'clim_raw': [[0,50], [0,1]],
        'compute_performance_name_and_x': create_name_and_x,
        'compute_performance_ref_key':'0008_mod_GI_M2_8',
        'do_plots':['fr_and_oi'],
        'coher_label':'Oscillation', 
        'fr_label':"Firing rate",
        'fontsize_x':7*scale,
        'fontsize_y':7*scale,
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