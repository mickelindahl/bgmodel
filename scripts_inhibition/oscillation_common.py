'''
Created on Aug 12, 2013

@author: lindahlm
'''
import numpy
import os


from os.path import expanduser

from toolbox import misc, pylab

from matplotlib.font_manager import FontProperties

from toolbox.data_to_disk import Storage_dic
from toolbox.network import manager
from toolbox.network.data_processing import Data_units_relation
from toolbox.network.manager import (add_perturbations, compute, 
                                     run, save, load, get_storage_list)
from toolbox.network.manager import Builder_slow_wave as Builder
from toolbox.my_signals import Data_bar
from simulate import (cmp_psd, show_fr, show_hr, show_psd,
                      show_coherence, show_phase_diff,
                      get_file_name, get_file_name_figs,
                      get_path_nest)
import toolbox.plot_settings as ps
import pprint
pp=pprint.pprint
    
DISPLAY=os.environ.get('DISPLAY')
THREADS=4

def mallet2008():
    d={'all':{'slow_wave':{'control':{'rate':25.8,
                                    'CV':0.49},
                         'lesioned':{'rate':20.6,
                                    'CV':1.29}},
            'activation':{'control':{'rate':33.6,
                                    'CV':0.43},
                         'lesioned':{'rate':14.8,
                                    'CV':0.74}}},
     'TI':{'slow_wave':{'lesioned':{'rate':24.4,
                                    'CV':1.28}},
           'activation':{'lesioned':{'rate':14.3,
                                     'CV':0.77}}},
     'TA':{'slow_wave':{'lesioned':{'rate':11.5,
                                    'CV':1.58}},
           'activation':{'lesioned':{'rate':19.3,
                                     'CV':0.61}}},
       'STN':{'slow_wave':{'control':{'rate':12.6,
                                    'CV':1.75},
                         'lesioned':{'rate':20.5,
                                    'CV':2.02}},
            'activation':{'control':{'rate':14.9,
                                    'CV':0.84},
                         'lesioned':{'rate':31.1,
                                    'CV':0.64}}},}
    return d


def get_all_slow_wave_rate(d):
    y = [d['all']['slow_wave']['control']['rate'], d['all']['slow_wave']['lesioned']['rate']]
    return y


def get_all_activation_rate(d):
    y = [d['all']['activation']['control']['rate'], d['all']['activation']['lesioned']['rate']]
    return y


def get_all_slow_wave_CV(d):
    y = [d['all']['slow_wave']['control']['CV'], d['all']['slow_wave']['lesioned']['CV']]
    return y


def get_all_activation_CV(d):
    y = [d['all']['activation']['control']['CV'], d['all']['activation']['lesioned']['CV']]
    return y


def get_TI_TA_slow_wave_rate(d):
    y = [d['TI']['slow_wave']['lesioned']['rate'], d['TA']['slow_wave']['lesioned']['rate']]
    return y


def get_TI_TA_slow_wave_CV(d):
    y = [d['TI']['slow_wave']['lesioned']['CV'], d['TA']['slow_wave']['lesioned']['CV']]
    return y


def get_TI_TA_activation_rate(d):
    y = [d['TI']['activation']['lesioned']['rate'], d['TA']['activation']['lesioned']['rate']]
    return y


def get_TI_TA_activation_CV(d):
    y = [d['TI']['activation']['lesioned']['CV'], d['TA']['activation']['lesioned']['CV']]
    return y

def get_STN_slow_wave_rate(d):
    y = [d['STN']['slow_wave']['control']['rate'], d['STN']['slow_wave']['lesioned']['rate']]
    return y


def get_STN_activation_rate(d):
    y = [d['STN']['activation']['control']['rate'], d['STN']['activation']['lesioned']['rate']]
    return y


def get_STN_slow_wave_CV(d):
    y = [d['STN']['slow_wave']['control']['CV'], d['STN']['slow_wave']['lesioned']['CV']]
    return y


def get_STN_activation_CV(d):
    y = [d['STN']['activation']['control']['CV'], d['STN']['activation']['lesioned']['CV']]
    return y



def plot_mallet2008():
    d=mallet2008()
    fig, axs=ps.get_figure(n_rows=3, 
                           n_cols=4, 
                           w=1200.0, 
                           h=400.0, 
                           fontsize=10) 
  
    i=0
    y = get_all_slow_wave_rate(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['GP control', 'GP lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,40])
    i += 1
    
    y = get_all_activation_rate(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['GP control', 'GP lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,40])
    i += 1
    
    y = get_all_slow_wave_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['GP control', 'GP lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,1.4])
    i += 1
    
    y = get_all_activation_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['GP control', 'GP lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,1.4])
    i += 1 
    
    y = get_TI_TA_slow_wave_rate(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['TI lesioned', 'TA lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,30])
    i += 1
    
    y = get_TI_TA_slow_wave_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['TI lesioned', 'TA lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,1.8])
    i += 1    

    y = get_TI_TA_activation_rate(d)
    Data_bar(**{'y':y}).bar(axs[i], alpha=0.5)
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['TI lesioned', 'TA lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,30])
    i += 1
    
    y = get_TI_TA_activation_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['TI lesioned', 'TA lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,1.8])
    i += 1     
    
    i=0
    y = get_all_slow_wave_rate(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['STN control', 'STN lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,40])
    i += 1
    
    y = get_STN_activation_rate(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Mean firing rate (Hz)')
    axs[i].set_xticklabels(['STN control', 'STN lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,40])
    i += 1
    
    y = get_STN_slow_wave_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['STN control', 'STN lesioned'])
    axs[i].set_title('Slow wave')
    axs[i].set_ylim([0,1.4])
    i += 1
    
    y = get_STN_activation_CV(d)
    Data_bar(**{'y':y}).bar(axs[i])
    axs[i].set_ylabel('Coefficient of variation')
    axs[i].set_xticklabels(['STN control', 'STN lesioned'])
    axs[i].set_title('Activation')
    axs[i].set_ylim([0,1.4])
    i += 1 
    
    return fig, axs
def add_GPe(d):
    for key in d.keys():
        if not 'GA' in d[key].keys():
            continue
        if not 'GI' in d[key].keys():
            continue
        s1=d[key]['GA']['spike_signal']
        s2=d[key]['GI']['spike_signal']
        s3=s1.merge(s2)
        d[key]['GP']={'spike_signal':s3}
        

# def cmp_statistical_test():

def get_kwargs_builder(**k_in):
    return {'print_time':False, 
            'local_num_threads':THREADS, 
            'save_conn':{'overwrite':False},
            'sim_time':10000.0, 
            'sim_stop':10000.0, 
            'size':10000.0, 
            'start_rec':0.0, 
            'sub_sampling':1}

def get_kwargs_engine():
    return {'verbose':True}

def get_networks(builder, k_bulder, k_director):
    return manager.get_networks(builder, 
                                get_kwargs_builder(**k_bulder),
                                k_director,  
                                get_kwargs_engine(),)

def create_relations(models_coher, dd):
    for key in dd.keys():
        for model in models_coher:
            k1, k2 = model.split('_')
            dd[key][model] = {}
            obj = Data_units_relation(model, dd[key][k1]['spike_signal'], 
                dd[key][k2]['spike_signal'])
            dd[key][model]['spike_signal'] = obj



def plot_letter(ax, name, c0, c1):
    ax.text(c0, c1, name, 
                horizontalalignment='center', 
                verticalalignment='center', 
                color='w', 
                transform=ax.transAxes,  
#                 fontsize=24,
                  
#                     rotation=270
                )
    
#     ax.text(c0, c1, name, 
#                 horizontalalignment='center', 
#                 verticalalignment='center', 
#                 color='w', 
#                 fontsize=22,
#                 transform=ax.transAxes)
# #                     rotation=270

def set_text_on_bars(axs, i, names, coords):
    for name, coord in zip(names, coords):
        
#         axs[i].scatter(coord[0], coord[1], color='w', edgecolor='k',
#                        linewidth=0.5,
#                    s=0.1, 
# #                    transform=axs[i].transAxes,
#                    marker=r'$'+name+'$')
        plot_letter(axs[i], name, coord[0],coord[1])
#         
#         axs[i].text(coord[0], coord[1], name, 
#                     horizontalalignment='center', 
#                     verticalalignment='center', 
#                     color='k', 
#                  transform=axs[i].transAxes,  
#                     fontsize=22,
#                       
# #                     rotation=270
#                     )
#         
#         axs[i].text(coord[0], coord[1], name, 
#                     horizontalalignment='center', 
#                     verticalalignment='center', 
#                     color='w', 
#                     fontsize=20,
#                    transform=axs[i].transAxes,
# #                     rotation=270
#                     )
#     

def plot_spk_stats_STN(d, axs, i, **k):
    y_lim_scale=1.1
    color_red=misc.make_N_colors('jet', 2)[1]
    
    leave_out=k.get('leave_out',[])
    
    y_mean, y_mean_SEM = [], []
    y_CV, y_CV_SEM = [], []
    names=['M', 'E', 'M', 'E']
    coords=[[0.1, 0.075],[0.33, 0.075],[0.67, 0.075],[ 0.9,0.075]]

    for key in sorted(d.keys()):
        v = d[key]
        for model in ['ST']:
            st = v[model]['spike_statistic']
            y_mean.append(st.rates['mean'])
            y_mean_SEM.append(st.rates['SEM'])
            y_CV.append(st.cv_isi['mean'])
            y_CV_SEM.append(st.cv_isi['SEM'])
    
    print y_mean
    print y_mean_SEM      
    
    dm=mallet2008()
    
    mode=k.get('statistics_mode', 'activation')
    if mode=='slow_wave':
        Y=[get_STN_slow_wave_rate(dm),
           get_STN_slow_wave_CV(dm)]
    if mode=='activation':
        Y=[get_STN_activation_rate(dm),
           get_STN_activation_CV(dm)]

    # *******
    # GPe FR
    # *******      
    Data_bar(**{'y':[[y_mean[0], y_mean[1]],
                     [Y[0][0],  Y[0][1]]],
                'y_std':[[y_mean_SEM[0], y_mean_SEM[1]],
                         [0.,  0.]]}).bar2(axs[i], **{'edgecolor':'k',
                                                      'top_lable_rotation':0,
                                                      'top_label_round_off':0})
          
    axs[i].set_ylabel('Rate (Hz)')
    axs[i].set_xticklabels(['Control', 'Lesion'])
    axs[i].set_title('STN')
    axs[i].set_ylim([0,40*y_lim_scale])
    
    set_text_on_bars(axs, i, names, coords)
    i += 1

    # *******
    # GPe CV
    # *******    
    Data_bar(**{'y':[[y_CV[0], y_CV[1]],
                     [Y[1][0], Y[1][1]]],
                'y_std':[[y_CV_SEM[0], y_CV_SEM[1]],
                         [0., 0.]]}).bar2(axs[i],
                                           **{'edgecolor':'k',
                                              
                                                      'top_label_round_off':1,
                                              'top_lable_rotation':0
                                              })
    
#     Data_bar(**{'y':y_CV}).bar(axs[i])
    axs[i].set_ylabel('CV')
    axs[i].set_xticklabels(['Control', 'Lesion'])
    axs[i].set_title('STN')
    axs[i].set_ylim([0, 1.9*y_lim_scale])
    set_text_on_bars(axs, i, names, coords)
    i += 1
    return i
def plot_spk_stats(d, axs, i, **k):
    y_lim_scale=1.1
    color_red=misc.make_N_colors('jet', 2)[1]
    
    leave_out=k.get('leave_out',[])
    
    y_mean, y_mean_SEM = [], []
    y_CV, y_CV_SEM = [], []
    names=['M', 'E', 'M', 'E']
    coords=[[0.1, 0.075],[0.33, 0.075],[0.67, 0.075],[ 0.9,0.075]]
    
    for key in sorted(d.keys()):
        v = d[key]
        for model in ['GP']:
            st = v[model]['spike_statistic']
            y_mean.append(st.rates['mean'])
            y_mean_SEM.append(st.rates['SEM'])
            y_CV.append(st.cv_isi['mean'])
            y_CV_SEM.append(st.cv_isi['SEM'])
    
    print y_mean
    print y_mean_SEM      
    
    dm=mallet2008()
    
    mode=k.get('statistics_mode', 'activation')
    if mode=='slow_wave':
        Y=[get_all_slow_wave_rate(dm),
           get_all_slow_wave_CV(dm),
           get_TI_TA_slow_wave_rate(dm),
           get_TI_TA_slow_wave_CV(dm)]
    if mode=='activation':
        Y=[get_all_activation_rate(dm),
           get_all_activation_CV(dm),
           get_TI_TA_activation_rate(dm),
           get_TI_TA_activation_CV(dm)]
            
    # *******
    # GPe FR
    # *******      
    Data_bar(**{'y':[[y_mean[0], y_mean[1]],
                     [Y[0][0],  Y[0][1]]],
                'y_std':[[y_mean_SEM[0], y_mean_SEM[1]],
                         [0.,  0.]]}).bar2(axs[i], **{'edgecolor':'k',
                                                      'top_lable_rotation':0,
                                                      'top_label_round_off':0})
          
    axs[i].set_ylabel('Rate (Hz)')
    axs[i].set_xticklabels(['Control', 'Lesion'])
    axs[i].set_title('GPe')
    axs[i].set_ylim([0,40*y_lim_scale])
    
    set_text_on_bars(axs, i, names, coords)
    i += 1

    # *******
    # GPe CV
    # *******    
    Data_bar(**{'y':[[y_CV[0], y_CV[1]],
                     [Y[1][0], Y[1][1]]],
                'y_std':[[y_CV_SEM[0], y_CV_SEM[1]],
                         [0., 0.]]}).bar2(axs[i],
                                           **{'edgecolor':'k',
                                              
                                                      'top_label_round_off':1,
                                              'top_lable_rotation':0
                                              })
    
#     Data_bar(**{'y':y_CV}).bar(axs[i])
    axs[i].set_ylabel('CV')
    axs[i].set_xticklabels(['Control', 'Lesion'])
    axs[i].set_title('GPe')
    axs[i].set_ylim([0, 1.9*y_lim_scale])
    set_text_on_bars(axs, i, names, coords)
    
    
    i += 1            
#     pylab.show()
    y_mean = []
    y_CV = []    
    
    for key in sorted(d.keys()):
        for model in ['GI', 'GA']:
            v = d[key]
            
            st = v[model]['spike_statistic']
            y_mean.append(st.rates['mean'])
            y_mean_SEM.append(st.rates['SEM'])
            y_CV.append(st.cv_isi['mean'])
            y_CV_SEM.append(st.cv_isi['SEM'])
        
    # *****************
    # TI/TA FR control
    # *****************   
    if 'control_fr' not in leave_out:
        Data_bar(**{'y':y_mean[0:2]}).bar(axs[i])
        axs[i].set_ylabel('Rate (Hz)')
        axs[i].set_title('Control')
        axs[i].set_xticklabels(['TI', 'TA'])
        axs[i].set_ylim([0,40*y_lim_scale])
        set_text_on_bars(axs, i, names[0::2], coords[0::2])
    
    
        i += 1
#     pylab.show()
    
    # *****************
    # TI/TA FR lesion
    # *****************   
    Data_bar(**{'y':[[y_mean[2],y_mean[3]],
                     [Y[2][0], Y[2][1]]],
                'y_std':[[y_mean_SEM[2],y_mean_SEM[3]],
                         [0., 0.]]}).bar2(axs[i],  **{'colors':[color_red,color_red], 
                                              'top_label_round_off':0,
                                                      'top_lable_rotation':0,
                                                      'hatchs':['/', 'o'], 
                                                      'edgecolor':'k'})
    axs[i].set_ylabel('Rate (Hz)')
    axs[i].set_title('Lesion')
    axs[i].set_xticklabels(['TI', 'TA'])
    axs[i].set_ylim([0,40*y_lim_scale])
    set_text_on_bars(axs, i, names, coords)   
    i += 1


    # *****************
    # TI/TA CV control
    # *****************  
    if 'control_cv' not in leave_out:
        Data_bar(**{'y':y_CV[0:2]}).bar(axs[i], **{'colors'})
        axs[i].set_ylabel('CV')
        axs[i].set_title('Control')
        axs[i].set_xticklabels(['TI', 'TA'])
        axs[i].set_ylim([0,1.9*y_lim_scale])
        set_text_on_bars(axs, i, names[0::2], coords[0::2])
        i += 1

    Data_bar(**{'y':[[y_CV[2], y_CV[3]],
                     [Y[3][0], Y[3][1]]],
                'y_std':[[y_CV_SEM[2], y_CV_SEM[3]],
                         [0., 0.]]}).bar2(axs[i],  **{'colors':[color_red,color_red], 
                                              'top_lable_rotation':0,
                                                      'top_label_round_off':1,
                                                      'hatchs':['/', 'o'], 
                                                      'edgecolor':'k'})
    axs[i].set_ylabel('CV')
    axs[i].set_title('Lesion')
    axs[i].set_xticklabels(['TI', 'TA'])
    axs[i].set_ylim([0,1.9*y_lim_scale])
    set_text_on_bars(axs, i, names, coords)
       
    i += 1
    
    
#     Data_bar(**{'y':y_CV}).bar(axs[i])
#     axs[i].set_ylabel('Coefficient of variation')
#     axs[i].set_xticklabels(['TA control', 'TA lesioned',
#                             'TI control', 'TI lesioned'])
#     i += 1
#     
    return i


def plot_activity_hist(d, axs, i):
    colors=misc.make_N_colors('jet', len( d.keys()))
    for i_key, key in enumerate(sorted(d.keys())):
        d[key]['GP']['activity_histogram'].plot(axs[i], 
                                                **{'color':colors[i_key]})
        axs[i].set_ylim([0,50])
        
        axs[i].set_title(key)
        d[key]['GP']['activity_histogram_stat'].hist(axs[i+1],
                                                     **{'color':colors[i_key]})
    
    i += 2
    return i


def translation_dic():
    return {'GP':'GPe',
            'GI':'TI',
            'GA':'TA',
            'ST':'STN'}

def plot_coherence(d, axs, i, **k):
    td=translation_dic()
#     models=k.get()
    ax = axs[i]
    colors=misc.make_N_colors('jet', len( d.keys()))
    for i_key, key in enumerate(sorted(d.keys())):
        v = d[key]
        for j, model in enumerate(['GP_GP', 'GI_GI', 'GI_GA', 'GA_GA']):
            ax = axs[i+j]
            ch = v[model]['mean_coherence']
            ch.plot(ax, **{'color':colors[i_key]})
            ax.set_xlim(k.get('xlim_cohere',[0,2]))
            ax.set_title(td[model[0:2]]+' vs '+td[model[-2:]])
            
    i+=4
    
    return i

def plot_coherence_STN(d, axs, i, **k):
    td=translation_dic()
#     models=k.get()
    ax = axs[i]
    colors=misc.make_N_colors('jet', len( d.keys()))
    for i_key, key in enumerate(sorted(d.keys())):
        v = d[key]
        for j, model in enumerate(['ST_ST', 'GA_ST', 'GI_ST', 'GA_ST']):
            ax = axs[i+j]
            ch = v[model]['mean_coherence']
            ch.plot(ax, **{'color':colors[i_key]})
            ax.set_xlim(k.get('xlim_cohere',[0,2]))
            ax.set_title(td[model[0:2]]+' vs '+td[model[-2:]])
            
    i+=4
    
    return i

def plot_phases_diff_with_cohere(d, axs, i, xmax=5, **k):
    td=translation_dic()
    models=k.get('models_pdwc', ['GP_GP', 'GI_GI', 'GI_GA', 'GA_GA', 
                                 'ST_ST', 'GP_ST', 'GA_ST', 'GI_ST',])
    colors=misc.make_N_colors('jet', len( d.keys()))
    for i_key, key in enumerate(sorted(d.keys())):
        v = d[key]
        for j, model in enumerate(models):
            ax = axs[i+j]
            ch = v[model]['phases_diff_with_cohere']
            
            ch.hist(ax, **{'color':colors[i_key], 
                           'all':k.get('all',True), 
                           'p_95':k.get('p_95',True)})
            ax.set_xlim([-numpy.pi, numpy.pi])
            ax.set_title(td[model[0:2]]+' vs '+td[model[-2:]])
    i+=8
    return i

def plot_isi(d, axs, i):
    
    colors=misc.make_N_colors('jet', len( d.keys()))
    for i_key, key in enumerate(sorted(d.keys())):
        v = d[key]
        for j, model in enumerate(['GA']):
            ax = axs[j+i]
            obj = v[model]['spike_statistic']
            m=numpy.mean(obj.isi['raw'][0])
            std=numpy.std(obj.isi['raw'][0])
            CV=std/m
            CV2=obj.cv_isi['raw'][0]
            CV3=obj.cv_isi['mean']
            l='m:{0:.4} s:{1:.4} CV:{2:.4} CV2:{3:.4} CV3:{4:.4}'.format(m,std,
                                                                         CV, 
                                                                         CV2,
                                                                         CV3)
            ax.hist(obj.isi['raw'][0], **{'label':l,
                                          'color':colors[i_key]})
            ax.legend()
#             ax.title()
            
    
    i+=1
        
   


def show_summed(d, **k):
#     import toolbox.plot_settings as ps  
    fig, axs=ps.get_figure(n_rows=5, 
                           n_cols=4, 
                           w=1400.0, 
                           h=800.0, 
                           fontsize=14,
                           frame_hight_y=0.6)   
    


    i=0
    i = plot_spk_stats(d, axs, i, **k)

    i = plot_activity_hist(d, axs, i)
    i = plot_coherence(d, axs, i, **k)
    i = plot_phases_diff_with_cohere(d, axs, i)
#     pylab.show()    

    for ax in axs:
        ax.my_set_no_ticks(yticks=5)
    return fig

def show_summed_STN(d, **k):
    kw={'n_rows':6, 
        'n_cols':16, 
        'w':72/2.54*11.6, 
        'h':175, 
        'fontsize':7,
        'frame_hight_y':0.5,
        'frame_hight_x':0.7,
        'title_fontsize':7,
        'font_size':7,
        'text_fontsize':7,
        'linewidth':1.,
        'gs_builder':gs_builder}
#     kwargs_fig=kwargs.get('kwargs_fig', kw)
    
    fig, axs=ps.get_figure2(**kw) 
       
    i=0
    i = plot_spk_stats_STN(d, axs, i, **k)
    i+=2


    i = plot_coherence_STN(d, axs, i, **k)
    i = plot_phases_diff_with_cohere(d, axs, i, **k)
 
    for ax in axs[0:4]:
        ax.my_set_no_ticks(yticks=3)
    for ax in axs[4:]:
        ax.my_set_no_ticks(yticks=2)
    
    axs[0].my_remove_axis(xaxis=True, yaxis=False)            
    axs[1].my_remove_axis(xaxis=False, yaxis=False)            
    axs[2].my_remove_axis(xaxis=True, yaxis=True)
    axs[3].my_remove_axis(xaxis=False, yaxis=True)    

    for i in range(4,7):
        axs[i].my_remove_axis(xaxis=True, yaxis=False,
                              keep_ticks=True)   
        axs[i].set_xlabel('')

    for i in range(8,11):
        axs[i].my_remove_axis(xaxis=True, yaxis=False,
                              keep_ticks=True)   
        axs[i].set_xlabel('')
    for i in range(0,12):
        axs[i].set_title('')#my_remove_axis(xaxis=False, yaxis=True)  
    
    for i in range(4,12):
        axs[i].set_ylabel('')
        
#     for i in range(4,8):
        axs[i].set_yticks([0.0, 0.5])
        
    for i in range(8,12):

        v=0
        for l in axs[i].lines:
            v=max(max(l._y),v)
 
        axs[i].set_ylim([0,v*1.1])
        
        axs[i].set_yticks([0.0, round(v*1.1/2,1)])
        axs[i].my_set_no_ticks(xticks=4)  
        
    axs[4].text(-0.45, 
                -1.1, 
                'Coherence', 
#                 fontsize=24,
                transform=axs[4].transAxes,
                verticalalignment='center', 
                rotation=90)  

    axs[4].legend(axs[4].lines[0::2], ['Control', 'Lesion'],
                   bbox_to_anchor=(2.2, 2.1), ncol=2,
#                    borderpad=0.5,
                   columnspacing=0.3,
                   handletextpad=0.1,
                    frameon=False)
    

    for i, s in zip([1],['STN']):
        font0 = FontProperties()
        font0.set_weight('bold')
        axs[i].text(0.5, 
                    -0.35, 
                    s, 
#                     fontsize=24,
                    fontproperties=font0,
                    transform=axs[i].transAxes,
                    horizontalalignment='center', 
                    rotation=0) 


    for i, s in enumerate(['ST-ST', 'ST-GP', 'ST-TI', 'ST-TA']):
        axs[i+4].text(1.08, 
                    0.5, 
                    s, 
#                     fontsize=18,
                    transform=axs[i+4].transAxes,
                    verticalalignment='center', 
                    horizontalalignment='center', 
                    rotation=270) 
        axs[i+8].text(1.08, 
                    0.5, 
                    s, 
#                     fontsize=18,
                    transform=axs[i+8].transAxes,
                    verticalalignment='center', 
                    horizontalalignment='center', 
                    rotation=270)   
    axs[8].text(-0.45, 
                -1.1,
                'Normalized count', 
#                 fontsize=24,
                transform=axs[8].transAxes,
                verticalalignment='center', 
                rotation=90)  
  
    return fig
def gs_builder(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.1 ), 
              hspace=kwargs.get('hspace', 0.1 ))

    iterator = [[slice(1,3),slice(0,3)],
                [slice(3,5),slice(0,3)],
                [slice(1,3),slice(3,6)],
                [slice(3,5),slice(3,6)],
                [slice(1,2),slice(8,11)],
                [slice(2,3),slice(8,11)],
                [slice(3,4),slice(8,11)],
                [slice(4,5),slice(8,11)],
                [slice(1,2),slice(13,16)],
                [slice(2,3),slice(13,16)],
                [slice(3,4),slice(13,16)],
                [slice(4,5),slice(13,16)],
                ]
    
    return iterator, gs, 
  
def show_summed2(d, **k):
#     import toolbox.plot_settings as ps  

    kw={'n_rows':6, 
        'n_cols':16, 
        'w':72/2.54*11.6, 
        'h':175, 
        'fontsize':7,
        'frame_hight_y':0.5,
        'frame_hight_x':0.7,
        'title_fontsize':7,
        'font_size':7,
        'text_fontsize':7,
        'linewidth':1.,
        'gs_builder':gs_builder}
#     kwargs_fig=kwargs.get('kwargs_fig', kw)
    
    fig, axs=ps.get_figure2(**kw) 
#     fig, axs=ps.get_figure(n_rows=3, 
#                            n_cols=4, 
#                            w=1000.0*0.58*2, 
#                            h=500.0*0.6*2, 
#                            fontsize=24,
#                            text_fontsize=20,
#                            font_size=20,
#                            frame_hight_y=0.55,
#                            frame_hight_x=0.6,
#                            linewidth=4.)   
    

    for ax in axs:
        ax.tick_params(direction='in',
                       length=2, top=False, right=False)  
    i=0
    i = plot_spk_stats(d, axs, i, **k)

#     i = plot_activity_hist(d, axs, i)
    i = plot_coherence(d, axs, i, **k)
    i = plot_phases_diff_with_cohere(d, axs, i, **k)
#     pylab.show()    

    for ax in axs[0:4]:
        ax.my_set_no_ticks(yticks=3)
    for ax in axs[4:]:
        ax.my_set_no_ticks(yticks=2)
    
    axs[0].my_remove_axis(xaxis=True, yaxis=False)            
    axs[1].my_remove_axis(xaxis=False, yaxis=False)            
    axs[2].my_remove_axis(xaxis=True, yaxis=True)
    axs[3].my_remove_axis(xaxis=False, yaxis=True)   
    
    for i in range(4,7):
        axs[i].my_remove_axis(xaxis=True, yaxis=False,
                              keep_ticks=True)   
        axs[i].set_xlabel('')

    for i in range(8,11):
        axs[i].my_remove_axis(xaxis=True, yaxis=False,
                              keep_ticks=True)   
        axs[i].set_xlabel('')
    for i in range(0,12):
        axs[i].set_title('')#my_remove_axis(xaxis=False, yaxis=True)  
    
    for i in range(4,12):
        axs[i].set_ylabel('')
        
#     for i in range(4,8):
        axs[i].set_yticks([0.0, 0.5])
        
    for i in range(8,12):

        v=0
        for l in axs[i].lines:
            v=max(max(l._y),v)
 
        axs[i].set_ylim([0,v*1.1])
        
        axs[i].set_yticks([0.0, round(v*1.1/2,1)])
        axs[i].my_set_no_ticks(xticks=4)  
        
    axs[4].text(-0.45, 
                -1.1, 
                'Coherence', 
#                 fontsize=24,
                transform=axs[4].transAxes,
                verticalalignment='center', 
                rotation=90)  

    axs[4].legend(axs[4].lines[0::2], ['Control', 'Lesion'],
                   bbox_to_anchor=(2.2, 2.1), ncol=2,
#                    borderpad=0.5,
                   columnspacing=0.3,
                   handletextpad=0.1,
                    frameon=False)
    

    for i, s in zip([1,3],['GPe', 'Lesion']):
        font0 = FontProperties()
        font0.set_weight('bold')
        axs[i].text(0.5, 
                    -0.35, 
                    s, 
#                     fontsize=24,
                    fontproperties=font0,
                    transform=axs[i].transAxes,
                    horizontalalignment='center', 
                    rotation=0) 

    for i, s in enumerate(['GP-GP', 'TI-TI', 'TI-TA', 'TA-TA']):
        axs[i+4].text(1.08, 
                    0.5, 
                    s, 
#                     fontsize=18,
                    transform=axs[i+4].transAxes,
                    verticalalignment='center', 
                    horizontalalignment='center', 
                    rotation=270) 
        axs[i+8].text(1.08, 
                    0.5, 
                    s, 
#                     fontsize=18,
                    transform=axs[i+8].transAxes,
                    verticalalignment='center', 
                    horizontalalignment='center', 
                    rotation=270)   
    axs[8].text(-0.45, 
                -1.1,
                'Normalized count', 
#                 fontsize=24,
                transform=axs[8].transAxes,
                verticalalignment='center', 
                rotation=90)  
    
    
    
#     plot_letter(axs[0], 'S', 0.1, 1.1)
    axs[0].text(0.1, 
                1.38,
                'M=Model', 
#                 fontsize=24,
                transform=axs[0].transAxes,
                va='center', 
#                 ha='center', 
#                 ha='center',
                rotation=0)  
    axs[0].text(0.1, 
                1.15,
                'E=Experiment, Mallet et al 2008', 
#                 fontsize=24,
                transform=axs[0].transAxes,
                va='center', 
#                 ha='center', 
#                 ha='center',
                rotation=0)  
    
    mode=k.get('statistics_mode', 'activation')
    if mode=='slow_wave':
        s='Slow wave'
    else:
        s='Activation (beta)'
#     s='Activation (beta)'
    font0 = FontProperties()
    font0.set_weight('bold')
    axs[8].text(1.25, 
            -1.,
            s, 
#             fontsize=28,
            transform=axs[8].transAxes,
            fontproperties=font0,
            va='center', 
#                 ha='center', 
#                 ha='center',
            rotation=270)  
    
        
    
    return fig

class Setup(object):
    
    def __init__(self, 
                 period, 
                 local_num_threads, **kwargs):
        self.period=period
        self.local_num_threads=local_num_threads
        self.nets_to_run=kwargs.get('nets_to_run', ['Net_0',
                                                    'Net_1' ])
        self.start_fr=kwargs.get('start_fr',10000.0)
        self.stop_fr=kwargs.get('stop_fr',20000.0)
        self.fs=kwargs.get('fs',1000.0)
  
    def builder(self):
        return {}
    
    def director(self):
        return {'nets_to_run':self.nets_to_run}  
 
    def activity_histogram(self):
        d = {'average':False,
             'period':self.period}
        
        return d



    def coherence(self):
        d = {'fs':self.fs, 'NFFT':1024 * 4, 
            'noverlap':int(1024 * 2), 
            'sample':50.,  
            'local_num_threads':self.local_num_threads}
        return d
 
    def pds(self):
        d = {'NFFT':1024 * 4, 
             'fs':1000., 
             'noverlap':1024 * 2, 
             'local_num_threads':self.local_num_threads}
        return d
    
   

    

    def phase_diff(self):
        d = {'inspect':False, 
             'lowcut':0.5, 
             'highcut':2., 
             'order':3, 
             'fs':1000.0, 
             'bin_extent':250., 
             'kernel_type':'gaussian', 
             'params':{'std_ms':125., 
                       'fs':1000.0}, 
             'local_num_threads':self.local_num_threads}
        
        return d
    
    def phases_diff_with_cohere(self):
        kwargs={
                'NFFT':1024 * 4, 
                'fs':100., 
                'noverlap':1024 * 2, 
                'sample':10.,   
                
               'lowcut':0.5, 
               'highcut':2., 
               'order':3, 

             'bin_extent':250., 
             'kernel_type':'gaussian', 
             'params':{'std_ms':125., 
                       'fs':100.0}, 
      
                'local_num_threads':self.local_num_threads}
        return kwargs
    
    def firing_rate(self):
        d={'average':False, 
           'local_num_threads':self.local_num_threads, 
           'win':100.0,}
        return d
    
    def plot_fr(self):
        d={'win':100.,
           't_start':self.start_fr,
           't_stop':self.stop_fr,
           'labels':['Control', 'Lesion'],
           
            'fig_and_axes':{'n_rows':8, 
                            'n_cols':1, 
                            'w':800.0*0.55*2*0.3, 
                            'h':600.0*0.55*2*0.3, 
                            'fontsize':7,
                            'frame_hight_y':0.8,
                            'frame_hight_x':0.78,
                            'linewidth':1.}}
        return d

    def plot_coherence(self):
        d={'xlim':[0, 10]}
        return d
    
    def plot_summed(self):
        d={'xlim_cohere':[0, 10],
           'statistics_mode':'activation'}
        return d

    def plot_summed2(self):
        d={'xlim_cohere':[0, 10],
           'all':False,
           'p_95':True,
           'leave_out':['control_fr', 'control_cv'],
           'statistics_mode':'slow_wave',
           'models_pdwc': ['GP_GP', 'GI_GI', 'GI_GA', 'GA_GA'],
           }
        return d
    
    def plot_summed_STN(self):
        d={'xlim_cohere':[0, 10],
           'all':False,
           'p_95':True,
           'leave_out':['control_fr', 'control_cv'],
           'statistics_mode':'slow_wave',
           'models_pdwc': ['ST_ST', 'GP_ST', 'GI_ST', 'GA_ST'],
           }
        return d
    
def simulate(builder=Builder,
             from_disk=0,
             perturbation_list=None,
             script_name=__file__.split('/')[-1][0:-3],
             setup=Setup(1000.0, THREADS)):
    
    k = get_kwargs_builder()

    d_pds = setup.pds()
    d_cohere = setup.coherence()
    d_phase_diff = setup.phase_diff()
    d_phases_diff_with_cohere = setup.phases_diff_with_cohere()
    d_firing_rate = setup.firing_rate()
    d_activity_hist=setup.activity_histogram()
    
    attr = [
        'firing_rate', 
        'mean_rates', 
        'spike_statistic'
        ]
    
    attr2=['psd',
           'activity_histogram',
           'activity_histogram_stat']
    
    attr_coher = [
                  'phase_diff', 
                  'phases_diff_with_cohere',
                  'mean_coherence'
                 ]
    
    kwargs_dic = {'firing_rate':d_firing_rate, 
                  'mean_rates':{'t_start':k['start_rec'] + 1000.}, 
                  'mean_coherence':d_cohere, 
                  'phase_diff':d_phase_diff, 
                  'phases_diff_with_cohere':d_phases_diff_with_cohere,
                  'spike_statistic':{'t_start':k['start_rec'] + 1000.}}
    
        
    models = ['M1', 'M2', 'FS', 'GI', 'GA', 'ST', 'SN', 'GP']
    models_coher = ['GI_GA', 'GI_GI', 'GA_GA', 'GA_ST', 'GI_ST', 'GP_GP',
                     'ST_ST', 'GP_ST',]
    

    
    info, nets, _ = get_networks(builder, 
                                 setup.builder(), 
                                 setup.director())
    add_perturbations(perturbation_list, nets)
    for p in sorted(perturbation_list.list):
        print p
#     print nets['Net_0'].par['nest']['M1_M1_gaba']['weight']

    key=nets.keys()[0]
    file_name = get_file_name(script_name, nets[key].par)
    file_name_figs = get_file_name_figs(script_name,  nets[key].par)
    path_nest=get_path_nest(script_name, nets.keys(), nets[key].par)

    print nets.keys()

    for net in nets.values():
        net.set_path_nest(path_nest)
    
#     sd = get_storage(file_name, info)

    # Adding nets no file name
    sd_list=get_storage_list(nets, file_name, info)
    
    d = {}
        
    from_disks = [from_disk] * len(nets.keys())
    
    if type(sd_list)==list:
        iterator=[nets.values(), from_disks, sd_list]
    else:
        iterator=[nets.values(), from_disks]
    
    for vals in zip(*iterator):
        if type(sd_list)==list:
            net, fd, sd=vals
        else:
            net, fd=vals
               
        if fd == 0:
            dd = run(net)
            add_GPe(dd)
#             dd = compute(dd, models, attr, **kwargs_dic)
            save(sd, dd)
        
        elif fd == 1:
  
            filt = [net.get_name()] + models + ['spike_signal']+ attr
            dd = load(sd, *filt)    
            dd = compute(dd, models, attr, **kwargs_dic)
 
            save(sd, dd)
            for keys, val in misc.dict_iter(dd):
                if keys[-1]=='spike_signal':
                    val.wrap.allowed.append('get_phases_diff_with_cohere')
#             
            print 'create_relations'
            create_relations(models_coher, dd)
            dd = compute(dd, models_coher, attr_coher, **kwargs_dic)
            
            cmp_psd(d_pds, models, dd) 
            cmp_activity_hist(models, dd, **d_activity_hist )
            cmp_activity_hist_stat(models, dd, **{'average':False})          
            
            save(sd, dd)
            
        elif fd == 2:
            filt = ([net.get_name()] 
                    + models + models_coher 
                    + attr + attr2 + attr_coher)
            dd = load(sd, *filt)

            #             cmp_statistical_test(models, dd)
        d = misc.dict_update(d, dd)
    
    return file_name_figs, from_disks, d, models, models_coher


def cmp_activity_hist(models, d, **kwargs):
    for key1 in d.keys():
        for model in models:
            obj=d[key1][model]['firing_rate'].get_activity_histogram(**kwargs)
            d[key1][model]['activity_histogram'] = obj

def cmp_activity_hist_stat(models, d, **kwargs):       

    for key1 in d.keys():
        for model in models:
            v=d[key1][model]['activity_histogram']
            obj=v.get_activity_histogram_stat(**kwargs)
            d[key1][model]['activity_histogram_stat'] = obj
            
            
def create_figs(file_name_figs, from_disks, d, models, models_coher, setup):

    d_plot_fr=setup.plot_fr()
    d_plot_coherence=setup.plot_coherence()
    d_plot_summed=setup.plot_summed()
    d_plot_summed2=setup.plot_summed2()
    d_plot_summed_STN=setup.plot_summed_STN()

    sd_figs = Storage_dic.load(file_name_figs)
    if numpy.all(numpy.array(from_disks) == 2):
        figs = []
        fig, axs=ps.get_figure(**d_plot_fr['fig_and_axes'])
        figs.append(fig)
        show_fr(d, models, axs, **d_plot_fr)
#         axs=figs[-1].get_axes()
        ps.shift('upp', axs, 0.08, n_rows=len(axs), n_cols=1)
        ps.shift('left', axs, 0.25, n_rows=len(axs), n_cols=1)
        for i, ax in enumerate(axs):      
            
            ax.my_set_no_ticks(xticks=5, yticks=2)
            ax.legend(bbox_to_anchor=[1.45,1.15])
            
            ax.set_ylabel('Hz')
            if i==7:
                pass
            else:
                ax.my_remove_axis(xaxis=True)
                ax.set_xlabel('')
        y_mean=[]
        for net in ['Net_0', 'Net_1']:
            st = d[net]['ST']['spike_statistic']
            y_mean.append(st.rates['mean'])
        axs[6].text(0.5,0.1,'C:{:.2f} Hz L:{:.2f} Hz'.format(*y_mean), 
                    transform=axs[5].transAxes,
                    verticalalignment='center', 
                    horizontalalignment='center')
        
#         figs.append(show_hr(d, models))
#         figs.append(show_psd(d, models=models))
#         figs.append(show_coherence(d, models=models_coher, **d_plot_coherence))
#         figs.append(show_phase_diff(d, models=models_coher))
        
#         figs.append(show_summed(d, **d_plot_summed))

        
        figs.append(show_summed2(d, **d_plot_summed2))
        figs.append(show_summed_STN(d, **d_plot_summed_STN))
#         axs=figs[-1].get_axes()
# #         ps.shift('right', axs, 0.25, n_rows=len(axs), n_cols=1)
#         axs[4].legend(axs[4].lines[0::2],['Control', 'Lesion'])
#         axs[8].legend(axs[8].lines,['Control', 'Lesion'], loc='lower center')
        
#         for ax in axs[8:]:
#             ax.my_set_no_ticks(xticks=6)
# 
#         for ax in axs[4:]:
#             ax.my_set_no_ticks(yticks=3)
        
        sd_figs.save_figs(figs, format='png', dpi=200)
        sd_figs.save_figs(figs[1:], format='svg', in_folder='svg')

def main(*args, **kwargs):
    
    v=simulate(*args, **kwargs)
    file_name_figs, from_disks, d, models, models_coher = v

    
    setup=kwargs['setup']
    create_figs(file_name_figs, from_disks, d, models, models_coher, setup)
    
#     if DISPLAY: pylab.show()  
    
    return d


def run_simulation(from_disk=0, local_num_threads=12, type_of_run='shared_memory'):
    from toolbox.network.default_params import Perturbation_list as pl
    from simulate import  get_path_rate_runs, pert_add_oscillations
    import oscillation_perturbations4 as op

    local_num_threads=12
    sim_time=10000.0
    size=500.0
    sub_sampling=25

    
    kwargs={
            'amp_base':[1],
            'freqs':[1.5],
            'freq_oscillation':20.,
            'local_num_threads':local_num_threads,
            'path_rate_runs':get_path_rate_runs('simulate_inhibition_ZZZ4/'),
            'perturbation_list':[op.get()[7]],
            'sim_time':sim_time,
            'size':size,
            }
    
    p=pert_add_oscillations(**kwargs)
        
    p_add=pl({'netw':{'size':size,
                  'sub_sampling':{'M1':sub_sampling,
                                  'M2':sub_sampling}}},
              '=')
    p+=p_add

    setup=Setup(1000.0, THREADS, **{'start_fr':0.0, 
                                     'stop_fr':10000.0})
    v=simulate(builder=Builder,
                from_disk=from_disk,
                perturbation_list=p,
                script_name=(__file__.split('/')[-1][0:-3]
                             +'/script_'+p.name+'_'+type_of_run),
                setup=setup)
    
    
    
    file_name_figs, from_disks, d, models, models_coher = v
    return d, file_name_figs, from_disks, models, models_coher, setup 
  

import unittest

class TetsOscillationRun(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_run(self):
        v=run_simulation(from_disk=2,local_num_threads=12)
        d, file_name_figs, from_disks, models, models_coher, setup=v        
#         pp(d)
        dd={}
        for key in sorted(d.keys()):
            net=d[key]
            for model in sorted(net.keys()):
                if 'firing_rate' in net[model].keys():
                    m=numpy.mean(net[model]['firing_rate'].y)
#                     print key, model, numpy.mean(net[model]['firing_rate'].y)
                    misc.dict_update(dd, {key:{model:m}})
#         pp(dd)
        data={'Net_0': {'FS': 17.348,
                       'GA': 14.53,
                       'GI': 36.478,
                       'GP': 32.180,
                       'M1': 0.254,
                       'M2': 0.065,
                       'SN': 28.835,
                       'ST': 11.729},
             'Net_1': {'FS': 14.282,
                       'GA': 12.570,
                       'GI': 29.638,
                       'GP': 26.294,
                       'M1': 0.047,
                       'M2': 1.069,
                       'SN': 32.307,
                       'ST': 13.816}}
        for key in dd.keys():
            for model in dd[key].keys():
                self.assertAlmostEqual(dd[key][model],
                                       data[key][model],
                                        1)
            

class TestOcsillation(unittest.TestCase):     
    def setUp(self):
        
        v=run_simulation(from_disk=2, 
                         local_num_threads=12)
        d, file_name_figs, from_disks, models, models_coher, setup=v
        
        
        
        self.setup=setup
        self.file_name_figs=file_name_figs
        self.from_disks=from_disks
        self.d=d
        self.models=models
        self.models_coher=models_coher  
        

        

    def test_isi(self):
#         import toolbox.plot_settings as ps
        fig, axs=ps.get_figure(n_rows=1, n_cols=1, 
                               w=1000.0, h=800.0, fontsize=10)  
           
        plot_isi(self.d, axs, 0)
#         pylab.show()
        print self.d

             
    def test_plot_mallet2008(self):
        plot_mallet2008()
        pylab.show()
         
    def testPlotStats(self):
#         import toolbox.plot_settings as ps
        fig, axs=ps.get_figure(n_rows=3, 
                           n_cols=2, 
                           w=1000.0, h=800.0, fontsize=10)  
           
        plot_spk_stats(self.d, axs, 0, **{'statistics_mode': 'slow_wave'})
#         pylab.show()
        print self.d

    def testPlotActivityHist(self):
        import toolbox.plot_settings as ps
        fig, axs=ps.get_figure(n_rows=2, 
                           n_cols=2, 
                           w=1000.0, h=800.0, fontsize=10)  
         
        plot_activity_hist(self.d, axs, 0)
#         plot_spk_stats(self.d, axs, 0)
        pylab.show()
        print self.d

    def testPlotCoherence(self):
        import toolbox.plot_settings as ps
        fig, axs=ps.get_figure(n_rows=1, 
                           n_cols=1, 
                           w=1000.0, h=800.0, fontsize=10)  
         
        plot_coherence(self.d, axs, 0)
#         plot_spk_stats(self.d, axs, 0)
        pylab.show()
        print self.d
 
    def testPlotPhasesDffWithCohere(self):
        import toolbox.plot_settings as ps
        fig, axs=ps.get_figure(n_rows=3, 
                           n_cols=2, 
                           w=1000.0, h=800.0, fontsize=10)  
          
        plot_phases_diff_with_cohere(self.d, axs, 0)
#         plot_spk_stats(self.d, axs, 0)
        pylab.show()
        print self.d   
         
          
    def testShowSummed(self):
        show_summed(self.d, **{'xlim_cohere':[0,7], 
                               'statistics_mode':'slow_wave'})
        pylab.show()
              
    def testShowSummed2(self):
        d={'xlim_cohere':[0, 10],
           'leave_out':['control_fr', 'control_cv'],
           'statistics_mode':'slow_wave',
           'models_pdwc': ['GP_GP', 'GI_GI', 
                           'GI_GA', 'GA_GA']}
        show_summed2(self.d, **d)
        pylab.show()

    def testShowSummed_STN(self):
        d={'xlim_cohere':[0, 10],
           'leave_out':['control_fr', 'control_cv'],
           'statistics_mode':'slow_wave',
           'models_pdwc': ['ST_ST', 'GP_ST', 
                           'GI_ST', 'GA_ST']}
        show_summed_STN(self.d, **d)
        pylab.show()

    def test_create_figs(self):
        create_figs(self.file_name_figs, 
                    self.from_disks, 
                    self.d, 
                    self.models, 
                    self.models_coher,
                    self.setup)
        pylab.show()
     
    def test_show_fr(self):
        d={'n_rows':8, 
            'n_cols':1, 
            'w':800.0*0.55*2, 
            'h':600.0*0.55*2, 
            'fontsize':11*2,
            'frame_hight_y':0.8,
            'frame_hight_x':0.78,
            'linewidth':3.}
        
        fig, axs=ps.get_figure(**d) 
        print len(axs)
        show_fr(self.d, self.models, axs, **{
                                        'win':20.,
                                        't_start':4000.0,
                                        't_stop':5000.0})
        pylab.show()
  
    def test_show_coherence(self):
        show_coherence(self.d, self.models_coher, **{'xlim':[0,10]})
        pylab.show()
 
 
    def test_show_phase_diff(self):
        show_phase_diff(self.d, self.models_coher)
        pylab.show()


class TestOscillationMPI(unittest.TestCase):
    
    def setUp(self):
        
        import subprocess
        from toolbox.data_to_disk import pickle_save, pickle_load        
        from toolbox.network import default_params
        
        s = default_params.HOME
        data_path= s+'/results/unittest/oscillation_common/run_simulation/'
        script_name=(os.getcwd()+'/test_scripts_MPI/'
                     +'oscillation_common_run_simulation_mpi.py')

        from_disk=2

        fileName=data_path+'data_in.pkl'
        fileOut=data_path+'data_out.pkl'
        
        np=12

        pickle_save([from_disk, np], fileName)

        p=subprocess.Popen(['mpirun', '-np', str(np), 'python', 
                            script_name, fileName, fileOut],
#                            stdout=subprocess.PIPE,
#                            stderr=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           )
         
        out, err = p.communicate()
#         print out
#         print err
        v=pickle_load(fileOut)
        d, file_name_figs, from_disks, models, models_coher, setup=v

        self.setuo=setup
        self.file_name_figs=file_name_figs
        self.from_disks=from_disks
        self.d=d
        self.models=models
        self.models_coher=models_coher     

    def test_run_simulation(self):
        pass
    
        

if __name__ == '__main__':
    d={
       TetsOscillationRun:[
#                               'test_run',
                           ],
       
        TestOcsillation:[
                      
                        'test_create_figs',
#                         'test_isi',
#                         'test_plot_mallet2008',
#                         'testPlotStats',
#                         'testPlotActivityHist',
#                         'testPlotCoherence',
#                         'testPlotPhasesDffWithCohere',
#                         'testShowSummed',
#                         'testShowSummed2',
#                         'testShowSummed_STN',  
#                         'test_show_fr',
#                         'test_show_coherence',
#                         'test_show_phase_diff',
                        ],
       TestOscillationMPI:[
#                         'test_run_simulation',
                           ]}


    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
    





    