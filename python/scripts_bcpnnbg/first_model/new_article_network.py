import numpy
import pylab
import os
import sys
import time as ttime
import pprint
import nest
# Get directory where model and code resides 
model_dir=   '/'.join(os.getcwd().split('/')[0:-1])    
code_dir=  '/'.join(os.getcwd().split('/')[0:-2])  
dev_dir=  '/'.join(os.getcwd().split('/')[0:-3])  
indata_dir=dev_dir+'/indata/'
# Add model, code and current directories to python path
sys.path.append(os.getcwd())  
sys.path.append(model_dir)
sys.path.append(code_dir+'/nest_toolbox') 


from src import my_nest, misc, my_topology, plot_settings
from src.my_axes import MyAxes 
import nest.topology as tp
from simulation_utils from scripts_inhibition.simulate_network


model_name=os.getcwd().split('/')[-2]
picture_dir='/'.join(os.getcwd().split('/')[0:-3]) + '/pictures/'+model_name 
SNAME  = sys.argv[0].split('/')[-1].split('.')[0]

OUTPUT_PATH  = os.getcwd()+'/output/' + sys.argv[0].split('/')[-1].split('.')[0]
ADJUST_XDATA_MS=500.
def plot_example_firing_rate(ax, groups, name,  color='b', ylim=[], ylabel=True, xlabel=False):
    time_bin=20


    colors=misc.make_N_colors('gist_rainbow', len(groups))
    #signal.my_firing_rate(bin=time_bin, display=ax,
    #                      kwargs={'color':color})


    
    for group, col in zip(groups, colors):
        signal=group.signals['spikes']
        print name, 'CV:', numpy.mean(signal.cv_isi())
        print name, 'mean:', signal.mean_rate()
        print name, 'std:', signal.mean_rate_std()
        hist=signal.spike_histogram(time_bin=1, normalized=True)
        spk_mean=numpy.mean(hist, axis=0)
        spk_mean=misc.convolve(spk_mean, 100, 'triangle',single=True)[0]
        print spk_mean.shape
        time=numpy.arange(1,len(spk_mean)+1)
        ax.plot(time,spk_mean, **{'color':col})
    

    ax.set_ylabel('Firing rate '+name+'\n (spikes/s)', fontsize=12.,multialignment='center') 
    ax.set_xlabel('Time (ms)')    
    
    ax.my_set_no_ticks( yticks=6, xticks = 8 ) 
    ax.set_ylim(ylim)
    ax.set_xlim([time[0],time[-1]])
    
    ax.my_remove_axis( xaxis=not xlabel, yaxis=not ylabel ) 
def plot_example_raster(ax, groups, name, ylabel=True, xlabel=False):
    global ADJUST_XDATA_MS
    

    colors=misc.make_N_colors('gist_rainbow', len(groups))
    for group, col in zip(groups, colors):
        group.signals['spikes'].raster_plot(display=ax,
                                      kwargs={'color':col, 'zorder':1})  
 
    lines = ax.lines
    ax.set_ylabel(name+' id')
    ax.my_set_no_ticks( yticks=6, xticks = 5 )
    ax.set_ylim([groups[0].ids[0],groups[-1].ids[-1]])
    
    
    for line in lines:
        line.set_xdata(line.get_xdata()-ADJUST_XDATA_MS)
    
    ax.set_xlim([0.,10500])
    
    #ax.set_xlim(misc.adjust_limit([0,1500]))   
    #if not ylabel: ax.set_ylabel('') 
    #if not xlabel: ax.set_xlabel('')
    ax.my_remove_axis( xaxis=not xlabel, yaxis=not ylabel )    
def plot_text(ax, info_string=''):
    

    tb = ''     
    tb = tb + info_string
    
    tb = tb + '\n'

    ax.text( 0.85, 0.5, tb , fontsize= font_size_text,
             horizontalalignment='right',
             verticalalignment='center',
             transform=ax.transAxes,     # to define coordinates in right scale
             **{ 'fontname' : 'monospace' })                           
    
    ax.my_remove_axis( xaxis=True, yaxis=True )
    ax.my_remove_spine(left=True,  bottom=True, right=True, top=True)    

base_rate=0.1
activation=[]
activation.append(numpy.array(numpy.genfromtxt(indata_dir+'activation_value_d1.csv', delimiter=';',dtype='float', names=False)))
activation.append(numpy.array(numpy.genfromtxt(indata_dir+'activation_value_d2.csv', delimiter=';',dtype='float', names=False)))

n_actions=activation[0].shape[1]
n_states=activation[0].shape[0]
for i in range(len(activation)):
    l=[]
    activation[i]=activation[i]*50.
    activation[i]=activation[i].transpose()
    for j, row in enumerate(activation[i]):
        l.append([])
        for r in row:
            l[j].extend([base_rate, r])
        l[j].append(0.1)
    

    activation[i]=numpy.array(l)
    
burst_time=500
inbetween_time=500;    
init_time=1000.
mod_times=[1]

for i in range(n_states):
    mod_times.extend([init_time+500*i+i*inbetween_time,init_time+500*i+i*inbetween_time+burst_time])
    
    

print mod_times
params_msn_d1={'base_rates':[0.1], 'base_times':[1], 'mod_rates': activation[0],
            'mod_times':mod_times}    
params_msn_d2={'base_rates':[0.1], 'base_times':[1], 'mod_rates': activation[1],
            'mod_times':mod_times}
params_stn={'rate':350., 'mod':False,'mod_rate':400., 'mod_times':[1000., 1000.+500.]} 

sim_time=mod_times[-1]+500.
N_MSN=1500
#synapse_models=['MSN_SNR_gaba_p1', 'GPE_SNR_gaba_p_stoc']
synapse_models=['MSN_SNR_gaba_p1', 'GPE_SNR_gaba_s_ref']
model_params={'misc':{'n_actions':{'n':n_actions}}}

save_result_at=OUTPUT_PATH+'/simulate_network.plk'
if 0:
    groups_dic=simulate_network(params_msn_d1, params_msn_d2, params_stn,
                           synapse_models, sim_time=sim_time, seed=1,
                           I_e_add={'SNR':300, 'STN':0,'GPE':30}, threads=4, 
                           start_rec=500.,model_params=model_params)    
    misc.pickle_save(groups_dic, save_result_at)  
else:
    groups_dic=misc.pickle_load(save_result_at)  
 
 #Inspect results
plot_settings.set_mode(pylab, mode='by_fontsize', w = 1100.0, h = 450.0+175.0, fontsize=12)
font_size_text = 8
fig = pylab.figure( facecolor = 'w' )
ax_list = []

ax_list.append( MyAxes(fig, [ .26-0.21,  .85-0.05, 0.2 + .165+0.05, .2 -0.05+0.02] ) )    # 
ax_list.append( MyAxes(fig, [ .80-.25,   .85-0.05, 0.2 + .165+0.05, .2-0.05 +0.02] ) )    # 
ax_list.append( MyAxes(fig, [ .26-0.21,  .65-0.04, 0.2 + .165+0.05, .2 -0.05+0.02] ) )    # 
ax_list.append( MyAxes(fig, [ .80-.25,   .65-0.04, 0.2 + .165+0.05, .2 -0.05+0.02] ) )    # 
ax_list.append( MyAxes(fig, [ .26-0.21,  .45-0.03, 0.2 + .165+0.05, .2 -0.05+0.02] ) )    # 
ax_list.append( MyAxes(fig, [ .80-.25,   .45-0.03, 0.2 + .165+0.05, .2 -0.05+0.02] ) )    # 
ax_list.append( MyAxes(fig, [ .26-0.21,  .25-0.02, 0.2 + .165+0.05, .2 -0.05+0.02] ) )    # 
ax_list.append( MyAxes(fig, [ .80-.25,   .25-0.02, 0.2 + .165+0.05, .2-0.05 +0.02] ) )    # 
ax_list.append( MyAxes(fig, [ .26-0.21,  .06, 0.2 + .165+0.05, .2 -0.05+0.02] ) )    # 
ax_list.append( MyAxes(fig, [ .80-.25,   .06, 0.2 + .165+0.05, .2-0.05+0.02 ] ) )    # 

ax=ax_list[0]
plot_example_raster(ax, groups_dic['MSN_D1'], '$MSN_{D1}$')
ax=ax_list[1]
plot_example_firing_rate(ax, groups_dic['MSN_D1'], '$MSN_{D1}$',ylim=[0,35])

ax=ax_list[2]
plot_example_raster(ax, groups_dic['MSN_D2'], '$MSN_{D2}$',)
ax=ax_list[3]
plot_example_firing_rate(ax, groups_dic['MSN_D2'], '$MSN_{D2}$', color='b', ylim=[0,35])

ax=ax_list[4]
plot_example_raster(ax, groups_dic['GPE'], 'GPe')
ax=ax_list[5]
plot_example_firing_rate(ax, groups_dic['GPE'], 'GPe',ylim=[0,70])
ax=ax_list[6]
plot_example_raster(ax, groups_dic['STN'], 'STN')

ax=ax_list[7]
plot_example_firing_rate(ax, groups_dic['STN'], 'STN', ylim=[0,35])
ax=ax_list[8]
plot_example_raster(ax, groups_dic['SNR'], 'SNr', xlabel=True)
ax=ax_list[9]
plot_example_firing_rate(ax, groups_dic['SNR'], 'SNr',ylim=[0,70], xlabel=True)

ax=ax_list[0]
#plot_text(ax, info_string=s)
pylab.show()

# dpi does not matter since svg and pdf are both vectorbased
fig.savefig( picture_dir + '/' + SNAME  + '.svg', format = 'svg') 
fig.savefig( picture_dir + '/' + SNAME  + '.pdf', format = 'pdf')