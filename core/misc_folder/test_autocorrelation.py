
import numpy
import pylab
import os
import sys


if len(sys.argv) != 1: mpiRun = True
else:                  mpiRun = False

sys.path.append(os.getcwd())    # Add current directory to python path                                                
current_path=os.getcwd()

# First add parent directory to python path
model_dir=   '/'.join(os.getcwd().split('/')[0:-1]) +'/model'       
code_dir=  '/'.join(os.getcwd().split('/')[0:-1]) 
picture_dir=  '/'.join(os.getcwd().split('/')[0:-3]) + '/pictures'     
                
sys.path.append(model_dir) 
sys.path.append(code_dir+'/nest_toolbox') 
spath  = os.getcwd()+'/output/' + sys.argv[0].split('/')[-1].split('.')[0]

# Then import model and network
from model_params import models
from src import misc, my_nest, my_signals, plot_settings
from src.my_population import MyGroup  
from src.my_axes import MyAxes 

# Simulate or use stored data
LOAD=False

neuronModels=['SNR_aeif']
synapseModels=['MSN_SNR_gaba_s_min', 'MSN_SNR_gaba_s_max',
               'MSN_SNR_gaba_p0', 'MSN_SNR_gaba_p1','MSN_SNR_gaba_p2']



def plot_example_msn(ax, MSN):
    ax_twinx=ax.my_twinx()
    MSN.signals['spikes'].raster_plot(display=ax_twinx,kwargs={'color':'k',
                                                               'zorder':1})  
    ax_twinx.set_ylabel('Neuron id')
    time_bin=500
    MSN.signals['spikes'].my_firing_rate( bin=time_bin, display=ax,
                                          kwargs={'color':'k',
                                                  'linewidth':3,})
    ax.my_set_no_ticks( yticks=4, xticks = 4 ) 
    ax_twinx.my_set_no_ticks( yticks=4, xticks = 4 ) 
    ax.set_title('bin=%i'%(time_bin),**{'fontsize':12})
    ax.set_ylabel('Frequency MSNs (Hz)')
    
def plot_example_snr(ax, SNR):
    time_bin=500
    
    colors = misc.make_N_colors('Blues', 5)
    colors=['g','r', colors[1], 'b', colors[3]]   
    
    signal=SNR.signals['spikes']
    signal.my_firing_rate(id_list=[SNR[0]], bin=time_bin, display=ax,
                          kwargs={'color':colors[0]})
    signal.my_firing_rate(id_list=[SNR[1]], bin=time_bin, display=ax,
                          kwargs={'color':colors[1]})
    signal.my_firing_rate(id_list=[SNR[3]], bin=time_bin, display=ax,
                          kwargs={'color':colors[3]})
    
    ax.my_set_no_ticks( yticks=4, xticks = 4 ) 
    ax.set_ylim([0,40])
    ax.text( 6500, 33, 'Ref 1' , **{ 'color' : colors[0] })  
    ax.text( 4000, 7, 'Ref 2' , **{ 'color' : colors[1] }) 
    ax.text( 6500, 24, 'Dyn MSN' , **{ 'color' : colors[3] }) 
    
    ax.set_title('bin=%i'%(time_bin),**{'fontsize':12})
    ax.set_ylabel('Frequency SNr (Hz)') 

def simulate_example_msn_snr():  
    nFun=0  # Function number
    nSim=0  # Simulation number within function
    
    rates=numpy.array([.1,.1])
    times=numpy.array([0.,25000.])
    nMSN =500
    simTime=100000.
    I_e=0.
    
    my_nest.ResetKernel()
    model_list=models()
    my_nest.MyLoadModels( model_list, neuronModels )
    my_nest.MyLoadModels( model_list, synapseModels )

    MSN = MyGroup( 'spike_generator', nMSN, mm_dt=1.0, mm=False, sd=False,
                   spath=spath, 
                   sname_nb=str(nFun)+str(nSim))  
    SNR = MyGroup( neuronModels[0], n=len(synapseModels), params={'I_e':I_e},
                   sd=True,
                   mm_dt = .1, mm=False, spath=spath, 
                   sname_nb=str(nFun)+str(nSim) )
    nSim+=1

    spikeTimes=[]
    for i in range(nMSN):
        spikes=misc.inh_poisson_spikes( rates, times,                        
                                    t_stop=simTime, 
                                    n_rep=1, seed=i )
        my_nest.SetStatus([MSN[i]], params={ 'spike_times':spikes } ) 
        for spk in spikes: spikeTimes.append((i,spk))   
    # add spike list for MSN to MSN spike list
    MSN.signals['spikes'] = my_signals.MySpikeList(spikeTimes, MSN.ids)     
    MSN.save_signal( 's') 
   
    noise=my_nest.Create('noise_generator', params={'std':100.})
    
    my_nest.Connect(noise,[SNR[0]],params={'receptor_type':5})
    my_nest.Connect(noise,[SNR[1]],params={'receptor_type':5})
    my_nest.Connect(noise,[SNR[2]],params={'receptor_type':5})
    
    for i, syn in enumerate(synapseModels):
        my_nest.ConvergentConnect(MSN[:],[SNR[i]], model=syn)
        
    my_nest.MySimulate( simTime )
    SNR.get_signal( 's' ) # retrieve signal




    
    SNR_rates=[SNR.signals['spikes'].mean_rates(0,5000), 
               SNR.signals['spikes'].mean_rates(5000, 10000)]     
    for i in range(0, len(SNR_rates)):      
        for j in range(0, len(SNR_rates[0])):
            SNR_rates[i][j]=int(SNR_rates[i][j])
    s='\n'
    s =s + 'Example plot MSN and SNr:\n' 
    s =s + 'Synapse models:\n'
    for syn in synapseModels:
        s = s + ' %s\n' % (syn )    
    s = s + ' %s %5s %3s \n' % ( 'N MSN:', str ( nMSN ),  '#' )    
    s = s + ' %s %5s %3s \n' % ( 'MSN Rates:',   str ( [str(round(r,1)) 
                                                        for r in rates]),'Hz' )     
    s = s + ' %s %5s %3s \n' % ( '\nSNR Rates 0-5000:\n',   
                                 str ( SNR_rates [0]) ,'Hz' )   
    s = s + ' %s %5s %3s \n' % ( '\nSNR Rates 10000-5000:\n',  
                                  str ( SNR_rates [1]) ,'Hz' )   
    s = s + ' %s %5s %3s \n' % ( '\nTimes:', str ( times), 'ms' )
    s = s + ' %s %5s %3s \n' % ( 'I_e:', str ( I_e ), 'pA' )
    
    infoString=s
 
    return MSN, SNR, infoString
 
 # SIMULATION
infoString=''

# simulate_example_msn_snr
MSN, SNR, s = simulate_example_msn_snr()
infoString=infoString+s   
 
 # DISPLAY
plot_settings.set_mode(mode='by_fontsize', w = 1100.0, h = 450.0, fontsize=12)
font_size_text = 8
fig = pylab.figure( facecolor = 'w' )

ax_list = []
ax_list.append( MyAxes(fig, [ .075, .37, .135, .26 ] ) )    # text box
ax_list.append( MyAxes(fig, [ .26,  .6,  .165, .34 ] ) )    # 
ax_list.append( MyAxes(fig, [ .53, .6,  .165, .34 ] ) )    # 
ax_list.append( MyAxes(fig, [ .8,   .6,  .165, .34 ] ) )    # 
ax_list.append( MyAxes(fig, [ .26,  .1,  .165, .34 ] ) )    # 
#ax_list.append( MyAxes(fig, [ .53, .1,  .165, .34 ] ) )    # 
#ax_list.append( MyAxes(fig, [ .8,   .1,  .165, .34 ] ) )    #   


# Example msn
ax=ax_list[1]
plot_example_msn(ax, MSN)

# Example snr
ax=ax_list[2]
plot_example_snr(ax, SNR)


ax=ax_list[3]
spikes=SNR.signals['spikes'].spiketrains[SNR.ids[0]].spike_times
hist_autocorr, xaxis=misc.autocorrelation(spikes, bin=1, max_time=500)
ax.plot(xaxis, hist_autocorr)

spikes=SNR.signals['spikes'].spiketrains[SNR.ids[1]].spike_times
hist_autocorr, xaxis=misc.autocorrelation(spikes, bin=1, max_time=500)
ax.plot(xaxis, hist_autocorr)

spikes=SNR.signals['spikes'].spiketrains[SNR.ids[2]].spike_times
hist_autocorr, xaxis=misc.autocorrelation(spikes, bin=1, max_time=500)
ax.plot(xaxis, hist_autocorr)



ax=ax_list[4]
spikes=MSN.signals['spikes'].spiketrains[MSN.ids[0]].spike_times
hist_autocorr, xaxis=misc.autocorrelation(spikes, bin=1, max_time=200)
ax.plot(xaxis, hist_autocorr)

pylab.show()
    