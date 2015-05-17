'''
Created on Sep 11, 2014

@author: mikael
'''


import pylab
import numpy
# from simulate_beta import Setup
from toolbox.network import default_params
from toolbox import my_nest, my_population
from toolbox import misc
from toolbox.my_population import MyNetworkNode
from toolbox.my_signals import Data_generic, Data_IF_curve, Data_scatter
import pprint
pp=pprint.pprint

from toolbox.network.manager import get_storage_list, save, load
from toolbox import directories as dr
from toolbox import data_to_disk
from toolbox.data_to_disk import Storage_dic
import os

path=dr.HOME_DATA+'/'+__file__.split('/')[-1][0:-3]    
if not os.path.isdir(path):
    data_to_disk.mkdir(path)
par=default_params.Inhibition()


class Setup(object):

    def __init__(self, period, local_num_threads, **k):
        self.fs=256.
        self.local_num_threads=local_num_threads
        self.period=period


def gs_builder(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',1)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.8 ), 
              hspace=kwargs.get('hspace', 0.1 ))

    iterator = [[slice(0,2),slice(0,2)],
                [slice(0,2),slice(2,4)],
                [slice(3,5),slice(0,2)],
                [slice(3,5),slice(2,4)],
                [slice(6,7),slice(0,2)],
                [slice(7,8),slice(0,2)],
                [slice(8,9),slice(0,2)],
                [slice(6,7),slice(2,4)],
                [slice(7,8),slice(2,4)],
                [slice(8,9),slice(2,4)],
                [slice(0,2),slice(4,6)],
                [slice(6,7),slice(4,6)],
                [slice(7,8),slice(4,6)],
                [slice(8,9),slice(4,6)],
                ]
    
    return iterator, gs,     


def get_fig_axs(scale=4):
    
    kw={'n_rows':9, 
        'n_cols':6, 
        'w':72/2.54*18*scale, 
        'h':300*scale, 
        'fontsize':7*scale,
        'frame_hight_y':0.5,
        'frame_hight_x':0.7,
        'title_fontsize':7*scale,
        'font_size':7*scale,
        'text_fontsize':7*scale,
        'linewidth':1.*scale,
        'gs_builder':gs_builder}
#     kwargs_fig=kwargs.get('kwargs_fig', kw)
    from toolbox import plot_settings as ps
    fig, axs=ps.get_figure2(**kw) 
    return fig, axs


def simulate_IV(**kw):

    I_vec=kw.get('iv_I_vec')
    my_nest.ResetKernel({'local_num_threads':1})

    sd={'active':True,
        'params':{'to_memory':True,'to_file':False, 'start':500.0 }}
    mm={'active':True,
        'params':{'interval':0.1,'to_memory':True,'to_file':False}}
    p=kw.get('iv_params')
    if 'type_id' in p.keys(): del p['type_id']
    mnn=MyNetworkNode('dummy',model=kw.get('model'), n=1, params=p, mm=mm, sd=sd)
    
    I_e0=my_nest.GetStatus(mnn[:])[0]['I_e']
    my_nest.SetStatus(mnn[:], params={'I_e':I_e0+kw.get('I_E')}) # Set I_e 

    x,y=mnn.run_IV_I_clamp(I_vec)
    print x,y
    dg=Data_generic(**{'x':x, 'y':y, 'xlabel':'Current (pA)', 'ylabel':'Voltage (mV)'})

    return {'IV':dg}  

def simulate_IF(**kw):
    
    I_vec_in=kw.get('if_I_vec')

    tStim = 700+1300
    my_nest.ResetKernel({'local_num_threads':1})

    sd={'active':True,
        'params':{'to_memory':True,'to_file':False, 'start':500.0 }}
#     mm={'active':True,
#         'params':{'interval':0.1,'to_memory':True,'to_file':False}}
    p=kw.get('if_params')
    if 'type_id' in p.keys(): del p['type_id']
    mnn=MyNetworkNode('dummy',model=kw.get('model'), n=1, params=p, sd=sd)

    I_e0=my_nest.GetStatus(mnn[:])[0]['I_e']
    my_nest.SetStatus(mnn[:], params={'I_e':I_e0+kw.get('I_E')}) # Set I_e 

    I_vec_out, fIsi, mIsi, lIsi = mnn.run_IF(I_vec_in, tStim=tStim)   
    
    d={'x':I_vec_out,'first':1000./fIsi,'mean':1000./mIsi,'last':1000./lIsi,}
    
    return  {'IF':Data_IF_curve(**d)}


def simulate_rebound_spike(**kw):
    
    n=len(kw.get('rs_curr'))
    
    simTime  = 3000.  # ms
    my_nest.ResetKernel({'local_num_threads':1})

    sd={'active':True, 'params':{'to_memory':True,'to_file':False}}
    mm={'active':True,
        'params':{'interval':0.1,'to_memory':True,'to_file':False}}
    p=kw.get('rs_params')
    
    if 'type_id' in p.keys(): del p['type_id']
    mnn=MyNetworkNode('dummy',model=kw.get('model'), n=n, params=p, mm=mm, sd=sd)

    my_nest.SetStatus(mnn[:], params={'I_e':kw.get('rs_I_e')}) # Set I_e
    
#     my_nest.SetStatus(mnn[:], params={'I_e':.5}) # Set I_e
#     I_e = my_nest.GetStatus(mnn.ids,'I_e')[0]    
    
    scg = my_nest.Create( 'step_current_generator',n=n )  
    rec=my_nest.GetStatus(mnn[:])[0]['receptor_types']
    
    i=0
    for t, c in zip(kw.get('rs_time'), kw.get('rs_curr')):
        my_nest.SetStatus([scg[i]], {'amplitude_times':[500.,t+500.],
                                'amplitude_values':[float(c),0.]})
        my_nest.Connect( [scg[i]], [mnn[i]],  params = { 'receptor_type' : rec['CURR'] } )
        i+=1
    
    my_nest.MySimulate(simTime)
    mnn.voltage_signal.my_set_spike_peak( 21, spkSignal= mnn.spike_signal )
    
    d={}
    for i in range(n):
        voltage=mnn.voltage_signal.analog_signals[i+1].signal
        x=numpy.linspace(0,simTime, len(voltage))
        dg=Data_generic(**{'x':x, 'y':voltage, 'xlabel':'Time (ms)', 'ylabel':'Voltage (mV)'})
        misc.dict_update(d, {'rs_voltage_{0}'.format(i):dg})
    rd=mnn.spike_signal.raw_data()
    dg=Data_scatter(**{'x':rd[:,0], 'y':rd[:,1], 'xlabel':'Time (ms)', 'ylabel':'Voltage (mV)'})
    misc.dict_update(d, {'rs_scatter':dg})
    
    return d

def simulate_ahp(**kw):
    
    n=len(kw.get('ahp_curr'))
    I_vec=kw.get('ahp_curr')
    
    simTime  = 3000.  # ms
    my_nest.ResetKernel({'local_num_threads':1})

    sd={'active':True, 'params':{'to_memory':True,'to_file':False}}
    mm={'active':True,'params':{'interval':0.1,'to_memory':True,'to_file':False}}
    p=kw.get('rs_params')
    
    if 'type_id' in p.keys(): del p['type_id']
    mnn=MyNetworkNode('dummy',model=kw.get('model'), n=n, params=p, mm=mm, sd=sd)

    my_nest.SetStatus(mnn[:], params={'I_e':kw.get('ahp_I_e')}) # Set I_e
    
    scg = my_nest.Create( 'step_current_generator',n=n )  
    rec=my_nest.GetStatus(mnn[:])[0]['receptor_types']
    
    for source, target, I in zip(scg, mnn[:], I_vec):
        my_nest.SetStatus([source], {'amplitude_times':[500.,1000.],
                                     'amplitude_values':[float(I),0.]})
        my_nest.Connect( [source], [target], 
                         params = { 'receptor_type' : rec['CURR'] } )
    
    
    my_nest.MySimulate(simTime)

    signal= mnn.spike_signal.time_slice(700,3000)
    
    delays=[]
    for i in range(n):
#         print signal.spiketrains[i+1.0].spike_times
        delays.append(max(numpy.diff(signal.spiketrains[i+1.0].spike_times)));
    
    dg=Data_generic(**{'x':I_vec, 'y':delays, 'xlabel':'Time (ms)', 'ylabel':'Voltage (mV)'})
    
    return {'ahp':dg}

def simulate_irregular_firing(**kw):
    
    n=len(kw.get('irf_curr'))
    I_vec=kw.get('irf_curr')
    
    simTime  = 2000.  # ms
    my_nest.ResetKernel({'local_num_threads':1})

    sd={'active':True, 'params':{'to_memory':True,'to_file':False}}
    mm={'active':True,'params':{'interval':0.1,'to_memory':True,'to_file':False}}
    p=kw.get('rs_params')
    
    if 'type_id' in p.keys(): del p['type_id']
    mnn=MyNetworkNode('dummy',model=kw.get('model'), n=n, params=p, mm=mm, sd=sd)
    
    I_e0 = my_nest.GetStatus(mnn.ids,'I_e')[0]    

    for i, I_e in enumerate(I_vec):
        my_nest.SetStatus([mnn[i]], params={'I_e':I_e+I_e0})
    
    scg = my_nest.Create( 'step_current_generator',n=n )  
    noise=my_nest.Create('noise_generator', params={'mean':0.,'std':10.})
    rec=my_nest.GetStatus(mnn[:])[0]['receptor_types']
    
    for source, target, I in zip(scg, mnn[:], I_vec):
        my_nest.SetStatus([source], {'amplitude_times':[1., simTime],
                                     'amplitude_values':[-5.,float(I)]})
        my_nest.Connect( [source], [target], 
                         params = { 'receptor_type' : rec['CURR'] } )
        my_nest.Connect( noise, [target], 
                         params = { 'receptor_type' : rec['CURR'] } )

    my_nest.MySimulate(simTime) 
    mnn.voltage_signal.my_set_spike_peak( 21, spkSignal= mnn.spike_signal )

    d={}
    for i in range(n):
        voltage=mnn.voltage_signal.analog_signals[i+1].signal
        x=numpy.linspace(0,simTime, len(voltage))
        dg=Data_generic(**{'x':x, 'y':voltage, 'xlabel':'Time (ms)', 'ylabel':'Voltage (mV)'})
        misc.dict_update(d, {'irf_voltage_{0}'.format(i):dg})
            
#     my_nest.MySimulate(simTime)
#     mnn.get_signal( 'v','V_m', stop=simTime ) # retrieve signal
#     mnn.get_signal( 's') # retrieve signal
#     mnn.signals['V_m'].my_set_spike_peak( 15, spkSignal= mnn.signals['spikes'] )

    return d


def get_nullcline(**kw):
    
    p=kw.get('nc_params').copy()
    p.update({'V':kw.get('nc_V')})
    x,y=my_population.get_nullcline_aeif(**p)

    dg=Data_generic(**{'x':x, 'y':y, 'ylabel':'Current (pA)', 'xlabel':'Voltage (mV)'})

    return {'nullcline':dg} 

def simulate(from_disk=0,
             kw={},
             net='Net_0',
             script_name=__file__.split('/')[-1][0:-3],
             setup=Setup(50,20),
            ):
    
    file_name = dr.HOME_DATA+'/'+script_name
    file_name_figs = dr.HOME_DATA+'/fig/'+script_name
    
    sd=get_storage_list([net], file_name, '')[0]
    
    d={}
    
    if from_disk==0:
        print net
        dd={}
        for call in kw.get('calls',[simulate_irregular_firing, 
                                    simulate_ahp,
                                    simulate_IV, 
                                    simulate_IF,
                                    get_nullcline,
                                    simulate_rebound_spike,]):
            misc.dict_update(dd,{net:call(**kw)})
        
        save(sd, dd)        
        
    elif from_disk==1:
        filt = kw.get('filter', [net]+['IV', 'IF', 'ahp', 'nullcline'] 
                      +['rs_scatter']+['rs_voltage_{0}'.format(i) for i in range(6)]
                      +['irf_voltage_{0}'.format(i) for i in range(3)])
        dd = load(sd, *filt)
        
    d = misc.dict_update(d, dd)    
    
    return d, file_name_figs, net
    
def create_figs(d, file_name_figs, net, **kw):
    
    fig, axs=get_fig_axs(scale=kw.get('scale',3))
    figs=[fig]
    
    color='b'

    d=d[net]
    ax=axs[0]
    d['IV'].plot(ax, **{'color':color})
    d['IV'].show_speed_point(ax, **{'at':-40, 'color':'k',
                                    'markersize':10*kw.get('scale',3), 
                                    'speed_unit':r'M$\Omega$',
                                    'unit_scale':1000.})
    ax.set_title('IV curve')

    ax=axs[1]
    d['IF'].plot(ax, part='first',**{'color':color,'linestyle':'-'})
    d['IF'].plot(ax, part='last', **{'color':color,'linestyle':'--'})
    d['IF'].show_speed_point(ax, **{'at_current':81, 'color':'k',
                                    'markersize':10*kw.get('scale',3), 'part':'last'})
    ax.set_title('IF curve')
 
    ax=axs[2]
    d['nullcline'].plot(ax,**{'color':color})
    ax.set_ylim([-100,600])
    ax.set_title('Nullcline')

    if 'rs_scatter' in d.keys():d['rs_scatter'].scatter(axs[3], **{'color':color, 'id0':0})
    for i in range(6):
        name='rs_voltage_{0}'.format(i)
        if not name in d.keys():
            continue
        d[name].plot(axs[4+i],**{'color':color})
        axs[4+i].set_xlim([0,2500])

    ax=axs[10]
    d['ahp'].plot(ax, **{'color':color})
    ax.set_title('Spike induced ahp')
    
    
    for i in range(3):
        name='irf_voltage_{0}'.format(i)
        if not name in d.keys():
            continue
        d[name].plot(axs[11+i],**{'color':color})
#         axs[4+i].set_xlim([0,2500])
    
    for ax in axs: ax.my_set_no_ticks(xticks=3, yticks=4)

    for i in [4,5,7,8]:
        axs[i].my_remove_axis(xaxis=True)
        
    for i in [4,6,7,9]:
        axs[i].set_ylabel('')
    
    for i in range(4,10):
        axs[i].set_yticks([-40, -80])    
    
    sd_figs = Storage_dic.load(file_name_figs)
    sd_figs.save_figs(figs, format='png', dpi=200)
    sd_figs.save_figs(figs[1:], format='svg', in_folder='svg')
 
def main(*args, **kwargs):
    
    args=simulate(*args, **kwargs)
    create_figs(*args,**kwargs)

    return d

def run_simulation(from_disk=0, local_num_threads=10):
    
    
    p_st={}
#     {
#           'a_1':.7,
#           'a_2':0.,
#           'b':0.15,
#           'Delta_T':5.6,
#           'E_L':-55.6,
#           'V_a':-55.6,
#           'V_th':-50.0,
#           'g_L':5.,
#           'V_reset_max_slope1':-50. }

    d=par.dic['nest']['ST']
#     pp(d)
    d.update(p_st)

    kw={
        'ahp_curr':numpy.arange(0, 350,20), 
        'ahp_I_e':.5,
        
        'I_E':0.0, 
        
        'if_I_vec':numpy.arange(-99, 301,10),
        'if_params':d,
        
        'irf_curr':[0,10,40],
        
        'iv_params':d,
        'iv_I_vec': numpy.arange(-200,0,10),
        
        'model':'my_aeif_cond_exp',
        
        'nc_params':d,
        'nc_V':numpy.arange(-80,-30,1),
        
        'rs_curr':[-70, -70, -70, -40, -70.,-100, ],
        'rs_time':[300., 450., 600., 300., 300., 300],
        'rs_params':d,
        'rs_I_e':.5,
        
        }
        
            
    args=simulate(from_disk=from_disk,
                   kw=kw,
                   net='Net_0',
                   script_name=__file__.split('/')[-1][0:-3],
                   setup=Setup(50,20) )
    
    return args
  
import unittest
class TestGP_STN_0csillation(unittest.TestCase):     
    def setUp(self):
        
        v=run_simulation(from_disk=0, local_num_threads=1)
        d, file_name_figs, net=v
                
        self.d=d
        self.file_name_figs=file_name_figs
        self.net=net
        
    def test_create_figs(self):
        create_figs(self.d,
                    self.file_name_figs, 
                    self.net,
                    **{'scale':4} )
        pylab.show()
     
    
if __name__ == '__main__':
    d={
       TestGP_STN_0csillation:[
                        'test_create_figs',
                        ],}

    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)    
    
    
    
