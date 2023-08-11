'''
Created on Sep 11, 2014

@author: mikael
'''


import pylab
import random
import numpy
from scripts_inhibition.base_oscillation_beta import Setup
from core.network import default_params
from core import my_nest, my_population
from core import misc
from core.my_population import MyNetworkNode
from core.my_signals import DataGeneric, DataIFCurve, DataScatter
import pprint
pp=pprint.pprint

from core.network.manager import get_storage_list, save, load
from core import directories as dr
from core import data_to_disk
import os

path=dr.HOME_DATA+'/'+__file__.split('/')[-1][0:-3]    
if not os.path.isdir(path):
    data_to_disk.mkdir(path)
par=default_params.Inhibition()
setup=Setup(50,20) 

def gs_builder(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',1)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.1 ), 
              hspace=kwargs.get('hspace', 0.1 ))

    iterator = [[slice(0,1),slice(0,1)],
                [slice(0,1),slice(1,2)],
                [slice(1,2),slice(0,1)],
                [slice(1,2),slice(1,2)],
                [slice(2,3),slice(0,1)],
                [slice(2,3),slice(1,2)],
                [slice(3,4),slice(0,2)],
                ]
    
    return iterator, gs,     


def get_fig_axs():
    scale=4
    kw={'n_rows':4, 
        'n_cols':2, 
        'w':72/2.54*11*scale, 
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
    from core import plot_settings as ps
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
    dg=DataGeneric(**{'x':x, 'y':y, 'xlabel':'Current (pA)', 'ylabel':'Voltage (mV)'})

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
    
    return  {'IF':DataIFCurve(**d)}


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
    
    my_nest.SetStatus(mnn[:], params={'I_e':.5}) # Set I_e
    I_e = my_nest.GetStatus(mnn.ids,'I_e')[0]    
    
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
        dg=DataGeneric(**{'x':x, 'y':voltage, 'xlabel':'Time (ms)', 'ylabel':'Voltage (mV)'})
        misc.dict_update(d, {'rs_voltage_{0}'.format(i):dg})
    rd=mnn.spike_signal.raw_data()
    dg=DataScatter(**{'x':rd[:,0], 'y':rd[:,1], 'xlabel':'Time (ms)', 'ylabel':'Voltage (mV)'})
    misc.dict_update(d, {'rs_scatter':dg})
    
    return d


def get_nullcline(**kw):
    
    p=kw.get('nc_params').copy()
    p.update({'V':kw.get('nc_V')})
    x,y=my_population.get_nullcline_aeif(**p)

    dg=DataGeneric(**{'x':x, 'y':y, 'ylabel':'Current (pA)', 'xlabel':'Voltage (mV)'})

    return {'nullcline':dg} 

if __name__=='__main__':
    
    d_st_list0=[
#                 {},
#                
#              {'a_1':1.,
#               'a_2':0.,
#               'b':0.25,
#               'Delta_T':8.4,
#               'E_L':-58.4,
#               'V_a':-58.4,
#               'V_th':-50.0,
#               'g_L':5.,
#               'V_reset_max_slope1':-50. },
              
            {'a_1':1.,
            'a_2':.0,
            'b':0.25,
            'Delta_T':16.4,
            'E_L':-80.,
            'V_a':-53.,
            'V_th':-50.0,
            'g_L':10.,
            'V_reset_max_slope1':-50. },

              {'a_1':1.,
              'a_2':0.,
              'b':0.25,
              'Delta_T':16.4,
              'E_L':-80.2,
              'V_a':-70.,
              'V_th':-50.0,
              'g_L':10.,
              'V_reset_max_slope1':-50. },
                                
#              {'a_1':1.,
#               'a_2':1.,
#               'b':0.25,
#               'Delta_T':8.4,
#               'E_L':-58.4,
#               'V_a':-58.4,
#               'V_th':-50.0,
#               'g_L':5.,
#               'V_reset_max_slope1':-50. },
# 
#                
#             {'a_1':.7,
#             'a_2':0.,
#             'b':0.15,
#             'Delta_T':5.6,
#             'E_L':-55.6,
#             'V_a':-55.6,
#             'V_th':-50.0,
#             'g_L':5.,
#             'V_reset_max_slope1':-50. },
#                 
#              {'a_1':1.,
#             'a_2':0.,
#             'b':0.25,
#             'Delta_T':2.8,
#             'E_L':-52.8,
#             'V_a':-52.8,
#             'V_th':-50.0,
#             'g_L':5.,
#             'V_reset_max_slope1':-50. },
             ]


    d_st0=par.dic['nest']['ST']
    d_st_list=[]
    for d in d_st_list0: 
        d_st0=par.dic['nest']['ST'].copy()
        d_st0.update(d)
        d_st_list.append(d_st0)


    from_disk=0
    kw_list=[]

    for p_st in d_st_list:
        kw={
            'rs_curr':[-40, -70.,-100],
            'rs_time':[300., 300., 300],
            'rs_params':p_st,
            
            'I_E':0.0, 
            
            'if_I_vec':numpy.arange(-99, 301,10),
            'if_params':p_st,
            
            'iv_params':p_st,
            'iv_I_vec': numpy.arange(-200,0,10),
            
            'model':'my_aeif_cond_exp',
            
            'nc_params':p_st,
            'nc_V':numpy.arange(-80,-0,1)}
        
            
        kw_list.append(kw)
    
    
    nets=['Net_{0:0>2}'.format(i) for i in range(len(kw_list))]
    sd_list=get_storage_list(nets, path, '')
    
    d={}
    for net, sd, kw in zip(nets, sd_list, kw_list):
    
        if from_disk==0:# and net=='Net_05':
            print net
            dd={}
            for call in [
                        simulate_IV, 
                        simulate_IF,
                            get_nullcline,
                           simulate_rebound_spike,
                         ]:
                misc.dict_update(dd,{net:call(**kw)})

            save(sd, dd)
        
#         if from_disk==1:
#             filt= [net]+['gi','st', 'gi_st']+['spike_signal']
#             dd = load(sd, *filt)
#             pp(dd)
#             dd=compute_attrs(dd, net)
#             save(sd, dd)
            
        elif from_disk==1:
            filt = [net]+['IV', 'IF', 'nullcline'] +['rs_scatter']
            dd = load(sd, *filt)
        
        d = misc.dict_update(d, dd)
    
    pp(d)
    fig, axs=get_fig_axs()
    
    colors=misc.make_N_colors('jet', len(nets))
    for i in range(len(nets)):
        dtmp=d['Net_0{0}'.format(i)]
        if 'IV' not in dtmp.keys(): continue
        dtmp['IV'].plot(axs[0], **{'color':colors[i]})

 
        dtmp=d['Net_0{0}'.format(i)]
        if 'IV' not in dtmp.keys(): continue
        dtmp['IF'].plot(axs[1], part='first',**{'color':colors[i],'linestyle':'-'})
        dtmp['IF'].plot(axs[1], part='mean', **{'color':colors[i],'linestyle':'--'})


        dtmp=d['Net_0{0}'.format(i)]
        if 'nullcline' not in dtmp.keys(): continue
        dtmp['nullcline'].plot(axs[2],**{'color':colors[i]})
        axs[2].set_ylim([-100,600])


        dtmp=d['Net_0{0}'.format(i)]
        if 'rs_scatter' not in dtmp.keys(): continue
        dtmp['rs_scatter'].scatter(axs[3], **{'color':colors[i], 'id0':i*3})
#         axs[2].set_ylim([-100,600])



        dtmp=d['Net_0{0}'.format(i)]
        if 'rs_0' not in dtmp.keys(): continue
        dtmp['rs_voltage_0'].plot(axs[4],**{'color':colors[i]})
#         axs[2].set_ylim([-100,600])


        dtmp=d['Net_0{0}'.format(i)]
        if 'rs_1' not in dtmp.keys(): continue
        dtmp['rs_voltage_1'].plot(axs[5],**{'color':colors[i]})
    

        dtmp=d['Net_0{0}'.format(i)]
        if 'rs_2' not in dtmp.keys(): continue
        dtmp['rs_voltage_2'].plot(axs[6],**{'color':colors[i]})
#         axs[2].set_ylim([-100,600])

#         axs[i].set_xlim(1000.0,1500.0)
 
#     colors=misc.make_N_colors('copper', len(kw_list))
#     
#     
#     for net, c in zip(nets, colors):
#         kw={ 'all':True,
#            'color':c,
#            'p_95':False,}
#         d[net]['gi_st']['phases_diff_with_cohere'].hist(axs[-1], **kw)
#     axs[-1].set_xlim(-numpy.pi, numpy.pi)
 
    pylab.show() 
    
    
    
    
    # pylab.plot(d['n'])
    