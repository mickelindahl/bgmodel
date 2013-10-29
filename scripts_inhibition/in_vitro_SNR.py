'''
Created on Aug 14, 2013

@author: lindahlm
'''

#! Imports
#! Imports
import numpy
import pylab
from network_classes import Single_units_in_vitro 
from in_vitro_base import In_vitro
from toolbox import plot_settings, data_handling

class In_vitro_SNR(In_vitro):      

    def __init__(self, Use_class, labels, dop, sname='', **kwargs):
        super( In_vitro_SNR, self ).__init__( Use_class, labels, dop, sname='', **kwargs)       
        
        if 'p_model' in kwargs.keys():
            self.p_model=dict(zip(self.labels, kwargs['p_model']))
    
    def simulate_IF_SNR(self, load, currents, labels, tStim):
        save_at=self.path_data+self.sname+'/'+'IF'
        if not load:
            for label in labels:
                variable={'p_model':self.p_model[label]}
                self.kwargs[label]['model_params_in'].update({'variable': variable})
                suiv=self.Use_class(1,  0., float('inf'), **self.kwargs[label])
                I_vec, fIsi, mIsi, lIsi = suiv.IF_curve(currents, tStim)   
                self.data_IF[label]=[I_vec, lIsi]
            data_handling.pickle_save(self.data_IF, save_at)
        else:
            self.data_IF=data_handling.pickle_load(save_at)
            
    def update_p_model(self, p_model):
        for label in labels:
            variable={'p_model':p_model}
            self.kwargs[label]['model_params_in'].update({'variable': variable})

    def show_SNR(self, labels):
        colors=['g','b', 'r','m','c','k']
        coords=[[ 0.05, p] for p in numpy.linspace(0.4,0.9, len(colors))]
        linestyles=['-', '-', '-', '-', '-', '-']

        fig, ax_list=self.get_figure(2, 2)
        
                
        self.plot_voltage_responses(ax_list[0], [labels[0]], [colors[0]], [coords[0]], [linestyles[0]])
        self.plot_IF(ax_list[1], labels, colors, coords, linestyles)
        
        fig.savefig( self.path_pictures  + self.sname  + '.svg', format = 'svg') 
            
    def show(self, labels):
        colors=['g','b', 'r','m']
        coords=[[ 0.05, 0.9], [0.05, 0.75], [0.05, 0.6], [0.05, 0.45]]
        linestyles=['-', '-', '-', '--']

        fig, ax_list=plot_settings.get_figure( n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=12)
                
        self.plot_IV(ax_list[0], labels[0:2], colors, coords, linestyles)
        self.plot_IF(ax_list[1], labels[0:2], colors, coords, linestyles)
        self.plot_IF_var(ax_list[2], labels[2:4], colors, coords, linestyles)
        self.plot_IF_var(ax_list[3], [labels[4]], colors, coords, linestyles)
        fig.savefig( self.path_pictures  + self.sname  + '.svg', format = 'svg') 

# Investigate what happends when E_L becomes bigger than V_th   
# Conclusion: Model behaves good        
#n=6        
#p_var_E_L=numpy.linspace(0.9,1.1,n)
#p_var=[[1,1,p] for p in p_var_E_L]
#labels=['SNR_A-dop'+str(i) for i in range(n)]
#tata_dop=[0.8]*n
#dopamine=[True]*n
#inv=In_vitro_SNR(Single_units_in_vitro, labels, tata_dop, dopamine, **{'p_model':p_var})
#
#inv.update_p_model([1.,1.,1.])
#inv.simulate_voltage_responses(1, [0.,-100.,0.], [1.,400.,600.], 0, 1000.0, [labels[0]])
#inv.simulate_IF(0, numpy.arange(-50, 50,10), labels, 5000.0)   


labels=['SNR-dop','SNR-no_dop', 
        'SNR-dop-C_m', 'SNR-dop-V_t', 'SNR-dop-C_m_V_t']
tata_dop=[0.8,0.0, 
          0.8,0.8,0.8]

inv=In_vitro_SNR(Single_units_in_vitro, labels, tata_dop)


inv.simulate_IV(1, numpy.arange(-200, 0,30), labels[0:2], 5000.0)
inv.simulate_IF(0, numpy.arange(-50, 100,10), labels[0:2], 5000.0)   
inv.simulate_IF_variation(1, numpy.arange(-50, 100,10), [labels[2]], 1000.0, 10, ['C_m'])
inv.simulate_IF_variation(1, numpy.arange(-50, 100,10), [labels[3]], 1000.0, 10, ['V_th'])
inv.simulate_IF_variation(1, numpy.arange(-50, 100,10), [labels[4]], 1000.0, 10, ['C_m','V_th'])
inv.show(labels)
pylab.show()

