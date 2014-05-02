'''
Created on Aug 14, 2013

@author: lindahlm
'''

#! Imports
#! Imports
import numpy
import pylab
from toolbox.network.engine import Single_units_in_vitro 
from in_vitro_base import In_vitro
from toolbox import plot_settings, data_to_disk

class In_vitro_GPE(In_vitro):      

    def __init__(self, Use_class, labels, dop,  sname='', **kwargs):
        super( In_vitro_GPE, self ).__init__( Use_class, labels, dop, sname='', **kwargs)       
        
        if 'p_model' in kwargs.keys():
            self.p_model=dict(zip(self.labels, kwargs['p_model']))
        

    def show_GPE(self, labels):
        colors=['g','b', 'r','m','c','k']
        coords=[[ 0.05, p] for p in numpy.linspace(0.4,0.9, len(colors))]
        linestyles=['-', '-', '-', '-', '-', '-']

        fig, ax_list=self.get_figure(2, 2)
        
                
        self.plot_voltage_responses(ax_list[0], [labels[0]], [colors[0]], [coords[0]], [linestyles[0]])
        self.plot_IF(ax_list[1], labels, colors, coords, linestyles)
        
        fig.savefig( self.path_pictures  + self.sname  + '.svg', format = 'svg') 
            
    def show(self, labels):
        colors=['g','b', 'r','m']
        coords=[[0.05, 0.9-i*0.1] for i in range(len(colors))] 
        linestyles=['-', '-', '-', '--']

        fig, ax_list=plot_settings.get_figure( n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=12)
                
        self.plot_IV(ax_list[0], labels[0:2], colors, coords, linestyles)
        self.plot_IF(ax_list[1], labels[0:2], colors, coords, linestyles)
        self.plot_IF_var(ax_list[2], labels[2:4], colors, coords, linestyles)
        self.plot_IF_var(ax_list[3], [labels[-1]], colors, coords, linestyles)
        fig.savefig( self.path_pictures  + self.sname  + '.svg', format = 'svg') 

# Investigate what happends when E_L becomes bigger than V_th        
# Conclusion: Model behaves good        
#n=6        
#p_var_E_L=numpy.linspace(0.9,1.1,n)
#p_var=[[1,1,p] for p in p_var_E_L]
#labels=['GPE_A-dop'+str(i) for i in range(n)]
#tata_dop=[0.8]*n
#dopamine=[True]*n
#inv=In_vitro_GPE(Single_units_in_vitro, labels, tata_dop, dopamine, **{'p_model':p_var})
#
#inv.update_p_model([1.,1.,1.])
#inv.simulate_voltage_responses(1, [0.,-100.,0.], [1.,400.,600.], 0, 1000.0, [labels[0]])
#inv.simulate_IF(0, numpy.arange(-50, 50,10), labels, 5000.0)   


labels=['GA-dop','GA-no_dop', 
        'GA-dop-C_m', 'GA-dop-V_t',
        'GA-dop-C_m_V_t']
tata_dop=[0.8,0.0, 
          0.8,0.8, 0.8]

inv=In_vitro_GPE(Single_units_in_vitro, labels, tata_dop)


inv.simulate_IV(1, numpy.arange(-200, 0,30), labels[0:2], 500.0)
inv.simulate_IF(0, numpy.arange(-10, 10,0.5), labels[0:2], 500.0)   
inv.simulate_IF_variation(0, numpy.arange(-50, 300,30), [labels[2]], 500.0, 10, ['C_m'])
inv.simulate_IF_variation(0, numpy.arange(-50, 300,30), [labels[3]], 500.0, 10, ['V_th'])
inv.simulate_IF_variation(0, numpy.arange(-50, 300,30), [labels[4]], 500.0, 10, ['C_m', 'V_th'])
inv.show(labels)
pylab.show()

