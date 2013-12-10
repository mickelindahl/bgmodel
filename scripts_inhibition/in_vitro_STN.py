'''
Created on Aug 23, 2013

@author: lindahlm
'''
import numpy
import pylab
from toolbox.network_construction import Single_units_in_vitro 
from in_vitro_base import In_vitro
from toolbox import plot_settings

class In_vitro_STN(In_vitro):
    
    def show(self, labels):
        colors=['g','b', 'r','m']
        coords=[[0.05, 0.9-i*0.1] for i in range(len(colors))] 
        linestyles=['-', '-', '-', '--']

        fig, ax_list=plot_settings.get_figure( n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=12)
                
        self.plot_IV(ax_list[0], [labels[0]], colors, coords, linestyles)
        self.plot_IF(ax_list[1], [labels[0]], colors, coords, linestyles)
        self.plot_IF_var(ax_list[2], labels[1:3], colors, coords, linestyles)
        self.plot_IF_var(ax_list[3], [labels[-1]], colors, coords, linestyles)
        
        fig.savefig( self.path_pictures  + self.sname  + '.svg', format = 'svg') 
            

labels=['ST-dop',
        'ST-dop-C_m', 'ST-dop-V_t',
        'ST-dop-C_m_V_t']
tata_dop=[0.8,0.8,0.8,0.8]

inv=In_vitro_STN(Single_units_in_vitro, labels, tata_dop)


inv.simulate_IV(1, numpy.arange(-200, 0,30), [labels[0]], 5000.0)
inv.simulate_IF(1, numpy.arange(0, 300,10), [labels[0]], 5000.0)   
inv.simulate_IF_variation(1, numpy.arange(0, 300,30), [labels[1]], 500.0, 10, ['C_m'])
inv.simulate_IF_variation(1, numpy.arange(0, 300,30), [labels[2]], 500.0, 10, ['V_th'])
inv.simulate_IF_variation(1, numpy.arange(0, 300,30), [labels[3]], 500.0, 10, ['C_m', 'V_th'])
inv.show(labels)
pylab.show()
