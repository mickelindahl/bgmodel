#! Imports
import numpy
import pylab
from toolbox.network.construction import Single_units_in_vitro 
from in_vitro_base import In_vitro
from toolbox import plot_settings

class In_vitro_MSN(In_vitro):
    pass

    def show(self, labels):
        colors=['g','b', 'r','m']
        coords=[[0.05, 0.9-i*0.1] for i in range(len(colors))] 
        linestyles=['-', '-', '-', '--']

        fig, ax_list=plot_settings.get_figure(n_rows=2, n_cols=2, w=1000.0, 
                                              h=800.0, fontsize=12)
                
        self.plot_IV(ax_list[0], labels[0:4], colors, coords, linestyles)
        self.plot_IF(ax_list[1], labels[0:4], colors, coords, linestyles)
        self.plot_IF_var(ax_list[2], labels[4:6], colors, coords, linestyles)
        self.plot_IF_var(ax_list[3], [labels[-1]], colors, coords, linestyles)
        
        fig.savefig( self.path_pictures  + self.sname  + '.svg', format = 'svg') 
        
        
labels=['M1-dop','M2-dop', 'M1-no_dop','M2-no_dop', 
        'M1-dop-C_m', 'M1-dop-V_t', 'M1-dop-C_m_V_t']
tata_dop=[0.8, 0.8, 0.0, 0.0, 
          0.8,0.8,0.8]
inv=In_vitro_MSN(Single_units_in_vitro, labels, tata_dop)


inv.simulate_IV(1, numpy.arange(-20, 200,10), labels[0:4], 500.0)
inv.simulate_IF(0, numpy.arange(150, 300,20), labels[0:4], 5000.0)   
inv.simulate_IF_variation(0, numpy.arange(210, 300,20), [labels[4]], 5000.0, 10, ['C_m'])
inv.simulate_IF_variation(0, numpy.arange(210, 300,20), [labels[5]], 5000.0, 10, ['V_th'])
inv.simulate_IF_variation(0, numpy.arange(210, 300,20), [labels[6]], 5000.0, 10, ['C_m', 'V_th'])
inv.show(labels)
pylab.show()


