#! Imports
import numpy
import pylab 
from toolbox.network_construction import Single_units_in_vitro 
from toolbox.network_handling_single_units import In_vitro
        
        
def main():
    
    labels=['FS-dop','FS-no_dop', 
            'FS-dop-C_m', 'FS-dop-V_t',
            'FS-dop-C_m_V_t']
    tata_dop=[0.8,0.0, 
              0.8,0.8,0.8]

    inv=In_vitro(Single_units_in_vitro, labels, tata_dop)

    inv.simulate_IV(1, numpy.arange(-200, 0,30), labels[0:2], 5000.0)
    inv.simulate_IF(1, numpy.arange(100, 300,10), labels[0:2], 5000.0)   
    inv.simulate_IF_variation(1, numpy.arange(0, 300,30), [labels[2]], 500.0, 10, ['C_m'])
    inv.simulate_IF_variation(1, numpy.arange(0, 300,30), [labels[3]], 500.0, 10, ['V_th'])
    inv.simulate_IF_variation(1, numpy.arange(0, 300,30), [labels[4]], 500.0, 10, ['C_m', 'V_th'])
    inv.show(labels)

if __name__ == "__main__":
    main()
    pylab.show()

