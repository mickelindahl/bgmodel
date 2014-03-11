'''
Created on Jun 27, 2013

@author: lindahlm
'''

from activity import Activity_model
import numpy
import pylab
from toolbox import plot_settings

class Activity_MSN(Activity_model):
    
    def show(self, labels_models, labels_fmin):
        fig, ax_list=plot_settings.get_figure( n_rows=3, n_cols=2, w=1000.0, h=800.0, fontsize=12)
        
        colors=['g','b','r', 'm', 'c', 'k']
        coords=[[ 0.05, p] for p in numpy.linspace(0.4,0.9, len(colors))]
        linesyles=['-','--','-','--','-','--','-','--',]  
        
        
        

        self.plot_input_output(ax_list[0], labels_models[0::2], colors, coords, linesyles)
        self.plot_input_output(ax_list[1], labels_models[1::2], colors, coords, linesyles)     
        self.plot_variable_population(ax_list[2], labels_fmin[0:2], colors, coords)
        self.plot_variable_population(ax_list[3], labels_fmin[2:4], colors, coords)
        self.plot_rheobase_variable_population(ax_list[4], labels_models[0:2], colors, coords, ylim=[0,100])
        
        fig.savefig( self.path_pictures  + self.sname  + '.svg', format = 'svg')

def main():
    
    D1_variables=[['node.C1.rate', 600.0,'node.M1.target_rate', 'get_mean_rate',['M1'],{}]]
    D2_variables=[['node.C2.rate', 800.0,'node.M2.target_rate', 'get_mean_rate',['M2'],{}]]
 
    stop=21000.0
    kwargs1={'included_models':['C1p', 'C2p', 'M1p', 'M2p', 'FSp' ,'GAp', 'M1']}
    kwargs2={'included_models':['C1p', 'C2p', 'M1p', 'M2p', 'FSp' ,'GAp', 'M2']}
    setup_models=[['$M1_{low}$',  'M1_low-M1-all-dop',  D1_variables, 1000.0, stop, 'C1p', kwargs1],
                 [' $M2_{low}$',  'M2_low-M2-all-dop',  D2_variables, 1000.0, stop, 'C2p', kwargs2],
                  ['$M1_{high}$', 'M1_high-M1-all-dop', D1_variables, 1000.0, stop, 'C1p', kwargs1],
                  ['$M2_{high}$', 'M2_high-M2-all-dop', D2_variables, 1000.0, stop, 'C2p', kwargs2],
                  ['$M1_{low-no-dop}$',  'M1_low-M1-all-no_dop',  [], 1000.0, stop, 'C1p', kwargs1],
                  ['$M2_{low-no-dop}$',  'M2_low-M2-all-no_dop',  [], 1000.0, stop, 'C2p', kwargs2],
                  ['$M1_{high-no-dop}$', 'M1_high-M1-all-no_dop', [], 1000.0, stop, 'C1p', kwargs1],
                  ['$M2_{high-no-dop}$', 'M2_high-M2-all-no_dop', [], 1000.0, stop, 'C2p', kwargs2],]
    labels_models=[sl[0] for sl in setup_models]

    stop=101000.0
    rand_setup={'n':100, 'rand_params':['C_m', 'V_th']}
    setup_fmin=[['$Fmin-M1_{low-rand}$',  labels_models[0::4], 1000.0, stop, rand_setup],
                ['$Fmin-M2_{low-rand}$',  labels_models[1::4], 1000.0, stop, rand_setup],
                ['$Fmin-M1_{high-rand}$', labels_models[2::4], 1000.0, stop, rand_setup],
                ['$Fmin-M2_{high-rand}$', labels_models[3::4], 1000.0, stop, rand_setup]]
    
    
    labels_fmin=[sl[0] for sl in setup_fmin]
    
    lesion_setup={'all':[]}

    am=Activity_MSN(1, lesion_setup, setup_models, setup_fmin)
    

    am.simulate_input_output([0]*8, labels_models, range(500,1050,50))
    
    am.find_params([0]*4, labels_fmin)
    
    am.simulate_variable_population([0]*4, labels_fmin, rand_setup={'rand_params': ['C_m','V_th'], 'n':200})    

    am.rheobase_variable_population([0]*2, labels_models, rand_setup={'rand_params': ['C_m','V_th'], 'n':200})
    
    am.show(labels_models, labels_fmin)
    pylab.show()
    
if __name__ == "__main__":
    main()  
