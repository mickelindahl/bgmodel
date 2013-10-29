'''
Created on Jun 27, 2013

@author: lindahlm
'''

from activity import Activity_model
import pylab
from toolbox import plot_settings
class Activity_STN(Activity_model):

    def show(self, labels_models, labels_fmin):
        fig, ax_list=plot_settings.get_figure(n_rows=2, n_cols=2,  w=1000.0, h=800.0, fontsize=14)
        
        colors=['g','b','r', 'm', 'c', 'k']
        coords=[[0.05, 0.9-i*0.1] for i in range(len(colors))]      
        linestyles=['-','-','--','--','-','--','-','--',]  
        self.plot_input_output(ax_list[0], labels_models[1:3], colors, coords, linestyles)
        self.plot_variable_population(ax_list[1],  [labels_fmin[0]], colors, coords)
        self.plot_variable_population(ax_list[2],  [labels_fmin[1]], colors, coords)
        
        return fig

def main():


    tune_vitro=[['node.ST.I_vivo', 10.0, 'node.ST.target_rate_in_vitro', 'get_mean_rate',['ST'],{}]]
    tune_vivo= [['node.CSp.rate', 190.0, 'node.ST.target_rate',          'get_mean_rate',['ST'],{}]]  
    stop=51000.0
    
    kwargs={'included_models':['CSp', 'GIp', 'ST']}
    setup_models=[['$STN_{vitro}$',       'ST-ST-no_input-dop', tune_vitro, 1000.0, stop,'CSp', kwargs],
                  ['$STN_{vivo}$',        'ST-ST-all-dop',      tune_vivo, 1000.0, stop,'CSp', kwargs],
                  ['$STN_{vivo-no-dop}$', 'ST-ST-all-no_dop', [], 1000.0, stop, 'CSp', kwargs], #STN is doubled during dop
                  ]
    labels_models=[sl[0] for sl in setup_models]
    
    rand_setup={'n':400, 'rand_params':['C_m', 'V_th']}
    stop=21000.0
    setup_fmin=[['$Fmin-STN_{vitro-rand}$', [labels_models[0]], 1000.0, stop, rand_setup],
                ['$Fmin-STN_{vivo-rand}$',   labels_models[1:], 1000.0, stop, rand_setup],
                ]
    
    labels_fmin=[sl[0] for sl in setup_fmin]
    
    lesion_setup={'all' : [], 
                  'no_input' : ['CSp', 'GIp']}

    am=Activity_STN(2, lesion_setup, setup_models, setup_fmin)
    am.simulate_input_output(  [1]*2, labels_models[1:3], range(0,1050,50))
    
    am.find_params([0]*2, labels_fmin)
    am.simulate_variable_population([0]*2, labels_fmin, rand_setup={'rand_params': ['C_m','V_th'], 'n':400})    


    
    am.show(labels_models, labels_fmin)
    pylab.show()
    
if __name__ == "__main__":
    main()      

