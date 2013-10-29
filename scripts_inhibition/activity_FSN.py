'''
Created on Jul 4, 2013

@author: lindahlm
'''

from activity import Activity_model
import pylab
from toolbox import plot_settings
class Activity_FSN(Activity_model):
    

    def show(self, labels_models, labels_fmin):
        fig, ax_list=plot_settings.get_figure(n_rows=2, n_cols=2,  w=1000.0, h=800.0, fontsize=14)
        
        colors=['g','b','r', 'm', 'c', 'k']
        coords=[[0.05, 0.9-i*0.1] for i in range(len(colors))]
        linestyles=['-','-','--','--','-','--','-','--',]  
        linestyles_hist=['solid', 'dashed','solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']
        self.plot_input_output(ax_list[0], labels_models, colors, coords, linestyles)
        
        self.plot_variable_population(ax_list[1], labels_fmin, colors, coords, linestyles_hist, ylim=[0,80])
        self.plot_rheobase_variable_population(ax_list[2], [labels_models[0]], colors, coords, ylim=[0,60])
        
        fig.savefig( self.path_pictures  + self.sname  + '.svg', format = 'svg')

def main():
    
    tune=[['node.CFp.rate', 900.0, 'node.FS.target_rate', 'get_mean_rate',['FS'],{}]]
    stop=21000.0
    
    kwargs={'included_models':['CFp', 'GAp', 'FSp', 'FS']}
    setup_models=[['$FSN_{low}$',         'FS_low-FS-all-dop',   tune, 1000.0, stop, 'CFp', kwargs],
                  ['$FSN_{high}$',        'FS_high-FS-all-dop',  tune, 1000.0, stop, 'CFp', kwargs],
                  ['$FSN_{low-no-dop}$',  'FS_low-FS-all-no_dop',  [], 1000.0, stop, 'CFp', kwargs],
                  ['$FSN_{high-no-dop}$', 'FS_high-FS-all-no_dop', [], 1000.0, stop, 'CFp', kwargs],
                  ]
    labels_models=[sl[0] for sl in setup_models]

    stop=21000.0
    rand_setup={'n':400, 'rand_params':['C_m', 'V_th']}
    setup_fmin=[['$Fmin-FSN_{low-rand}$',  labels_models[0::2], 1000.0, stop, rand_setup],
                ['$Fmin-FSN_{high-rand}$', labels_models[1::2], 1000.0, stop, rand_setup],]
    
    labels_fmin=[sl[0] for sl in setup_fmin]
    
    lesion_setup={'all':[]}

    am=Activity_FSN(4, lesion_setup, setup_models, setup_fmin)
    
    am.simulate_input_output(  [1]*4, labels_models, range(500,1050,50))   
    am.find_params([0]*2, labels_fmin)

    am.simulate_variable_population([0]*2, labels_fmin, rand_setup={'rand_params': ['C_m','V_th'], 
                                                                    'n':400})    
    am.rheobase_variable_population([1]*1, labels_models, rand_setup={'rand_params': ['C_m','V_th'], 
                                                                      'n':400})
    
    am.show(labels_models, labels_fmin)
    pylab.show()
    
if __name__ == "__main__":
    main()     
    
    
