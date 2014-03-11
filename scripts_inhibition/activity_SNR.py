'''
Created on Aug 22, 2013

@author: lindahlm
'''
from activity import Activity_model
import pylab
from toolbox import plot_settings
class Activity_SNR(Activity_model):
    

    def show(self, labels_models, labels_fmin):
        
        fig, ax_list=plot_settings.get_figure( n_rows=1, n_cols=1,  w=1000.0, h=800.0, fontsize=12)
            
        colors=['g','b','r', 'm', 'c', 'k']
        linestyles=['solid', 'dashed','solid', 'dashed']
        coords=[[0.05, 0.9-i*0.04] for i in range(len(colors))] 
        
        self.plot_variable_population(ax_list[0], labels_fmin, colors, coords, linestyles, ylim=[0, 50.])
        
        return fig


def main():
    
    # * STN frequency is doubled during dopamine conditions.
    # * MSN frequency goes up
    
    tune_in_vitro=[['node.SN.I_vivo', 5.0,   'node.SN.target_rate_in_vitro', 'get_mean_rate',['SN'],{}]]
    tune_in_vivo=[ ['node.ESp.rate'  , 500.0, 'node.SN.target_rate',          'get_mean_rate',['SN'],{}]]
    stop=2000.0
    
    kwarg={'included_models':['ESp', 'M1p', 'GIp','STp', 'SN']}
    
    setup_models=[['$SNr_{vitro}$','SN-SN-no_input-dop', tune_in_vitro, 1000.0, stop,'ESp', kwarg],
                  ['$SNr_{vivo}$', 'SN-SN-all-dop',      tune_in_vivo,  1000.0, stop,'ESp', kwarg],
                  ['$SNr_{vivo-no-dop}$',  'SN-SN-all-no_dop', {}, 1000.0, stop, 'ESp', kwarg], #STN is doubled during dop
                  ]
    labels_models=[sl[0] for sl in setup_models]
    
    rand_setup={'n':200, 'rand_params':['C_m', 'V_th']}
    stop=21000.0
    setup_fmin=[['$Fmin-SNR_{vitro-rand}$',  [labels_models[0]], 1000.0, stop, rand_setup],
                ['$Fmin-SNR_{vivo-rand}$',   labels_models[1:], 1000.0, stop, rand_setup]
                ]      
    labels_fmin=[sl[0] for sl in setup_fmin]
 
    lesion_setup={'all':[], 
                  'no_input':['ESp', 'M1p', 'GIp', 'STp']}
    
    am=Activity_SNR(2, lesion_setup, setup_models, setup_fmin)
    am.find_params([0]*2, labels_fmin)
    am.simulate_variable_population([0]*2, labels_fmin, rand_setup={'rand_params': ['C_m','V_th'], 'n':200})    
     
    fig=am.show(labels_models, labels_fmin)
    pylab.show()
    fig.savefig( am.path_pictures  + am.sname  + '.svg', format = 'svg') 
    
if __name__ == "__main__":
    main()    
    