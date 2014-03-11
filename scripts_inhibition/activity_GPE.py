'''
Created on Jun 27, 2013

@author: lindahlm
'''
from activity import Activity_model
import numpy
import pylab
from toolbox import plot_settings
class Activity_GPE(Activity_model):
    

    def show(self, labels_models, labels_fmin):
        
        fig, ax_list=plot_settings.get_figure(n_rows=3, n_cols=2, w=1000.0, h=800.0, fontsize=12)
            
        colors=['g','b','r', 'm', 'c', 'k']
        linestyles=['solid', 'dashed','solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']
        coords=[[0.05, 0.9-i*0.1] for i in range(len(colors))] 
        
        self.plot_input_output(ax_list[0],labels_models[2:4], colors, coords)
        self.plot_input_output(ax_list[0],labels_models[2:4], colors, coords)
        self.plot_input_output_comb(ax_list[0], labels_models[2:4], colors[2], coords[2])
        self.plot_variable_population(ax_list[1], labels_fmin[0:4], colors, coords, linestyles, ylim=[0, 100])
        self.plot_variable_population(ax_list[2], labels_fmin[4:], colors, coords, linestyles,  ylim=[0, 100])
        self.plot_example(ax_list[4], [labels_models[0]], colors, coords)
        self.plot_example(ax_list[5], [labels_models[1]], colors, coords)
       
        return fig

    def plot_input_output_comb(self, ax, labels, color, coord):
        model_list, save_labels=self.get_models(labels)
        d=[]
        p=[0.2,1-0.2]
        for i, mo in enumerate(model_list):
            if not len(d):
                d=numpy.array(mo.data_input_output)
                d[1]*=p[i]
            else: d[1]+=numpy.array(mo.data_input_output[1])*p[i]
        #d=d/len(model_list)
        ax.plot(d[0], d[1], color)
        ax.text( coord[0], coord[1], 'combined', transform=ax.transAxes, 
                     fontsize=pylab.rcParams['font.size'], backgroundcolor = 'w', **{'color': color})
        ax.plot([300,1400],[30,30],'--k')
        pylab.setp(ax.lines, linewidth=2.0) # Need to pu ti before  
        ax.set_ylabel('Rate MSN (spikes/s)') 
        ax.set_xlabel('Cortical input (events/s)')

        
        ax.my_set_no_ticks( yticks=8, xticks = 6 )

def main():
    
    # * STN frequency is doubled during dopamine conditions.
    # * MSN frequency goes up
    GPE_A_tune_vitro=[['node.GA.I_vivo', -5.0, 'node.GA.target_rate_in_vitro', 'get_mean_rate',['GA'],{}]]
    GPE_I_tune_vitro=[['node.GI.I_vivo', 5.0,  'node.GI.target_rate_in_vitro', 'get_mean_rate',['GI'],{}]]
    GPE_A_tune_vivo= [['node.EAp.rate', 200.0, 'node.GA.target_rate', 'get_mean_rate',['GA'],{}]]
    GPE_I_tune_vivo= [['node.EIp.rate', 200.0, 'node.GI.target_rate', 'get_mean_rate',['GI'],{}]]
    stop=5000.0    
    
    kwargs1={'included_models':['EAp', 'EIp', 'GAp', 'GIp', 'STp', 'GA']}
    kwargs2={'included_models':['EAp', 'EIp', 'M2p' ,'GAp', 'GIp', 'STp', 'GI']}
    setup_models=[['$GPE-A_{vitro}$',      'GA-GA-no_input-dop', GPE_A_tune_vitro, 1000.0, stop, 'EAp', kwargs1],   
                  ['$GPE-I_{vitro}$',      'GI-GI-no_input-dop', GPE_I_tune_vitro, 1000.0, stop, 'EIp', kwargs2],   
                  ['$GPE-A_{vivo}$',       'GA-GA-all-dop', GPE_A_tune_vivo, 1000.0, stop, 'EAp', kwargs1],   
                  ['$GPE-I_{vivo}$',       'GI-GI-all-dop', GPE_I_tune_vivo, 1000.0, stop, 'EIp', kwargs2],                  
                  ['$GPE-A_{vivo-no-dop}$','GA-GA-all-no_dop', {}, 1000.0, stop, 'EAp', kwargs1], #STN is doubled during dop
                  ['$GPE-I_{vivo-no-dop}$','GI-GI-all-no_dop', {}, 1000.0, stop, 'EIp', kwargs2],
                  ]
    labels_models=[sl[0] for sl in setup_models]
    
    rand_setup={'n':200, 'rand_params':['C_m', 'V_th']}
    rand_setup_w_noise={'n':100, 'rand_params':['C_m', 'V_th'], 'rand_white_noise':20.}
    stop=21000.0
    setup_fmin=[['$Fmin-GPE-A_{vitro-rand}$',  [labels_models[0]], 1000.0, stop, rand_setup],
                ['$Fmin-GPE-I_{vitro-rand}$',  [labels_models[1]], 1000.0, stop, rand_setup],
                ['$Fmin-GPE-A_{vitro-rand-noise}$',  [labels_models[0]], 1000.0, stop, rand_setup_w_noise],
                ['$Fmin-GPE-I_{vitro-rand-noise}$',  [labels_models[1]], 1000.0, stop, rand_setup_w_noise],
                ['$Fmin-GPE-A_{vivo-rand}$',   labels_models[2::2], 1000.0, stop, rand_setup],
                ['$Fmin-GPE-I_{vivo-rand}$',   labels_models[3::2], 1000.0, stop, rand_setup],
                
                ]
        
    labels_fmin=[sl[0] for sl in setup_fmin]

    
    lesion_setup={'all':[],  
                  'no_input':['EAp', 'EIp', 'M2p' ,'GAp', 'GIp', 'STp']}
    am=Activity_GPE(1, lesion_setup, setup_models, setup_fmin)
    
    
    am.simulate_example([0]*2, labels_models[0:2], rand_setup={'rand_white_noise':20.0})
    am.find_params([0]*6, labels_fmin)
    
    am.simulate_variable_population([0]*6, labels_fmin, rand_setup={'rand_params': ['C_m','V_th'], 'n':200})    
      
    am.simulate_input_output([0]*2, labels_models[2:4], range(300,1500,100), ['EAp','EIp'])
    
    fig=am.show(labels_models, labels_fmin)
    pylab.show()
    fig.savefig( am.path_pictures  + am.sname  + '.svg', format = 'svg') 
    
if __name__ == "__main__":
    main()    
 
