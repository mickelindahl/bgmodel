#! Imports
import numpy
import pylab
import sys

 
from toolbox import plot_settings, data_handling
from network_classes import Single_units_in_vitro 

                    
class In_vitro(object):
    
    def __init__(self, Use_class, labels, dop, sname='', **kwargs):
        

        self.data_IF={}   
        self.data_IF_variation={}
        self.data_voltage_responses={}
        self.data_IV={}

        self.labels=labels
        
        self.kwargs={} #Kwargs for network class
        for i, label in enumerate(labels):
            self.kwargs[label]={'model_name': label.split('-')[0]}
            par_in={'netw':{'tata_dop':dop[i]}}
            self.kwargs[label].update({'par_in': par_in})
        self.Use_class=Use_class  
          
        if not sname:
            self.sname=sys.argv[0].split('/')[-1].split('.')[0]
        else:
            self.sname=sname
            
        self.path_data=self.Use_class().path_data  
        self.path_pictures=self.Use_class().path_pictures   
        
    def plot_IV(self, ax, labels, colors, coords, linestyles):
            
        for i, label in enumerate(labels):
            ax.plot(self.data_IV[label][0][:], self.data_IV[label][1][:], 
                    **{'color':colors[i], 'linestyle':linestyles[i]})
            ax.text( coords[i][0], coords[i][1], label, 
                     transform=ax.transAxes, fontsize=14, 
                     **{'color': colors[i]})
        
        ax.my_set_no_ticks( yticks=8, xticks = 6 )  
        ax.set_xlabel('Current (pA)') 
        ax.set_ylabel('Potential (mV)') 
        pylab.setp(ax.lines, linewidth=2.0) # Need to pu ti before generating legend
        #ax.set_xlim([-10, 200])
        
    def plot_IF(self, ax, labels, colors, coords, linestyles, xlim=[]):
        
        for i, label in enumerate(labels):
            ax.plot(self.data_IF[label][0], 1000./self.data_IF[label][1], 
                    **{'color':colors[i], 'linestyle':linestyles[i]})
            ax.text( coords[i][0], coords[i][1], label, 
                     transform=ax.transAxes, fontsize=14, 
                     **{'color': colors[i]})
    
        ax.my_set_no_ticks( yticks=8, xticks = 6 )  
        ax.set_xlabel('Current (pA)') 
        ax.set_ylabel('Rate (spikes/s)')
        if len(xlim): ax.set_xlim(xlim)
        #ax.set_xlim(self.data_IF[label][0][0], self.data_IF[label][0][-1])
    
    def plot_IF_var(self, ax, labels, colors, coords, linestyles):
        
        for i, label in enumerate(labels):
            ax.plot(self.data_IF_variation[label][0].transpose(), 
                    1000./self.data_IF_variation[label][1].transpose(), 
                    **{'color':colors[i], 'linestyle':linestyles[i]})
            ax.text( coords[i][0], coords[i][1], label, 
                     transform=ax.transAxes, fontsize=14, 
                     **{'color': colors[i]})
             
        ax.my_set_no_ticks( yticks=8, xticks = 6 )  
        ax.set_xlabel('Current (pA)') 
        ax.set_ylabel('Rate (spikes/s)')
            
    def plot_voltage_responses(self, ax, labels, colors, coords, linestyles): 
        for i, label in enumerate(labels):
            ax.plot(self.data_voltage_responses[label][0], 
                    self.data_voltage_responses[label][1], 
                    **{'color':colors[i], 'linestyle':linestyles[i]})
            ax.text( coords[i][0], coords[i][1], label, 
                     transform=ax.transAxes, fontsize=14, 
                     **{'color': colors[i]})
    
        ax.my_set_no_ticks( yticks=8, xticks = 6 )  
        ax.set_xlabel('Time (ms)') 
        ax.set_ylabel('Potential (mV)') 
      
    def simulate_IV(self, load, currents, labels, tStim):
        save_at=self.path_data+self.sname+'/'+'IV'
        if not load:
        
            for label in labels:
                kwargs=self.kwargs[label]
                suiv=self.Use_class(1,  0., float('inf'), **kwargs)
            
                I_vec, voltage=suiv.IV_curve(currents, tStim)
                self.data_IV[label]=[I_vec, voltage]
            data_handling.pickle_save(self.data_IV, save_at)
        else:
            self.data_IV=data_handling.pickle_load(save_at)
                        
    def simulate_IF_variation(self, load, currents, labels, tStim, n, randomization):
        save_at=self.path_data+self.sname+'/'+'IF_variation'
        
        if not load:
            for label in labels:
                self.kwargs[label].update({'n':n})
                suiv=self.Use_class(1,  0., float('inf'), **self.kwargs[label])
                I_vec, fIsi, mIsi, lIsi = suiv.IF_variation(currents, tStim, randomization)
                self.data_IF_variation[label]=[I_vec, lIsi]
            data_handling.pickle_save(self.data_IF_variation, save_at)
        else:
            self.data_IF_variation=data_handling.pickle_load(save_at)
                                          
    def simulate_IF(self, load, currents, labels, tStim):
        save_at=self.path_data+self.sname+'/'+'IF'
        if not load:
            for label in labels:
                suiv=self.Use_class(1,  0., float('inf'), **self.kwargs[label])
                I_vec, fIsi, mIsi, lIsi = suiv.IF_curve(currents, tStim)   
                self.data_IF[label]=[I_vec, lIsi]
            data_handling.pickle_save(self.data_IF, save_at)
        else:
            self.data_IF=data_handling.pickle_load(save_at)

    def simulate_voltage_responses(self, load, currents, times, start, stop, labels):
        save_at=self.path_data+self.sname+'/'+'voltage_responses'
        if not load:
            for label in labels:
                suiv=self.Use_class(1,  0., float('inf'), **self.kwargs[label])
                times, voltages= suiv.voltage_respose_curve(currents, times, start, stop)   
                self.data_voltage_responses[label]=[times, voltages]
            data_handling.pickle_save(self.data_voltage_responses, save_at)
        else:
            self.data_voltage_responses=data_handling.pickle_load(save_at)
    
    
    def show(self, labels):
        colors=['g','b', 'r','m']
        coords=[[ 0.05, 0.9], [0.05, 0.75], [0.05, 0.6], [0.05, 0.45]]
        linestyles=['-', '-', '-', '--']

        fig, ax_list=plot_settings.get_figure(2, 2)
                
        self.plot_IV(ax_list[0], labels[0:4], colors[0:4], coords[0:4], linestyles[0:4])
        self.plot_IF(ax_list[1], labels[0:4], colors[0:4], coords[0:4], linestyles[0:4])
        self.plot_IF_var(ax_list[2], labels[4:], colors[0:2], coords[0:2], linestyles[0:2])
        
        fig.savefig( self.path_pictures  + self.sname  + '.svg', format = 'svg') 
        
        
def main():
    
    labels=['MSN_D1-dop', 'MSN_D2-dop', 'MSN_D1-no_dop','MSN_D2-no_dop', 
            'MSN_D1-dop-C_m', 'MSN_D1-dop-V_t']
    tata_dop=[0.8,0.8,0.0,0.0, 
              0.8,0.8]
    dopamine=[True, True, True, True, 
              True, True]
    inv=In_vitro(Single_units_in_vitro, labels, tata_dop, dopamine)
    

    inv.simulate_IV(1, numpy.arange(-20, 200,10), labels[0:4], 5000.0)
    inv.simulate_IF(1, numpy.arange(210, 300,10), labels[0:4], 5000.0)   
    inv.simulate_IF_variation(1, numpy.arange(210, 300,10), [labels[4]], 5000.0, 10, ['C_m'])
    inv.simulate_IF_variation(1, numpy.arange(210, 300,10), [labels[5]], 5000.0, 10, ['V_th'])
    inv.show(labels)
  
#main()
#pylab.show()

