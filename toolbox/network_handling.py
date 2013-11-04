'''
Created on Oct 15, 2013

@author: lindahlm

Classes for running models, collect and process data and plot result.
'''

from toolbox import data_to_disk, plot_settings, misc
from toolbox import signal_processing
from toolbox.data_processing import Data_unit, Data_units_dic, Data_units_relation_dic

import copy
import numpy
import pylab
import os
import sys
import time


#network_class=Inhibition_base


class Network_model(object):            
    '''
    Class for running a model, process data and plot it. Uses a class from 
    network_classes for building the model. Uses classes from data_processing
    for processing data.
    '''
    def __init__(self, label, flag, size, start, stop, network_class, threads, **kwargs):

        type_dopamine=flag.split('-')
        if isinstance(type_dopamine, list):type_dopamine=type_dopamine[0]

        self.data={'firing_rate':None, 
                   'firing_rate_sets':None,
                   'ids':None, 
                   'isis':None,
                   'mean_rates':None,
                   'rasters':None,
                   'rasters_sets':None}
          
        self.dud=None
        self.dudr=None
        self.label=label    
        self.network_class=network_class    
        self.par_rep={}
        self.size=size
        self.start=start
        self.stop=stop
        self.simulation_info=''
        self.threads=threads
        self.type_dopamine=type_dopamine
        
        self.par_rep={}
        self.par_rep['netw']={'size':size}
             
        if 'perturbation' in kwargs.keys():
            self.perturbation=kwargs['perturbation']
        else:
            self.perturbation=None 
        
        path=sys.argv[0].split('/')[-1].split('.')[0]
        path=path+'/size-'+str(int(size))+'/'
            
        self.path_data=self.network_class().path_data+path
            
        #ax.my_remove_axis(xaxis=False, yaxis=False )
        
    def data_load(self, filename):
        self.data=data_to_disk.pickle_load(filename)    
        
    def data_save(self, filename):
        data_to_disk.pickle_save(self.data, filename)
        
    def get_data_unit_dic(self, models):

        dud=Data_units_dic(models)
        dud.set_firing_rate(self.data['firing_rate'])
        dud.set_firing_rate_sets(self.data['firing_rate_sets'])
        dud.set_ids(self.data['ids'])
        dud.set_isis(self.data['isis'])
        dud.set_mean_rates(self.data['mean_rates'])
        dud.set_rasters(self.data['rasters'])   
        dud.set_rasters_sets(self.data['rasters_sets'])
        
        if 'GI' in dud.model_list and 'GA' in dud.model_list:
            dud['GP']=dud['GI'].merge(dud['GA'])
        
        self.dud=dud
            
    
    def get_power_density_spectrum(self, load, save_at, models, setup):
        NFFT, kernel_extent, kernel_type, kernel_params=setup
        for model in models:
            save_model_at=save_at+'/'+model
            du=self.dud[model]
            du.get_power_density_spectrum(load, save_model_at, self.start, self.stop, NFFT, kernel_extent, kernel_type, kernel_params)
    
    def get_coherence(self, load, save_at, relations, setup):

        NFFT, kernel_extent, kernel_type, kernel_params, n_sample=setup   
        
        self.durd=Data_units_relation_dic(relations)
        
        for relation in relations:
            save_model_at=save_at+'/'+relation
            dur=self.durd[relation]
            model1, model2=relation.split('_')
            if not load:
                bd1=self.dud[model1].convert2bin( self.start, self.stop, n_sample, 1)
                bd2=self.dud[model2].convert2bin( self.start, self.stop, n_sample, 1)
            else: bd1, bd2=None, None
            dur.get_coherence(load, save_model_at, bd1, bd2, self.start, self.stop, NFFT, kernel_extent, 
                          kernel_type, kernel_params)


    def get_phase(self, load, save_at, models, setup):

        lowcut, highcut, order, kernel_extent, kernel_type, kernel_params=setup  
        
        for model in models:
            save_model_at=save_at+'/'+model
            du=self.dud[model]
            du.get_phase(load, save_model_at, self.start, self.stop, lowcut, highcut, order, 
                                   1000.0, kernel_extent, kernel_type, kernel_params)

    def get_phases(self, load, save_at, models, setup):

        lowcut, highcut, order, kernel_extent, kernel_type, kernel_params=setup  
        
        for model in models:
            save_model_at=save_at+'/'+model
            du=self.dud[model]
            du.get_phases(load, save_model_at, self.start, self.stop, lowcut, highcut, order, 
                                   1000.0, kernel_extent, kernel_type, kernel_params)
            
    def plot_cohere(self, ax_list, data_cohere, models):
        
        colors=['b','b','r','r']
        linestyles=['-','--','-','--']
        
        
        for i, model in enumerate(models):
            for j, dud in enumerate(data_cohere):
                ax=ax_list[i]
                key=model+'-'+model
                ax.plot(dud[key].data['coherence'][0], dud[key].data['coherence'][1], color=colors[j], linestyle=linestyles[j])
            ax.legend(['Dop', 'Dop-full', 'No-dop', 'No-dop-full'],prop={'size':8})
            ax.set_xlim([0,80])
            ax.text( 0.1,0.5, model, transform=ax.transAxes, 
                     fontsize=pylab.rcParams['font.size']+2, 
                     **{'color': 'k'})     
            if not i==0:
                ax.my_remove_axis(xaxis=True, yaxis=False )
            if i==0:
                ax.set_xlabel('Frequency (Hz)') 
            
            if i==len(models)-1:
                ax.set_title('Coherence')
            ax.my_set_no_ticks( yticks=5, xticks = 4 )   
            
    def plot_firing_rate_bcpnn(self, ax_list, label, models, colors, coords, xlim=[5000, 6000]):

        nm=self
        for j, model in enumerate(models):
            ax=ax_list[j]
            du=nm.dud[model]
            hist=du.firing_rate_sets
            times=du.firing_rate_sets_time
            hist=misc.convolve(hist, 10, 'triangle',single=False)
            
            
            m=0
            std=0
            SNR=0
            n=len(du.firing_rate)
            for i, h in enumerate(hist):
                time=numpy.arange(1,len(h)+1)
                ax.plot(times, h, color=colors[i])
                m+=numpy.mean(h)/n
                std+=numpy.std(h)/n
                if m>0:
                    SNR+=std/m/n
            
                ax.text( coords[i][0], coords[i][1], 'Set '+str(i), transform=ax.transAxes, 
                          fontsize=pylab.rcParams['font.size'], 
                          backgroundcolor = 'w', **{'color': colors[i]})
            
            ax.set_xlim(xlim)
            ax.set_title(model+' '+label+' m='+str(round(m,3))+' SNR='+str(round(SNR,1)))
            ax.set_ylabel('  Rate') 
            ax.my_set_no_ticks( yticks=3, xticks = 4 )      
            if ((j+1) % len(models)==0):
                ax.set_xlabel('Time (ms)')   

             
    def plot_rasters_bcpnn(self, ax_list, label, models, colors, coords, xlim=[5000., 6000.]):
        k=0
  
        nm=self
        for model in models:
            ax=ax_list[k]
            k+=1
            du=nm.dud[model]
            ids=[]
            for i, rasters in enumerate(du.rasters_sets):
                if len(rasters[1]) > 0:               
                    ax.plot(rasters[0], rasters[1], ',', color=colors[i])
                ids+=list(du.rasters_sets_ids[i])
                    
                ax.text( coords[i][0], coords[i][1], 'Set '+str(i), transform=ax.transAxes, 
                          fontsize=pylab.rcParams['font.size'], 
                          backgroundcolor = 'w', **{'color': colors[i]})    
            ids=numpy.array(ids)    
            ax.set_xlim(xlim)
            ax.set_ylim([min(ids)-1, max(ids)+1])
            ax.set_ylabel('Neuron') 
            ax.my_set_no_ticks( yticks=3, xticks = 4 )    
            
            
            
        #ax.set_xlim([5000,6000])
        #if not i==0:
        #    ax.my_remove_axis(xaxis=True, yaxis=False )
            if k % len(models)==0:
                ax.set_xlabel('Time (ms)')


    def update_parms(self):
        
        if self.type_dopamine=='dop': 
            misc.dict_recursive_add(self.par_rep, ['netw', 'tata_dop'], 0.8)
        elif self.type_dopamine=='no_dop': 
            misc.dict_recursive_add(self.par_rep, ['netw', 'tata_dop'], 0.0)
     
     
    def simulate(self, model_list, print_time=True, **kwargs): 

        print '\n*******************\nSimulatation setup\n*******************\n'
        self.update_parms()
        kwargs.update({'par_rep':self.par_rep, 'perturbation':self.perturbation })
        inh=self.network_class(self.threads, self.start, self.stop, **kwargs)
        
        inh.calibrate()
        inh.inputs()
        inh.build()
        inh.randomize_params(['V_m', 'C_m'])
        inh.connect()
        inh.run(print_time=print_time)

        s=inh.get_simtime_data()
        s+=' Network size='+str(inh.par['netw']['size'])
        s+=' '+self.type_dopamine
        
        self.data['firing_rate']=inh.get_firing_rate(model_list)   
        self.data['firing_rate_sets']=inh.get_firing_rate_sets(model_list) 
        self.data['ids']=inh.get_ids(model_list)      
        self.data['isis']=inh.get_isis(model_list)          
        self.data['mean_rates']=inh.get_mean_rates(model_list)       
        self.data['rasters']=inh.get_rasters(model_list)
        self.data['rasters_sets']=inh.get_rasters_sets(model_list)
        self.simulation_info=s                          

    
                   
class Network_models_dic():

    def __init__(self, threads, setup_list_models, Network_model_class):
         
        self.dic={}
        self.network_model_list=[]
        for setup in setup_list_models:
            if len(setup)==6: setup.append({})
            label, flag, size, start, stop, network_class, kwargs=setup
            args=[label, flag, size, start, stop, network_class, threads]
            assert issubclass(Network_model_class, Network_model), '%d class not a subclass of Network_model'%Network_model_class
            self.network_model_list.append(label)
            self.dic[label]=Network_model_class(*args, **kwargs)

        self.path_pictures=network_class().path_pictures+sys.argv[0].split('/')[-1].split('.')[0]

    def __getitem__(self, key):
        if key in self.network_model_list:
            return self.dic[key]
        else:
            raise Exception("Network %d is not present in the Network_models_dic. See model_list()" %key)

    def __setitem__(self, key, val):
        assert isinstance(val, Data_unit), "An Network_models_dic object can only contain Network_model objects"
        self.dic[key] = val    


    def get_mean_cohere(self, networks, relations, band):
        xticklabels=networks
        N=len(xticklabels)
        M=len(relations)
        
        y=numpy.zeros([N,M])
        for i, label in enumerate(networks):
            nm=self.dic[label]
            for j, relation in enumerate(relations):
                du=nm.durd[relation]
                f, mean_Cxy, p_conf95 = du.cohere
                
                idx=(f>band[0])*(f<band[1])
                band_Cxy=mean_Cxy[idx]
                
                y[i,j]=numpy.mean(band_Cxy)
          
        return y  
    def get_mean_cohere_std(self, networks, relations,band):
        N,M=len(networks), len(relations)

        y_std=numpy.zeros([N,M])
        for i, label in enumerate(networks):
            nm=self.dic[label]
            for j, relation in enumerate(relations):
                du=nm.durd[relation]
                f, mean_Cxy, p_conf95 = du.cohere
                
                idx=(f>band[0])*(f<band[1])
                band_Cxy=mean_Cxy[idx]

                y_std[i,j]=numpy.std(band_Cxy)
        return y_std
    def get_mean_cohere_change(self, networks1, networks2, relations, band):
        y1=self.get_mean_cohere(networks1, relations, band)
        y2=self.get_mean_cohere(networks2, relations, band)        
        return (y2-y1)/y1
                
    def get_mean_rate(self, networks, models):
        N, M=len(networks), len(models)
        y=numpy.zeros([N,M])

        for i, label in enumerate(networks):
            nm=self.dic[label]
            for j, model in enumerate(models):
                du=nm.dud[model]              
                mrs=du.mean_rates
                y[i,j]=numpy.mean(mrs)
        return y
    
    def get_mean_rate_std(self, networks, models):
        N, M=len(networks), len(models)
        y_std=numpy.zeros([N,M])

        for i, label in enumerate(networks):
            nm=self.dic[label]
            for j, model in enumerate(models):
                du=nm.dud[model]              
                mrs=du.mean_rates
                y_std[i,j]=numpy.std(mrs)
        return y_std

    def get_mean_rate_change(self, networks1, networks2, models):
        y1=self.get_mean_rate(networks1, models)
        y2=self.get_mean_rate(networks2, models)        
        return (y2-y1)/y1
        
    def plot_coherence(self, ax_list, labels, relations, colors, coords, xlim_coher=[0,80]):
        for i, label in enumerate(labels):

            nm=self.dic[label]
            for j, relation in enumerate(relations):
                ax=ax_list[j]
                dur=nm.durd[relation]
                ax.plot(dur.cohere[0], dur.cohere[1], color=colors[i])
                ax.plot(dur.cohere[0], dur.cohere[2], color='k', linestyle='--')
        
    
                #if not i==0:
                #    ax.my_remove_axis(xaxis=True, yaxis=False )
                if i==0 and j==len(relations)-1:
                    ax.set_xlabel('Frequency (Hz)') 
                
                if i==0:
                    ax.set_title(relation)
                    ax.set_ylabel('Coherence')
                ax.my_set_no_ticks( yticks=5, xticks = 4 )
                ax.set_xlim(xlim_coher)

    def plot_signal_processing_example(self, ax, y, title='', xlim=[], legend=[],  color='b', x=[]):
        if len(x): ax.plot(x, y, color) 
        else: ax.plot(y)
        if len(xlim): ax.set_xlim(xlim)                
        if title: ax.set_title(title)
        if len(legend): ax.legend(legend)
        
    def plot_power_density_spectrum(self, ax_list, labels, models, colors, coords, x_lim_pds=[0,80]):
        for i, label in enumerate(labels):

            nm=self.dic[label]
            for j, model in enumerate(models):
                ax=ax_list[j]
                du=nm.dud[model]
                ax.plot(du.pds[0], du.pds[1], color=colors[i])
        
    
                #if not i==0:
                #    ax.my_remove_axis(xaxis=True, yaxis=False )
                if i==0 and j==len(models)-1:
                    ax.set_xlabel('Frequency (Hz)') 
                
                if i==0:
                    ax.set_title(model)
                    ax.set_ylabel('Pds')
                ax.my_set_no_ticks( yticks=5, xticks = 4 )
                ax.set_xlim(x_lim_pds)
 
    def plot_rasters(self, ax_list, labels, models, colors, coords, x_lim=[5000., 6000.]):
        k=0
        for i, label in enumerate(labels):
            nm=self.dic[label]
            for model in models:
                ax=ax_list[k]
                k+=1
                print model
                du=nm.dud[model]
                if len(du.rasters[1]) > 0:
                    
                    ax.plot(du.rasters[0], du.rasters[1], ',',color=colors[i])
                
                
                ax.set_xlim(x_lim)
                ax.set_ylabel('Neuron') 
                ax.my_set_no_ticks( yticks=3, xticks = 4 )    
                
            #ax.set_xlim([5000,6000])
            #if not i==0:
            #    ax.my_remove_axis(xaxis=True, yaxis=False )
                if k % len(models)==0:
                    ax.set_xlabel('Time (ms)')    
        
                    
    def plot_rates_hist(self, ax_list, labels, models, colors, coords, xlim=[0, 100]):
       
        for i, label in enumerate(labels):

            nm=self.dic[label]
            for j, model in enumerate(models):
                ax=ax_list[j]
                du=nm.dud[model]

                mrs=copy.deepcopy(du.isis)
                for k in range(len(du.isis)):
                    if len(mrs[k]):
                        mrs[k]=1000.0/numpy.mean(mrs[k])
                    else:
                        mrs[k]=0
                
                if model in ['MSN_D1', 'MSN_D2']:
                    extent=[0,20]
                else:
                    extent=[0,100]
                    
                h, e, pathes=ax.hist(mrs, color=colors[i], histtype='step', range=extent, bins=50.0)
                #ax.set_xlim(xlim)
                ax.set_ylim([0, sum(h)*0.4])
                if i==0:
                    ax.set_title(model)
                
            #ax.set_title(model)
                ax.set_ylabel('Number (#)') 
            #ax.set_xlabel('Time (ms)') 
                ax.my_set_no_ticks( yticks=3, xticks = 4 )      
                if i==0  and ((j+1) % len(models)==0):
                    ax.set_xlabel('Time (ms)')   
            #ax.my_remove_axis(xaxis=True, yaxis=False )    
                                     
    def plot_firing_rate(self, ax_list, labels, models, colors, coords, xlim=[5000, 6000]):
       
        for i, label in enumerate(labels):

            nm=self.dic[label]
            for j, model in enumerate(models):
                ax=ax_list[j]
                du=nm.dud[model]
                hist=du.firing_rate
                hist=misc.convolve(hist, 10, 'triangle',single=True)
                time=numpy.arange(1,len(hist)+1)
                
                m=numpy.mean(hist)
                std=numpy.std(hist)
                SNR=std/m
                ax.text( coords[i][0], coords[i][1], label+' m='+str(round(m,3))+' SNR='+str(round(SNR,1)), transform=ax.transAxes, 
                     fontsize=pylab.rcParams['font.size'], 
                     backgroundcolor = 'w', **{'color': colors[i]})
                ax.plot(time, hist, color=colors[i])
                ax.set_xlim(xlim)
                if i==0:
                    ax.set_title(model)
                
            #ax.set_title(model)
                ax.set_ylabel('  Rate') 
            #ax.set_xlabel('Time (ms)') 
                ax.my_set_no_ticks( yticks=3, xticks = 4 )      
                if i==0  and ((j+1) % len(models)==0):
                    ax.set_xlabel('Time (ms)')   
            #ax.my_remove_axis(xaxis=True, yaxis=False )    
     
    def plot_firing_rate_bar(self, ax, networks, models, alpha, colors, coords):      
        y=self.get_mean_rate(networks, models)
        y_std=self.get_mean_rate_std(networks, models)
        N, M = y.shape
        ind = numpy.arange(N)
        width=1./(1+M)

        y_labels=[]
        y_coords=[]
        for i in range(N):
            for j in range(M):
                y_labels.append(str(round(y[i,j],1)))
                y_coords.append([width*j+0.1*width+i, y[i,j]])
        
        # Absolut values     
        rects=[]
        for i in range(M):
            rects.append(ax.bar(ind+width*i, y[:,i], width,color=colors[i], alpha=alpha ))
            (_, caplines, _) =ax.errorbar(ind+width*i+width*0.5, y[:,i], yerr=y_std[:, i], 
                                          fmt='o', color='k', markersize=5, linewidth=1.0,  
                                          capsize=5.0,markeredgewidth=1.0 )

        for label, coord in zip(y_labels,y_coords): 
            ax.text( coord[0], coord[1], label,
                     fontsize=pylab.rcParams['text.fontsize'], 
                     **{'color': 'k'})

        ax.set_xticks( numpy.arange(0.4,len(y)+0.4,1) )
        ax.set_xticklabels( networks, rotation=25, ha='right')          
        ax.set_ylabel('Rate')
        ax.set_xlabel('Parameter')
                        
        labels=models
        for label, coord, color in zip(labels,coords,colors):
            ax.text( coord[0], coord[1], label, transform=ax.transAxes, 
                     fontsize=pylab.rcParams['text.fontsize'], 
                    **{'color': color})
            
    def plot_firing_rate_bar_change(self, ax, networks1, networks2, models, alpha, colors, coords):      
        y=self.get_mean_rate_change(networks1, networks2, models)*100.
        N, M = y.shape
        ind = numpy.arange(N)
        width=1./(1+M)

        y_labels=[]
        y_coords=[]
        for i in range(N):
            for j in range(M):
                y_labels.append(str(round(y[i,j],1)))
                y_coords.append([width*j+0.1*width+i, y[i,j]])
        
        # Absolut values     
        rects=[]
        for i in range(M):
            rects.append(ax.bar(ind+width*i, y[:,i], width,color=colors[i], alpha=alpha ))

        for label, coord in zip(y_labels,y_coords): 
            ax.text( coord[0], coord[1], label,
                     fontsize=pylab.rcParams['text.fontsize'], 
                     **{'color': 'k'})
        
        xticklabels=[]
        for i in range(N):
            xticklabels.append(networks1[i]+networks2[i])
            

        ax.set_xticks( numpy.arange(0.4,len(y)+0.4,1) )
        ax.set_xticklabels( xticklabels, rotation=25, ha='right')          
        ax.set_ylabel('Rate Change')
        ax.set_xlabel('Parameter')
                        
        labels=models
        for label, coord, color in zip(labels,coords,colors):
            ax.text( coord[0], coord[1], label, transform=ax.transAxes, 
                     fontsize=pylab.rcParams['text.fontsize'], 
                    **{'color': color})        
  
    def plot_cohere_hist(self, ax, networks, relations, band, alpha, colors, coords):      
        y=self.get_mean_cohere(networks, relations, band)
        y_std=self.get_mean_cohere_std(networks, relations, band)
        
        N, M=y.shape
        ind = numpy.arange(N)
        width=1./(1+M)
      
        rects=[]
        for i in range(M):
            rects.append(ax.bar(ind+width*i, y[:,i], width, color=colors[i], alpha=alpha ))
            (_, caplines, _) =ax.errorbar(ind+width*i+width*0.5, y[:,i], yerr=y_std[:, 0], 
                                          fmt='o', color='k', markersize=5, linewidth=1.0,  
                                          capsize=5.0,markeredgewidth=1.0 )


        ax.set_xticks( numpy.arange(0.4,len(y)+0.4,1) )
        ax.set_xticklabels( networks, rotation=25, ha='right')
        ax.set_title('Freq band:'+str(band))
        ax.set_ylabel('Coherence')
        ax.set_xlabel('Parameter')
                    
        labels=relations
        for label, coord, color in zip(labels,coords,colors):
            ax.text( coord[0], coord[1], label, transform=ax.transAxes, 
                 fontsize=pylab.rcParams['text.fontsize'], 
                 **{'color': color})

    def plot_cohere_hist_change(self, ax, networks1, networks2, relations, band, alpha, colors, coords):      
        y=self.get_mean_cohere_change(networks1, networks2, relations, band)
        
        N, M=y.shape
        ind = numpy.arange(N)
        width=1./(1+M)
      
        rects=[]
        for i in range(M):
            rects.append(ax.bar(ind+width*i, y[:,i], width,color=colors[i], alpha=alpha ))

        xticklabels=[]
        for i in range(N):
            xticklabels.append(networks1[i]+networks2[i])

        ax.set_xticks( numpy.arange(0.4,len(y)+0.4,1) )
        ax.set_xticklabels( xticklabels, rotation=25, ha='right')
        ax.set_title('Freq band:'+str(band))
        ax.set_ylabel('Coherence change (%)')
        ax.set_xlabel('Parameter')
                    
        labels=relations
        for label, coord, color in zip(labels,coords,colors):
            ax.text( coord[0], coord[1], label, transform=ax.transAxes, 
                 fontsize=pylab.rcParams['text.fontsize'], 
                 **{'color': color})
                            
    def simulate(self, loads, labels, models, **kwargs_simulation):
        
        
        for load, label in zip(loads, labels):
 
            nm=self.dic[label]  
            save_txt_at= nm.path_data+'simulate_log' 
            save_at=nm.path_data+'simulate-' + nm.label

            if load==0 or ((load==2) and os.path.exists(save_at)):       
                
                nm.simulate(models, **kwargs_simulation)  
                data_to_disk.txt_save_to_label(nm.simulation_info, label, save_txt_at)
                nm.data_save(save_at)                     
                              
            else:
                nm.data_load(save_at)
                #ids_dic,rates_dic, raster_dic=data_to_disk.pickle_load(save_at)
            
            nm.get_data_unit_dic(models)
  
    def signal_coherence(self, loads, labels, relations, setup):
        
        for load, label in zip(loads, labels):
 
            nm=self.dic[label]
            
            save_at=nm.path_data+'signal_coherence-'+nm.label
            nm.get_coherence(load, save_at, relations, setup)

    def signal_pds(self, loads, labels, models, setup):
        
        for load, label in zip(loads, labels):
 
            nm=self.dic[label]
            
            save_at=nm.path_data+'signal_pds-'+nm.label
                  
            nm.get_power_density_spectrum(load, save_at, models, setup)

    def signal_phase(self, loads, labels, models, setup):
        
        for load, label in zip(loads, labels):
 
            nm=self.dic[label]
            
            save_at=nm.path_data+'signal_phase-'+nm.label
                  
            nm.get_phase(load, save_at, models, setup)

    def signal_phases(self, loads, labels, models, setup):
        
        for load, label in zip(loads, labels):
 
            nm=self.dic[label]
            
            save_at=nm.path_data+'signal_phases-'+nm.label
                  
            nm.get_phases(load, save_at, models, setup)
            
    def show_signal_processing_example(self, label,  model):
        model='GPE_I'
        NFFT=256
        kernel_extent=10.
        kernel_type='gaussian'
        kernel_params={'std_ms':5, 'fs':1000.0}
        n_data=0
        xlim=[0,1000]
       
        nm=self.dic[label]
        spk=nm.dud[model].convert2bin( nm.start, nm.stop, 2, 1)
        spk_rates=nm.dud[model].spike_rates
        spk_conv=misc.convolve(spk, kernel_extent, kernel_type, axis=0, single=False, params=kernel_params, no_mean=True)
        #px, f=signal_processing.psd(spk_rates, NFFT=NFFT, Fs=1000, noverlap=NFFT/2)
        px1, f=signal_processing.psd(spk_conv[0,:], NFFT=NFFT, Fs=1000, noverlap=NFFT/2)
        px2, f=signal_processing.psd(spk_conv[1,:], NFFT=NFFT, Fs=1000, noverlap=NFFT/2)
        cp, f=signal_processing.csd(spk_conv[0,:], spk_conv[1,:], NFFT=NFFT, Fs=1000, noverlap=NFFT/2)
        coh, f=signal_processing.cohere(spk_conv[0,:], spk_conv[1,:], NFFT=NFFT, Fs=1000, noverlap=NFFT/2)
        
        fig, ax_list=plot_settings.get_figure( n_rows=4, n_cols=2, w=1000.0, h=800.0, fontsize=12)


        self.plot_signal_processing_example( ax_list[0],  spk.transpose(), 'Raw spike trains', xlim, ['Neuron 1', 'Neuron 2'])
        self.plot_signal_processing_example( ax_list[1],  spk_conv.transpose(),'Convolved '+kernel_type+' '+str(kernel_extent)+' '+str(kernel_params), xlim, ['Neuron 1', 'Neuron 2'])
        self.plot_signal_processing_example( ax_list[2],  spk_rates,'Nuclei rates', xlim)
        self.plot_signal_processing_example( ax_list[3],  spk_rates,'Pds spike rates', [0,80])
        self.plot_signal_processing_example( ax_list[4],  px1,'Pds Neuron 1', [0,80], x=f)
        self.plot_signal_processing_example( ax_list[5],  px2,'Pds Neuron 2', [0,50], x=f, color='g')
        self.plot_signal_processing_example( ax_list[6],  abs(cp),'Cds', [0,80], x=f)
        self.plot_signal_processing_example( ax_list[7],  abs(coh),'Coherence', [0,80], x=f)

        
        for ax in ax_list:
            ax.my_set_no_ticks( yticks=5, xticks = 4 )          
        return fig
        
    def show_exclude_rasters(self, labels, models, relations, xlim=[5000,6000], xlim_pds=[0,80],  xlim_coher=[0,80]):
        
        n_models=len(models)
        fig, ax_list=plot_settings.get_figure( n_rows=n_models, n_cols=3, w=1000.0, h=800.0, fontsize=8,  order='row')
                
                
        colors=['g','b','r', 'm', 'c', 'k']
        coords=[[0.05, 0.8-i*0.15] for i in range(len(colors))]
        
        self.plot_firing_rate(ax_list, labels, models, colors, coords, xlim)
        self.plot_power_density_spectrum(ax_list[n_models:], labels, models, colors, coords, xlim_pds) 
        self.plot_coherence(ax_list[2*n_models:], labels, relations, colors, coords, xlim_coher)        
        return fig          
    
    def show(self, labels, models, relations):
        
        n_models=len(models)
        fig, ax_list=plot_settings.get_figure( n_rows=n_models, n_cols=6, w=1400.0, h=800.0, fontsize=8,  order='row')
                                
        colors=['g','b','r', 'm', 'c', 'k']
        coords=[[0.05, 0.8-i*0.15] for i in range(len(colors))]
                        
        self.plot_rasters(ax_list, labels, models, colors, coords)
        self.plot_firing_rate(ax_list[2*n_models:], labels, models, colors, coords)
        self.plot_rates_hist(ax_list[3*n_models:], labels, models, colors, coords)
        self.plot_power_density_spectrum(ax_list[4*n_models:], labels, models, colors, coords) 
        self.plot_coherence(ax_list[5*n_models:], labels, relations, colors, coords)        
        return fig           

    def show_bcpnn(self, labels, models, xlim=[0, 250*10]):
        
        n_models=len(models)
        colors=['g','b','r','m','c','g','b','r','m','c']
        coords=[[0.05+0.1*(i/5), 0.1+i%5*0.15] for i in range(len(colors))]
        figs=[]
        for label in labels:
            fig, ax_list=plot_settings.get_figure( n_rows=n_models, n_cols=2, w=1400.0, h=800.0, fontsize=8,  order='row')
            figs.append(fig)                    

            self[label].plot_rasters_bcpnn(ax_list[:n_models], label, models, colors, 
                                    coords, xlim=[self[label].start, self[label].stop])            
            self[label].plot_firing_rate_bcpnn(ax_list[n_models:], label, models, colors, 
                                    coords, xlim=[self[label].start, self[label].stop])             

     
        return figs
    
    def show_compact(self, networks, models, relations, band=[15,25]):

        fig, ax_list=plot_settings.get_figure( n_rows=4, n_cols=1, 
                                               w=1400.0, h=800.0, fontsize=8,  order='row')
        colors=['g','b','r', 'm', 'c', 'k']
        coords=[[0.05, 0.8-i*0.15] for i in range(len(colors))]
                        
        self.plot_firing_rate_bar(ax_list[0], networks, models, .5, colors, coords)
        self.plot_firing_rate_bar_change(ax_list[1], networks[::2],networks[1::2], models, .5, colors, coords)
        self.plot_cohere_hist(ax_list[2], networks, relations, band, .5, colors, coords)
        self.plot_cohere_hist_change(ax_list[3], networks[::2], networks[1::2], relations, band, .5, colors, coords)
        
        return fig    


import unittest
class TestNetwork_model(unittest.TestCase):
    
    def setUp(self):     
        from toolbox.network_construction import Bcpnn
        
        
        size=500.0
        self.start=500.0
        self.stop=250.*10.+self.start
        flag='dop'
        self.label='unittest'
        network_class=Bcpnn
        threads=1
        kwargs={'save_conn':False, 'sd_params':{'to_file':True, 'to_memory':False}}
        
        '''
        data_path_learned_conn={'M1':'~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-fake/CO_M1.pkl'}
        data_path_learned_conn['M2']='~/git/bgmodel/scripts_bcpnnbg/data_conns/conn-fake/CO_M2.pkl'
        
        self.par_rep={'conn':{}}
        for model in ['M1', 'M2']:
            self.par_rep['conn']['CO_'+model+'_ampa']={'data_path_learned_conn':data_path_learned_conn[model]}
            self.par_rep['conn']['CO_'+model+'_nmda']={'data_path_learned_conn':data_path_learned_conn[model]}
        '''
        self.nm=Network_model(self.label, flag, size, self.start, self.stop, network_class, threads, **kwargs)
        
        self.model_list=['CO','M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
        self.path_data=self.nm.path_data+'unittest-h0'
        #self.path_data_h0=self.nm.path_data+'unittest-h0'
           
        self.colors=['g','b','r','m','c','g','b','r','m','c']
        self.coords=[[0.05+0.1*(i/5), 0.1+i%5*0.15] for i in range(len(self.colors))]
        
    ''' 
    def test_simulate(self):
        #self.nm.par_rep.update(self.par_rep)
        self.nm.simulate(self.model_list, print_time=False)            
        self.assertEqual(len(self.nm.data.keys()), 7)
        self.nm.data_save(self.path_data)
    ''' 
    def test_get_data_unit_dic(self):       
        self.nm.data_load(self.path_data)
        self.nm.get_data_unit_dic(self.model_list)
        self.assertEqual(len(self.nm.data.keys()), 7)      
        
            
    def test_plot_firing_rate_bcpnn(self):
        fig, ax_list=plot_settings.get_figure( n_rows=len(self.model_list), n_cols=1, w=500.0, h=800.0, fontsize=8,  order='row')
        
        self.nm.data_load(self.path_data)
        self.nm.get_data_unit_dic(self.model_list)
        self.nm.plot_firing_rate_bcpnn(ax_list, self.label, self.model_list, self.colors, 
                                    self.coords, xlim=[self.start, self.stop])
        
    def test_plot_rasters_bcpnn(self):
        fig, ax_list=plot_settings.get_figure( n_rows=len(self.model_list), n_cols=1, w=500.0, h=800.0, fontsize=8,  order='row')
        
        self.nm.data_load(self.path_data)
        self.nm.get_data_unit_dic(self.model_list)
        self.nm.plot_rasters_bcpnn(ax_list, self.label, self.model_list, self.colors, 
                                    self.coords, xlim=[self.start, self.stop])
      
        #pylab.show()      
       
class TestNetwork_mode_dic(unittest.TestCase):
    
    def setUp(self):     
        from toolbox.network_construction import Bcpnn     
        # @TODO test run several models and show functions for bcpnn 
        self.size=500.0
        self.start=1.0
        self.stop=500.
        network_class=Bcpnn
        threads=1 
        setup_list=[['unittest1',     'dop'],
                    ['unittest2',     'no_dop']]  
        #Inhibition_no_parrot
        #for setup in setup_list: setup.extend([10000., 1000.0, 11000.0, Inhibition_no_parrot, {}])
        for setup in setup_list: setup.extend([self.size, self.start, self.stop, network_class, {}])
        self.labels=[sl[0] for sl in setup_list]
        self.model_list=['CO','M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']        
        self.kwargs={'sd_params':{'to_file':True, 'to_memory':False}}
        
        self.nms=Network_models_dic(threads, setup_list, Network_model)
        
    def test_simulate(self):    
        self.nms.simulate([1]*len(self.labels), self.labels, self.model_list, **self.kwargs)
        
    def test_show_bcpnn(self):
        self.nms.simulate([0]*len(self.labels), self.labels, self.model_list, **self.kwargs)
        self.nms.show_bcpnn(self.labels, self.model_list, xlim=[self.start, self.stop])    
        pylab.show()
        
if __name__ == '__main__':
    unittest.main()    
           