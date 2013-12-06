'''
Created on Oct 15, 2013

@author: lindahlm

Classes for running models, collect and process data and plot result.
'''

from toolbox import data_to_disk, plot_settings, misc
from toolbox import signal_processing
from toolbox.network_construction import Inhibition_base
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
    def __init__(self, label, **kwargs):


        self.data={'firing_rate':None, 
                   'firing_rate_sets':None,
                   'ids':None, 
                   'isis':None,
                   'mean_rates':None,
                   'rasters':None,
                   'rasters_sets':None}
          
        self.dud=None
        self.durd=None
        self.label=label    
              
        self.simulation_info=''        

        class_network_construction=kwargs.get('class_network_construction', Inhibition_base)  
        par_rep=kwargs.get('par_rep',{})
        perturbation=kwargs.get('perturbation', None)
        kwargs_network=kwargs.get('kwargs_network', None)
        
        self.network=class_network_construction( par_rep, perturbation, **kwargs_network) 
  
    @property
    def path_data(self):
        path=sys.argv[0].split('/')[-1].split('.')[0]
        path=path+'/size-'+str(int(self.network.par['netw']['size']))+'/'
        return self.network.path_data+path
    
    @property
    def type_dopamine(self):
        if self.network.par['netw']['tata_dop']==0.0:
            return 'no_dop'
        if self.network.par['netw']['tata_dop']==0.8:
            return 'dop'        
        else:
            return 'interm'  
    
    @property
    def start(self):
        return self.network.par['simu']['start_rec']

    @property
    def stop(self):
        return self.network.par['simu']['stop_rec']
    
    
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
            ax=ax_list[j].twinx()
            du=nm.dud[model]
            hist=du.firing_rate_sets
            times=du.firing_rate_sets_time
            hist=misc.convolve(hist, 10, 'triangle',single=False)
            
            
            m=0
            std=0
            SNR=0
            n=len(du.firing_rate)
            for i, h in enumerate(hist):
                if not model in ['SN', 'GI']:
                    time=numpy.arange(1,len(h)+1)
                    ax.plot(times, h, color='w', linewidth=3.)
                    ax.plot(times, h, color=colors[i])
                    
                    m+=numpy.mean(h)/n
                    std+=numpy.std(h)/n
                    if m>0:
                        SNR+=std/m/n
            
                #ax.text( coords[i][0], coords[i][1], 'Set '+str(i), transform=ax.transAxes, 
                #          fontsize=pylab.rcParams['font.size'], 
                #          backgroundcolor = 'w', **{'color': colors[i]})
            
            ax.set_xlim(xlim)
            #ax.set_title(model+' '+label+' m='+str(round(m,3))+' SNR='+str(round(SNR,1)))
            ax.set_ylabel('  Rate') 
            ax.my_set_no_ticks( yticks=3, xticks = 4 )      
            #if ((j+1) % len(models)==0):
                #ax.set_xlabel('Time (ms)')   

            if (j+1) % len(models)==0:
                pass
            else:
                ax.my_remove_axis(xaxis=True, yaxis=False )  
    def plot_rasters_bcpnn(self, ax_list, label, models, colors, coords, xlim=[5000., 6000.], plot_label='', plot_label_prefix=''):
        k=0
  
        nm=self
        for model in models:
            ax=ax_list[k]
            
            du=nm.dud[model]
            ids=[]
            for i, rasters in enumerate(du.rasters_sets):
                if len(rasters[1]) > 0:             
                    rasters[1]  
                    if model in ['M1','M2']:
                        mod=1
                    elif model in ['F1','F2', 'FS']:
                        mod=10
                    else:
                        mod=5
                    ax.plot(rasters[0][numpy.mod(rasters[1],mod)==0], rasters[1][numpy.mod(rasters[1],mod)==0], ',', color=colors[i])
                ids+=list(du.rasters_sets_ids[i])
                if not plot_label_prefix: pl='Set'
                else: pl=plot_label_prefix[k]  
                ax.text( coords[i][0], coords[i][1], pl+' '+str(i+1), transform=ax.transAxes, 
                          fontsize=pylab.rcParams['font.size'], 
                          backgroundcolor = 'w', **{'color': colors[i]})    
            if not plot_label: pl=plot_label
            else: pl=plot_label[k]
            ax.text( 0.8, 0.1, pl, transform=ax.transAxes, 
                        fontsize=pylab.rcParams['font.size'], 
                        backgroundcolor = 'w',
                         **{'color': 'k'})    
            ids=numpy.array(ids)  
              
            ax.set_xlim(xlim)
            ax.set_ylim([min(ids)-1, max(ids)+1])
            ax.set_ylabel('Neuron') 
            ax.my_set_no_ticks( yticks=3, xticks = 4 )    
            k+=1
            
            
        #ax.set_xlim([5000,6000])

                
            if k % len(models)==0:
                ax.set_xlabel('Time (ms)')
            else:
                ax.my_remove_axis(xaxis=True, yaxis=False )
            
     
    def simulate(self, model_list, print_time=True, **kwargs): 

        #print '\n*******************\nSimulatation setup\n*******************\n'
        
        network=self.network
        
        network.calibrate()
        network.inputs()
        network.build()
        network.randomize_params(['V_m', 'C_m'])
        network.connect()
        network.run(print_time=print_time)

        s=network.get_simtime_data()
        s+=' Network size='+str(network.par['netw']['size'])
        s+=' '+self.type_dopamine
        
        self.data['firing_rate']=network.get_firing_rate(model_list)   
        self.data['firing_rate_sets']=network.get_firing_rate_sets(model_list) 
        self.data['ids']=network.get_ids(model_list)      
        self.data['isis']=network.get_isis(model_list)          
        self.data['mean_rates']=network.get_mean_rates(model_list)       
        self.data['rasters']=network.get_rasters(model_list)
        self.data['rasters_sets']=network.get_rasters_sets(model_list)
        self.simulation_info=s                          

    
                   
class Network_models_dic():

    def __init__(self, setup_list_models, class_network_handling):
         
        self.dic={}
        for setup in setup_list_models:
            if len(setup)==6: setup.append({})
            label, kwargs=setup
            args=[label]
            assert issubclass(class_network_handling, Network_model), '%d class not a subclass of Network_model'%class_network_handling
            self.dic[label]=class_network_handling(*args, **kwargs)
    
    @property
    def network_model_list(self):
        return sorted(self.dic.keys())
    
    @property
    def path_pictures(self):
        label=self.network_model_list[0]
        self.path_pictures=self.dic[label].network_class().path_pictures+sys.argv[0].split('/')[-1].split('.')[0]

    
    @property
    def stop(self):   
        stop=None
        for label in self.network_model_list:
            if stop==None:
                stop=self.dic[label].stop
            else:
                if self.dic[label].stop > stop:
                    stop=self.dic[label].stop 

    @property
    def start(self):   
        start=None
        for label in self.network_model_list:
            if start==None: 
                start=self.dic[label].start
            else:
                if self.dic[label].start < start:
                    start=self.dic[label].start 

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
        r=(y2-y1)/y1
        r[numpy.isnan(r)]=0
        r[numpy.isinf(r)]=0
        return r
                
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
    numpy.isinf
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
        r=(y2-y1)/y1
        r[numpy.isnan(r)]=0
        r[numpy.isinf(r)]=0
        return r
        
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
    
    '''
    def plot_bcpnn_contrast(labels, odel=['SN']):
        
        for i in range(len(labels)):
            self[labels[i]].get_bcpnn_contras()
            
    '''    
        
        
        
                            
    def simulate(self, loads, labels, rec_models):
        
        
        if not isinstance(rec_models[0], list):
            rec_models=[rec_models]*len(loads)
        
        for load, label, rec_from in zip(loads, labels, rec_models):
 
            nm=self.dic[label]  
            save_txt_at= nm.path_data+'simulate_log' 
            save_at=nm.path_data+'simulate-' + nm.label

            if load==0 or ((load==2) and os.path.exists(save_at)):       
                
                
                nm.simulate(rec_from)  
                data_to_disk.txt_save_to_label(nm.simulation_info, label, save_txt_at)
                nm.data_save(save_at)                     
                              
            else:
                nm.data_load(save_at)
                #ids_dic,rates_dic, raster_dic=data_to_disk.pickle_load(save_at)
            
            nm.get_data_unit_dic(rec_from)
  
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

    def show_bcpnn(self, labels, models, xlim=[0, 250*10], plot_lables_models=[], plot_lables_prefix_models=[]):
        
        n_models=len(models)
        colors=['g','b','r','m','c',(0.5,0,0),(0.0,0.5,0),(0.0,0.0,0.5),'k',(0.5,0.5,0.5),]
        coords=[[0.05+0.1*(i/5), 0.1+i%5*0.15] for i in range(len(colors))]
        figs=[]
        i=0
        n_models=max([len(rec_models) for rec_models in models] )
        
        for label in labels:
            fig, ax_list=plot_settings.get_figure( n_rows=n_models, n_cols=1, w=500.0, h=800.0, fontsize=16,  order='row')
            figs.append(fig)                    
            
            if len(plot_lables_models): plot_label=plot_lables_models[i]
            else:  plot_label=''
            if len(plot_lables_models): plot_label_prefix=plot_lables_prefix_models[i]
            else:  plot_label_prefix=''
            self[label].plot_rasters_bcpnn(ax_list[:n_models], label, models[i], colors, 
                                    coords, xlim=[self[label].start, self[label].stop], plot_label=plot_label, plot_label_prefix=plot_label_prefix)            
            self[label].plot_firing_rate_bcpnn(ax_list[:n_models], label, models[i], colors, 
                                    coords, xlim=[self[label].start, self[label].stop])   
            i+=1          

        #self.plot_bcpnn_contrast()
        
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
    stop=500.0
    kwargs = {'kwargs_network':{'save_conn':False, 'verbose':False},
              'par_rep':{'simu':{'threads':4, 'sd_params':{'to_file':True, 'to_memory':False},
                                 'print_time':False, 'start_rec':1.0, 'stop_rec':stop, 
                                 'sim_time':stop,},
                                 'netw':{'size':500.0, 'tata_dop':0.8}}}
    colors=['g','b','r','m','c','g','b','r','m','c']
    coords=[[0.05+0.1*(i/5), 0.1+i%5*0.15] for i in range(len(colors))]
    
    def setUp(self):     
                
        self.label='unittest-inh_base'
        self.model_list=['M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
        nm=Network_model(self.label, **self.kwargs)
        self.path_data=nm.path_data + self.label

    def test_a_simulate(self):
        #self.nm.par_rep.update(self.par_rep)
        nm=Network_model(self.label, **self.kwargs)
        nm.simulate(self.model_list, print_time=False)            
        self.assertEqual(len(nm.data.keys()), 7)
        nm.data_save(self.path_data)      

    def test_get_data_unit_dic(self):       
        nm=Network_model(self.label, **self.kwargs)
        nm.data_load(self.path_data)
        nm.get_data_unit_dic(self.model_list)
        self.assertEqual(len(nm.data.keys()), 7)    

class TestNetwork_model_slow_wave(TestNetwork_model):
    
    def setUp(self):     
        from toolbox.network_construction import Slow_wave
           
        self.label='unittest-slow_wave'
        self.kwargs.update({'class_network_construction':Slow_wave })
        self.model_list=['M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
        nm=Network_model(self.label, **self.kwargs)
        self.path_data=nm.path_data + self.label      
             
class TestNetwork_model_h0(TestNetwork_model):
    
    def setUp(self):     
        from toolbox.network_construction import Bcpnn_h0
           
        self.label='unittest-h0'
        self.kwargs.update({'class_network_construction':Bcpnn_h0 })
        self.model_list=['CO','M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
        nm=Network_model(self.label, **self.kwargs)
        self.path_data=nm.path_data + self.label
                
                 
    def test_plot_firing_rate_bcpnn(self):
        fig, ax_list=plot_settings.get_figure( n_rows=len(self.model_list), n_cols=1, w=500.0, h=800.0, fontsize=8,  order='row')
        nm=Network_model(self.label, **self.kwargs)
        nm.data_load(self.path_data)
        nm.get_data_unit_dic(self.model_list)
        nm.plot_firing_rate_bcpnn(ax_list, self.label, self.model_list, self.colors, 
                                  self.coords, xlim=[nm.start, nm.stop])
        
    def test_plot_rasters_bcpnn(self):
        fig, ax_list=plot_settings.get_figure( n_rows=len(self.model_list), n_cols=1, w=500.0, h=900.0, fontsize=8,  order='row')
        nm=Network_model(self.label, **self.kwargs)
        nm.data_load(self.path_data)
        nm.get_data_unit_dic(self.model_list)
        nm.plot_rasters_bcpnn(ax_list, self.label, self.model_list, self.colors, 
                              self.coords, xlim=[nm.start, nm.stop])
      
        #pylab.show()      
        
        
class TestNetwork_model_h1(TestNetwork_model_h0):
    
    def setUp(self):     
        from toolbox.network_construction import Bcpnn_h1
        
           
        self.label='unittest-h1'
        self.kwargs.update({'class_network_construction':Bcpnn_h1 })
        self.model_list=['CO', 'M1','M2', 'F1', 'F2', 'GA', 'GI', 'ST', 'SN']
        nm=Network_model(self.label, **self.kwargs)
        self.path_data=nm.path_data + self.label



class TestNetwork_mode_dic(unittest.TestCase):
    
    def setUp(self):     
        from toolbox.network_construction import Bcpnn_h0, Bcpnn_h1, Slow_wave     
        # @TODO test run several models and show functions for bcpnn 
        network_class=[Inhibition_base, Slow_wave,  Bcpnn_h0, Bcpnn_h1]
        self.setup_list=[['unittest1'],
                         ['unittest2'],
                         ['unittest3'], 
                         ['unittest4']] 
        stop=500.0
        for i, setup in enumerate(self.setup_list): 
            kwargs = {'class_network_construction':network_class[i],   
                      'kwargs_network':{'save_conn':False, 'verbose':False}, 
                      'par_rep':{'simu':{'threads':4, 'sd_params':{'to_file':True, 'to_memory':False},
                                 'print_time':False, 'start_rec':1.0, 'stop_rec':stop,'sim_time':stop},
                                 'netw':{'size':500.0, 
                                         'tata_dop':0.8}}}
            setup.append(kwargs)

        self.labels=[sl[0] for sl in self.setup_list]
        self.model_lists=[['M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN'],
                          ['M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN'],
                          ['CO','M1', 'M2', 'FS', 'GA', 'GI', 'ST', 'SN'],
                          ['CO','M1', 'M2', 'F1', 'F2', 'GA', 'GI', 'ST', 'SN']   ]        
        
        
    def test_a_simulate(self):    
        nms=Network_models_dic(self.setup_list, Network_model)
        nms.simulate([0]*len(self.labels), self.labels, self.model_lists)
        
    def test_show_bcpnn(self):
        nms=Network_models_dic(self.setup_list, Network_model)
        nms.simulate([1]*len(self.labels), self.labels, self.model_lists)
        nms.show_bcpnn(self.labels, self.model_lists, xlim=[nms.start, nms.stop])    
        # pylab.show()
        
    def test_signal_pds(self):
        nms=Network_models_dic(self.setup_list, Network_model)
        nms.simulate([1]*len(self.labels), self.labels, self.model_lists)
        pds_setup    =[256, 10., 'gaussian',{ 'std_ms':5, 'fs':1000.0}]
        pds_models=['M1', 'M2', 'GA', 'GI', 'ST', 'SN']
        nms.signal_pds([0]*len(self.labels), self.labels, pds_models, pds_setup)
        
        for nm in nms.dic.values():
            for model in pds_models:
                du=nm.dud.dic[model]
                self.assertFalse(numpy.isnan(du.pds[0]).any())
                self.assertFalse(numpy.isnan(du.pds[1]).any())
        print nms 
        
    def test_signal_coherence(self):
        nms=Network_models_dic(self.setup_list, Network_model)
        nms.simulate([1]*len(self.labels), self.labels, self.model_lists)
        cohere_setup=[256, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}, 40]
        cohere_relations=['GA_GA', 'GI_GI', 'GA_GI','ST_GA', 'ST_GA']
        nms.signal_coherence([0]*len(self.labels), self.labels, cohere_relations, cohere_setup)
        
        for nm in nms.dic.values():
            for dur in nm.durd.dic.values():
                self.assertFalse(numpy.isnan(dur.cohere[0]).any())
                self.assertFalse(numpy.isnan(dur.cohere[1]).any())
        print nms

def load_tests(tests, loader):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite
     
if __name__ == '__main__':
    #test_cases = (TestNetwork_model, TestNetwork_model_slow_wave, TestNetwork_model_h0, TestNetwork_model_h1, TestNetwork_mode_dic)
    test_cases = [TestNetwork_mode_dic]
    loader=unittest.TestLoader()
    suite=load_tests(test_cases, loader)
    
    #suite = unittest.TestSuite()
    #suite.addTest(TestNetwork_mode_dic('test_signal_pds'))
    #suite.addTest(TestNetwork_mode_dic('test_signal_coherence'))
    
    #TestNetwork_model_h1
    unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main()    
           