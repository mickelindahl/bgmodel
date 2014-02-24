#! Imports



import copy
import numpy
import pylab
import sys
import scipy.optimize as opt
from toolbox.network.construction import Single_units_activity 
from toolbox import data_to_disk, plot_settings, misc, my_nest
from toolbox.network.structure import Base_dic
from toolbox.misc import Stopwatch, Stop_stdout

                    
class In_vitro(object):
    
    def __init__(self, Use_class, labels, dop, sname='', **kwargs):
        

        self.data_IF={}   
        self.data_IF_variation={}
        self.data_voltage_responses={}
        self.data_IV={}

        self.labels=labels
        
        self.kwargs={} #Kwargs for network class
        self.par_rep={} #Kwargs for network class
        for i, label in enumerate(labels):
            self.kwargs[label]={'model_name': label.split('-')[0]}
            self.par_rep[label]={'simu':{'threads':1, 'start_rec':0., 'stop_rec': float('inf'),
                                         'sd_params': {'to_file': True, 'to_memory': False}}, 
                                 'netw': {'tata_dop':dop[i]}}
        
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
            model=label.split('-')[0]
        
            k={'color':colors[i], 'linestyle':linestyles[i]}
            self.data_IF[label].get_model(model).plot_IF_curve(ax, **k)
            ax.text( coords[i][0], coords[i][1], label, 
                     transform=ax.transAxes, fontsize=14, 
                     **{'color': colors[i]})
    
        ax.my_set_no_ticks( yticks=8, xticks = 6 )  
        if len(xlim): ax.set_xlim(xlim)

    
    def plot_IF_var(self, ax, labels, colors, coords, linestyles):
        
        for i, label in enumerate(labels):
            model=label.split('-')[0]
            k={'color':colors[i], 'linestyle':linestyles[i]}
            dud=self.data_IF_variation[label]
            dud.get_model(model).plot_IF_curve(ax, **k)

            ax.text( coords[i][0], coords[i][1], label, 
                     transform=ax.transAxes, fontsize=14, 
                     **{'color': colors[i]})             
        ax.my_set_no_ticks( yticks=8, xticks = 6 )  

            
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
                suiv=self.Use_class(self.par_rep[label], **self.kwargs[label])
            
                I_vec, voltage=suiv.IV_curve(currents, tStim)
                self.data_IV[label]=[I_vec, voltage]
            data_to_disk.pickle_save(self.data_IV, save_at)
        else:
            self.data_IV=data_to_disk.pickle_load(save_at)
                        
    def simulate_IF_variation(self, load, currents, labels, tStim, n, randomization):
        save_at=self.path_data+self.sname+'/'+'IF_variation'
        
        if not load:
            for label in labels:
                self.kwargs[label].update({'n':n})
                suiv=self.Use_class(self.par_rep[label], **self.kwargs[label])
                
                
                dud = suiv.IF_variation(currents, tStim, randomization)
                self.data_IF_variation[label]=dud
                data_to_disk.pickle_save(self.data_IF_variation[label], save_at+label)
        else:
            for label in labels:
                self.data_IF_variation[label]=data_to_disk.pickle_load(save_at+label)
                                          
    def simulate_IF(self, load, currents, labels, tStim):
        save_at=self.path_data+self.sname+'/'+'IF'
        if not load:
            for label in labels:
                suiv=self.Use_class(self.par_rep[label], **self.kwargs[label])
                dud= suiv.IF_curve(currents, tStim)   
                self.data_IF[label]=dud#[I_vec, lIsi]
            data_to_disk.pickle_save(self.data_IF, save_at)
        else:
            self.data_IF=data_to_disk.pickle_load(save_at)

    def simulate_voltage_responses(self, load, currents, times, start, stop, labels):
        save_at=self.path_data+self.sname+'/'+'voltage_responses'
        if not load:
            for label in labels:
                suiv=self.Use_class(self.par_rep[label], **self.kwargs[label])
                times, voltages= suiv.voltage_respose_curve(currents, times, start, stop)   
                self.data_voltage_responses[label]=[times, voltages]
            data_to_disk.pickle_save(self.data_voltage_responses, save_at)
        else:
            self.data_voltage_responses=data_to_disk.pickle_load(save_at)
    
    
    def show(self, labels):
        colors=['g','b', 'r','m']
        coords=[[0.05, 0.9-i*0.1] for i in range(len(colors))] 
        linestyles=['-', '-', '-', '--']

        fig, ax_list=plot_settings.get_figure(n_rows=2, n_cols=2,  w=1000.0, h=800.0, fontsize=14)
                
        self.plot_IV(ax_list[0], labels[0:2], colors, coords, linestyles)
        self.plot_IF(ax_list[1], labels[0:2], colors, coords, linestyles)
        self.plot_IF_var(ax_list[2], labels[2:4], colors, coords, linestyles)
        self.plot_IF_var(ax_list[3], [labels[-1]], colors, coords, linestyles)
        
        fig.savefig( self.path_pictures  + self.sname  + '.svg', format = 'svg') 
'''
Activity_model(self.model_lables[0], self.kwargs_list[0])
        kwargs=[{'input_models':['CFp', 'GAp', 'FSp', 'FS'], 'verbose':False,
                       'lesioned':[], 'nest_model':None, 
                       'par_rep':{'simu':{'threads':1, 'start_rec':1000.0, 'stop_rec':stop, 
                                          'sim_time':stop, 'print_time':False},
                                  'netw':{'tata_dop':None}}}]
    
        kwargs+=[deepcopy(kwargs[0])]
        kwargs[1].update({'par_xopt':{'node':{'CFp':{'rate': 970}}}, 
                               'par_key_ftar':[['node','FS','target_rate']], 
                               'fun_ftar':['get_mean_rate'], 
                               'par_key_input':['node','CFp'],
                               'args_ftar':[['FS']],
                               'kwargs_ftar':[{}]})        
                               '''


def builder():
    pass


class Fmin_wrapper():
    '''Used for optimization'''
    def __init__(self, network, *args, **kwargs):
              
        self.data=None
        self.m=network        
        self.verbose=kwargs.get('verbose', False)
        
    def __repr__(self):
        return self.__class__.__name__+':'+self.name    

    def add(self, attr, from_disk, *args, **kwargs):
        file_name=self.m.get_path_data()+attr
        
        if from_disk:
            val=data_to_disk.pickle_load(file_name)
            self.set(attr, val)
            
        else:
            call=getattr(self, 'sim_'+attr)
            with Stop_stdout(not self.verbose):
                val=call(*args, **kwargs)
            data_to_disk.pickle_save(val, file_name)
            self.set(attr, val)
            
    def set(self, attr, val):
        if attr in self.data.keys():
            self.data[attr]=val
        else:
            raise AttributeError
class Network_wrapper(object):
    '''Use design pattern dependancy injection'''
    def __init__(self, network, *args, **kwargs):
              
        self.data={'FF_curve':None,
                  }
        self.m=network        
        self.verbose=kwargs.get('verbose', False)
        
    def __repr__(self):
        return self.__class__.__name__+':'+self.name

#    def __getattr__(self, name):
#        print name
#        if name in self.data.keys():
#            return self.data[name]
#        else:
#            raise AttributeError
    
    def get(self, *args, **kwargs):
        return self.m.get(*args, **kwargs)
        
    def add(self, attr, from_disk, *args, **kwargs):
        file_name=self.m.get_path_data()+attr
        
        if from_disk:
            val=data_to_disk.pickle_load(file_name)
            self.set(attr, val)
            
        else:
            call=getattr(self, 'sim_'+attr)
            with Stop_stdout(not self.verbose):
                val=call(*args, **kwargs)
            data_to_disk.pickle_save(val, file_name)
            self.set(attr, val)
            
    def plot_FF_curve(self, *args, **kwargs):
        self.data['FF_curve'].plot_FF_curve(*args, **kwargs)         
       
    def sim_FF_curve(self,input_rates, stim_time, **kwargs):
        kwargs={'stim':input_rates,
                'stim_name':'node.'+self.get('FF_input')+'.rate',
                'stim_time':stim_time,
                'model':self.get('FF_output')}
        
        return self.m.sim_FF_curve(**kwargs)
       
#  
#    def build(self):
#        #params_in, kwargs=self.get_params()
#        asu=Single_units_activity(self.par_rep, **self.kwargs)
#        asu.build()
#        #if len(self.rand_params):
#        #    asu.randomize_params(self.rand_params)
#    
#        return asu  
    
    def set(self, attr, val):
        if attr in self.data.keys():
            self.data[attr]=val
        else:
            raise AttributeError
        
    
class Network_wrapper_list():
    def __init__(self, network_list, **kwargs):
        self.allowed=['plot_FF_curve']
        self.l=network_list
        self.verbose=kwargs.get('verbose', False)

    def __getattr__(self, name):
        if name in self.allowed:
            self.attr=name
            return self._caller
        else:
            raise AttributeError(name)  

    def __iter__(self):
        
        for val in self.l:
            yield val
    
    def __repr__(self):
        return self.__class__.__name__+':'+str([str(l) for l in self]) 
    
    def _caller(self, *args, **kwargs):
        a=[]
        
        if args==[]:
            args=[args]*len(self.l)
        
        for obj in self:
            call=getattr(obj, self.attr)
            with Stop_stdout(not self.verbose):
                d=call(*args, **kwargs)
            if d:
                a.append(d)               
        return a
    
    def add(self, attr, from_disk, *args, **kwargs):
        for net, fd in zip(self, from_disk):
            with Stop_stdout(not self.verbose):
                net.add(attr, fd, *args, **kwargs)
        
    def append(self, val):
        self.l.append(val)
        
                           
class Activity_model_dic(Base_dic):
    
    def __init__(self, *args, **kwargs):#, setup_list_fmin):
        super( Activity_model_dic, self ).__init__(*args, **kwargs) 
        
        
#        self.label_list=[] 
#         
#        self.obj_models={}
#        self.obj_fmins={}
#        for setup in setup_list_models:
#            if len(setup)==4: setup.append({})
#            label, flag,  kwargs=setup
#            
#            self.label_list.append(label)
#            self.obj_models[label]=Activity_model(label, flag, lesion_setup, **kwargs)
#        
#        for setup in setup_list_fmin:
#            if len(setup)==2: setup.append({})
#            label_fmin, label_list_models, kwargs=setup
#            obj_model_list=[]
#            for label in label_list_models:
#                obj_model_list.append(copy.deepcopy(self.obj_models[label]))
#                
#            self.obj_fmins[label_fmin]=Fmin(label_fmin, obj_model_list, **kwargs)

        #self.path_data=Single_units_activity.get_path_data()
       # self.path_pictures=Single_units_activity().path_pictures
        self.sname=sys.argv[0].split('/')[-1].split('.')[0]
        
        
    @property
    def threads(self):
        threads=[]
        for mo in self.get_models(self.label_list)[0]:
            threads.append(mo.threads)
            
        for val in self.obj_fmins.values():
            for mo in self.get_models(val.label_list)[0]:
                threads.append(mo.threads)
        assert threads[0]==sum(threads)/len(threads), "Inconsistant threads parameter between models"
        return threads[0]
    
    @threads.setter
    def threads(self, val):
        for mo in self.get_models(self.label_list)[0]:
            mo.threads=val
            
        for mo in self.get_models(self.obj_fmins.keys())[0]:
            mo.threads=val
                
    def add(self, *a, **k):
        class_name=k.get('class', 'Activity_model')
        the_class=misc.import_class(('toolbox.network.handling_single_units.'
                                     +class_name))

        # del k['class']
        self.dic[a[0]]=the_class(*a, **k)
       
        
         
    
    def add_model_fmins(self, model, setup):
        self.obj_fmins[model].add_model(self.lesion_setup, setup, len(self.obj_models.keys()))
    
    def get_model(self, name):    
        return self.dic[name]
        
    def get_xopt(self, labels):
        if isinstance(labels, str) :
            labels=[labels]
        
        xopt_labels=[]
        for label in labels:
                
            if label in self.obj_models.keys():
                xopt_labels.append('')
            else:
                xopt=self.obj_fmins[label].xopt
                xopt_labels.extend([' '+'xopt='+str(numpy.round(xopt, 1))]*len(self.obj_fmins[label].obj_model_list))
        return xopt_labels
    
                
    def plot_example_mm(self, ax, labels, colors, coords, mm_type):
        
        model_list, save_labels =self.get_models(labels)
        
        for i, mo in enumerate(model_list):
            d=mo.data_example_mm
            for key in d[1].keys():
                if key[0]==mm_type:
                    ax.plot(d[0][key], d[1][key], **{'label':key})
            ax.text( coords[i][0], coords[i][1], mo.label, transform=ax.transAxes, 
                     fontsize=pylab.rcParams['font.size'], 
                     backgroundcolor = 'w', **{'color': colors[i]})
        
        pylab.setp(ax.lines, linewidth=2.0) # Need to pu ti before  
        ax.set_ylabel('Membrane potential (mV)') 
        ax.set_xlabel('Time (ms)') 
        ax.legend(numpoints=1)
    def plot_example(self, ax, labels, colors, coords):
        
        model_list, save_labels =self.get_models(labels)
        
        for i, mo in enumerate(model_list):
            d=mo.data_example
            ax.plot(d[0], d[1], colors[i])
            ax.text( coords[i][0], coords[i][1], mo.label, transform=ax.transAxes, 
                     fontsize=pylab.rcParams['font.size'], 
                     backgroundcolor = 'w', **{'color': colors[i]})
        
        pylab.setp(ax.lines, linewidth=2.0) # Need to pu ti before  
        ax.set_ylabel('Membrane potential (mV)') 
        ax.set_xlabel('Time (ms)') 
        
        
    def plot_example_rates_fmin(self, ax, labels, colors, coords, ylim=None):

        obj_model_list=[]
        for label in labels:
            for mo in self.obj_fmins[label].obj_model_list:
                mo.label+=' '+'xopt='+str(numpy.round(self.obj_fmins[label].xopt, 1))
            obj_model_list.extend(self.obj_fmins[label].obj_model_list)

        width=0.8
        alpha=1.0
        rects=[]
        xticklabels=[]
        for i, mo in enumerate(obj_model_list):
            y=mo.data_mean_rate
            rects.append(ax.bar(0+i, y, width, color=colors[i], alpha=alpha ))
            ax.text( coords[i][0], coords[i][1], mo.label, transform=ax.transAxes, 
                     fontsize=pylab.rcParams['font.size'], 
                     backgroundcolor = 'w', **{'color': colors[i]})
            xticklabels.append(mo.label)

        if ylim: ax.set_ylim(ylim)
        ax.set_xlim([0,len(obj_model_list)])
        #ax.set_xticks( numpy.arange(0.4,len(obj_model_list)+0.4,1) )
        #ax.set_xticklabels( xticklabels, rotation=15, ha='right')

          
    def plot_input_output(self, ax, labels, colors, coords, linestyle=[], ylim=None, xlim=None):
        
        model_list=self.dic.values()
        if not len(linestyle):linestyle=['-']*len(model_list)
        for i, mo in enumerate(model_list):
            d=mo.data['input_output']
            kwargs={'color':colors[i], 'linestyle':linestyle[i], 
                    'label':mo.label}
            d[mo.study_name].plot_FF_curve(ax, **kwargs)
            
            #ax.plot(d[0], d[1], colors[i], linestyle=linestyle[i])
            #ax.text( coords[i][0], coords[i][1], mo.label, transform=ax.transAxes, 
            #         fontsize=pylab.rcParams['font.size'], backgroundcolor = 'w', **{'color': colors[i]})
        
        pylab.setp(ax.lines, linewidth=2.0) # Need to pu ti before  
        ax.set_ylabel('Rate MSN (spikes/s)') 
        ax.set_xlabel('Cortical input (events/s)')
        ax.legend(numpoints=2, loc='upper left')
        ax.my_set_no_ticks( yticks=8, xticks = 6 )
        #ax.set_xlim(misc.adjust_limit([550, 950]))
        if ylim: ax.set_ylim(misc.adjust_limit(ylim))
        if xlim: ax.set_xlim(misc.adjust_limit(xlim))
           

    def plot_variable_population(self, ax, labels, colors, coords, linestyles=[], ylim=None):        
        hist=[]
        model_list, save_labels = self.get_models(labels)
        xopt_labels =self.get_xopt(labels)
        for i, mo in enumerate(model_list):
            d=mo.data_variable_population
            h, e, pathes=ax.hist(d, color=colors[i], histtype='step')
            hist.append(h)
            
            ax.text( coords[i][0], coords[i][1], mo.label+xopt_labels[i]+' mr='+str(round(numpy.mean(d),2)), transform=ax.transAxes, 
                     fontsize=pylab.rcParams['font.size'], backgroundcolor = 'w',  **{'color': colors[i]})
        
            pylab.setp(pathes, linewidth=2.0) # Need to pu ti before  
            if len(linestyles):pylab.setp(pathes, linestyle=linestyles[i]) 
        
        #ax.set_ylim([0,numpy.max(numpy.max(numpy.array(hist)))])
        if ylim: ax.set_ylim(ylim)
    
    def plot_rheobase_variable_population(self, ax, labels, colors, coords, ylim=None):

        for i, label in enumerate(labels):
            d=self.obj_models[label].data_rheobase_variable_population
            h, e, pathes=ax.hist(d, color=colors[i], histtype='step')
            ax.text( coords[i][0], coords[i][1], label+' mr='+str(round(numpy.mean(d),2))+' std='+str(round(numpy.std(d),2)), transform=ax.transAxes, 
                     fontsize=pylab.rcParams['font.size'], backgroundcolor = 'w', **{'color': colors[i]})
        
            pylab.setp(pathes, linewidth=2.0) # Need to pu ti before  
       
        ax.set_ylabel('Numbers (#)') 
        ax.set_xlabel('Rheobase current (pA)')
        
        ax.my_set_no_ticks( yticks=8, xticks = 6 ) 
        if ylim: ax.set_ylim(ylim)
        
        
#    def find_params(self, loads, labels_fmin):
#        
#        
#        for load, label_fmin in zip(loads, labels_fmin):
#            
#            save_at=self.path_data+self.sname+'/find_param'+label_fmin              
#            save_log_at=self.path_data+self.sname+'/find_param.log'  
#
#            s=self.obj_fmins[label_fmin].fmin(load, save_at)
#            data_to_disk.txt_save_to_label(s, label_fmin, save_log_at )
#            
#            
        
    def simulate_example(self, loads, labels,  rand_setup={}):

        for load, label in zip(loads, labels):
            model_list, save_labels=self.get_models(label)
            
            for mo, sv_label in zip(model_list, save_labels):
                save_at=mo.path_data+self.sname+'/simulate_example-'+sv_label
                if not load:
    
                    if rand_setup: mo.set_rand(rand_setup)
                        
                    asu=mo.simulate(record_voltage=True)     
                    asu.get_spikes(asu.study_name)
                    mo.data_mean_rate=asu.get_mean_rate(asu.study_name)
                    mo.data_example=asu.get_voltage_trace(asu.study_name)
                    
                    data_to_disk.pickle_save([mo.data_example, mo.data_mean_rate], save_at)      
                
                else:
                    mo.data_example, mo.data_mean_rate=data_to_disk.pickle_load(save_at)
    
    def simulate_example_mm(self, loads, labels,  rand_setup={}, mm_types=['I']):

        for load, label in zip(loads, labels):
            model_list, save_labels=self.get_models(label)
            
            for mo, sv_label in zip(model_list, save_labels):
                save_at=mo.path_data+self.sname+'/simulate_example_synaptic_currents-'+sv_label
                if not load:
    
                    if rand_setup: mo.set_rand(rand_setup)
                        
                    asu=mo.simulate(record_mm=mm_types)     
                    
                    mo.data_example_mm=asu.get_mm(asu.study_name, mm_types)
                    
                    data_to_disk.pickle_save([mo.data_example_mm], save_at)      
                
                else:
                    mo.data_example_mm=data_to_disk.pickle_load(save_at)      
                                       
    def simulate_input_output(self, loads, models, input_rates, rand_setup={},
                              stim_time=500.0):
        
#        if not isinstance(input_names, list):
#            input_names=[input_names]
#        if len(input_names)==1:
#            input_names=input_names*len(labels)
#            
           
        for load, name in zip(loads, models):
            am=self.get_model(name)
            save_at=(am.path_data+self.sname+'/simulate_input_vs_model-'
                    +str(name))
     
            if not load:
                kwargs={'stim':input_rates,
                        'stim_name':'node.'+am.stim_model+'.rate',
                        'stim_time':stim_time,
                        'model':am.study_name}
                dud=am.sim_FF_curve(**kwargs)
          
#                    for r in input_rates:
#                        if not input_name: input_name=mo.input_name
#                        misc.dict_recursive_add(mo.par_var, ['node', input_name, 'rate'], float(r))
#                        
#                        if rand_setup: mo.set_rand(rand_setup)
#
#                        
#                        asu=mo.simulate(record_voltage=False)
#                        mr=asu.get_spike_statistics(asu.study_name)[0]
#                        output_rates.append(mr)
#                        
#                    data_to_disk.pickle_save([input_rates, output_rates], save_at)
                data_to_disk.pickle_save(dud, save_at)
            else:
                dud=data_to_disk.pickle_load(save_at)
            am.data['input_output']=dud
        

    def simulate_variable_population(self, loads, labels, rand_setup={}):
        
        for load, label in zip(loads, labels):
            model_list, save_labels=self.get_models(label)
            
            for mo, sv_label in zip(model_list, save_labels):
                
                save_at=self.path_data+self.sname+'/simulate_variable_population-'+sv_label
                
                if rand_setup: mo.set_rand(rand_setup)
                
                if not load:  
    
                    asu=mo.simulate()  
                    mrs=asu.get_mean_rates(asu.study_name)
                    data_to_disk.pickle_save(mrs, save_at)
                    
                else: 
                    mrs=data_to_disk.pickle_load(save_at)
                    
                mo.data_variable_population=mrs
            
    def rheobase_variable_population(self, loads, models, rand_setup={}):
        save_at=self.path_data+self.sname+'/rheobase_variable_population'
        for load, model in zip(loads, models):
            mo=self.obj_models[model]
            save_at=save_at+'-'+mo.label
            if rand_setup: mo.set_rand(rand_setup)
            if not load:  

                asu=mo.build()
                gs=my_nest.GetStatus
                ids=asu.units_dic[mo.study_name].population.ids
                params=gs(ids, ['k','E_L','V_th','b_2','V_b', 'p_2','C_m'])  

                v_rheo=[]
                I_rheo=[]
                for p in params:
                    k, E_L, V_th, b_2, V_b, p_2, C_m=p
                    f=lambda V:-( k*(V-V_th)*(V-E_L)-b_2*(V-V_b)**p_2)
                    f_v_rheo=lambda V :((2*V-V_th-E_L)*k
                                        -(p_2*b_2*(V-V_b)**(p_2-1)))**2
                    out=opt.fmin(f_v_rheo, -60, args=(), maxiter=200, 
                                 maxfun=200, full_output=1, retall=1, disp=0)
                    [xopt,fopt, iter, funcalls , warnflag, allvecs] = out
                    v_rheo.append(xopt)
                    I_rheo.append(abs(f(xopt[0])))
                    
                data_to_disk.pickle_save(I_rheo, save_at)
                
            else: 
                I_rheo=data_to_disk.pickle_load(save_at)
                
            mo.data_rheobase_variable_population=I_rheo
            
    def show(self, labels_models, labels_fmin, labels_variable_pop):
        
        fig, ax_list=plot_settings.get_figure(n_rows=3, n_cols=3, 
                                              w=1000.0, h=800.0, fontsize=12)
        
        colors=['g','b','r', 'm', 'c', 'k']*2
        coords=[[0.05, 0.9-i*0.1] for i in range(len(colors))]
        #self.plot_example_models(ax_list[0], [labels_models[0]], colors, coords)
        self.plot_input_output(ax_list[0], labels_models[0:6], colors, coords)
        self.plot_example_fmin(ax_list[1], labels_fmin, colors, coords)
        self.plot_variable_population(ax_list[2], labels_variable_pop[0:4], colors, coords)
        self.plot_variable_population(ax_list[3], labels_variable_pop[4:8], colors, coords)
        
        return fig

    def show2x2(self, labels_models, labels_fmin):
        fig, ax_list=plot_settings.get_figure(n_rows=2, n_cols=2,  w=1000.0, h=800.0, fontsize=14)
        
        colors=['g','b','r', 'm', 'c', 'k']*2
        coords=[[0.05, 0.9-i*0.1] for i in range(len(colors))]
        linestyles=['-','-','--','--','-','--','-','--',]  
        linestyles_hist=['solid', 'dashed','solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']
        self.plot_input_output(ax_list[0], labels_models, colors, coords, linestyles)
        
        self.plot_variable_population(ax_list[1], labels_fmin, colors, coords, linestyles_hist, ylim=[0,80])
        self.plot_rheobase_variable_population(ax_list[2], [labels_models[0]], colors, coords, ylim=[0,60])
        
        fig.savefig( self.path_pictures  + self.sname  + '.svg', format = 'svg')
        
        
        
import unittest
from copy import deepcopy
from toolbox.network.construction import Unittest, Network_list
class TestModel_wrapper(unittest.TestCase):

    def setUp(self):
        name='unittest'
        self.Network=Unittest
        net=[self.Network(name)]
        self.mw=Network_wrapper(net[0])

     
    def test_add_FF_curve(self):
        self.mw.add('FF_curve', False, *[range(2500,3000,100), 500.])
        self.mw.add('FF_curve', True, *[range(2500,3000,100), 500.])
        #self.mw.plot_FF_curve()
        #pylab.show()

class TestModel_wrapperList(unittest.TestCase):

    def setUp(self):
        name='unittest'
        nl=Network_list([])
        k={'par_rep':{'netw':{'FF_curve':{'output':'n3'}},
                      'node':{'n4':{'rate':2500.0}}}}
        l=[Network_wrapper(Unittest('U1')),
           Network_wrapper(Unittest('U2', **k))]
        self.nwl=Network_wrapper_list(l)   
       
    def test_add_FF_curve(self):
        self.nwl.add('FF_curve', [False]*2, range(2500,3000,200), 500.)
        self.nwl.add('FF_curve', [True]*2, range(2500,3000,200), 500.)
        #self.nwl.plot_FF_curve()
        #pylab.show()
            
            
if __name__ == '__main__':
    test_classes_to_run=[TestModel_wrapper,
                         #TestModel_wrapperList
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)    
    