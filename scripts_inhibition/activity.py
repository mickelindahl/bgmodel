'''
Created on Jun 27, 2013

@author: lindahlm
'''
import copy
from network_classes import Single_units_activity 
import numpy
import pylab
from toolbox import data_handling, plot_settings, misc, my_nest
import time
import sys
import scipy.optimize as opt

class Activity_model_obj(object):
    
    def __init__(self, label, flag, variables_setup, start, stop, threads,  lesion_setup, input_name, **kwargs):
        
        nest_model, study_name, lesion_type, dopamine_type =flag.split('-')
        
        self.data_input_output=[]
        self.data_example=[]
        self.data_example_mm={}
        self.data_mean_rate=[]
        self.data_fitting=[]
        self.data_variable_population=[]
        self.data_rheobase_variable_population=[]
        
        self.dopamine_type=dopamine_type
        self.input_name=input_name
        self.label=label
        self.lesion_type=lesion_type
        self.lesion_setup=lesion_setup
        
        if 'params_in' in kwargs.keys():
            self.params_in=kwargs['params_in']
        else:
            self.params_in={}
        
        self.included_models=kwargs['included_models'] 
        
        misc.dict_recursive_add(self.params_in, ['node', study_name, 'n'], 1)
        misc.dict_recursive_add(self.params_in, ['node', study_name, 'model'],nest_model)
        
        self.rand_params=[]
        self.rand_white_noise=False
        self.start=start
        self.stop=stop
        self.study_name=study_name  
        self.threads=threads        
        self.variables={}
        self.variables_key_x0=[]
        self.variables_setup=variables_setup
        self.variables_eval=[]
        
        # Here method is a network_class method
        for key, x0, target_val_path, method, method_args, method_kwargs in self.variables_setup:
            
            # Update variable dictionary
            misc.dict_recursive_add(self.variables, key.split('.'), x0)         
            self.variables_key_x0.append([key.split('.'), x0])
            self.variables_eval.append([target_val_path.split('.'), method, method_args, method_kwargs])  

    
    def get_params(self):#_D1(nest_model):

        kwargs={}
        params_in={}
        #params_in=misc.dict_merge(params_in, self.params_in)

        # Update with variable parameters
        params_in=misc.dict_merge(params_in, self.variables)

        kwargs['study_name']=self.study_name   
        kwargs['included_models']=copy.deepcopy(self.included_models)         
                          
        for source in self.lesion_setup[self.lesion_type]:
            params_in=misc.dict_recursive_add(params_in, ['node',source,'rate'], 0.0) 
            kwargs['included_models'].remove(source)
                                         
        if self.dopamine_type=='dop':    tata_dop=0.8
        if self.dopamine_type=='no_dop': tata_dop=0.0
     
        params_in=misc.dict_recursive_add(params_in, ['netw','tata_dop'], tata_dop) 
        
        return params_in, kwargs 
    
    def set_rand(self, rand_setup):
        if 'rand_params' in rand_setup.keys():
            self.rand_params=rand_setup['rand_params']
        if 'n' in rand_setup.keys():
            misc.dict_recursive_add(self.params_in, ['node', self.study_name, 'n'], rand_setup['n'])
            #self.study_n=rand_setup['n']   
        if 'rand_white_noise' in rand_setup.keys():
            self.rand_white_noise=rand_setup['rand_white_noise'] 
    
    def simulate(self, record_voltage=False, record_mm=False):
        
        params_in, kwargs=self.get_params()
        self.params_in=misc.dict_merge(self.params_in, params_in) 
        kwargs['save_conn']=False
        asu=Single_units_activity(self.threads, self.start, self.stop, **kwargs)
        asu.calibrate() 
        asu.par.update(self.params_in)
        asu.par.overwrite_dependables(self.params_in)
        asu.inputs()
        asu.build()
        if self.rand_params: asu.randomize_params(self.rand_params)
        asu.record_voltage(record_voltage)
        asu.record_mm(record_mm)
        asu.connect()
        if self.rand_white_noise:
            noise=my_nest.Create('noise_generator', params={'mean':0.,'std':self.rand_white_noise})
            rec=my_nest.GetStatus(asu.units_dic[self.study_name].population.ids)[0]['receptor_types']
            targets=asu.units_dic[self.study_name].population.ids
            for target in targets:
                my_nest.Connect( noise, [target],  params = { 'receptor_type' : rec['CURR'] } )
               
        asu.run()
      
        return asu        
  
  
    def build(self):
        params_in, kwargs=self.get_params()
        asu=Single_units_activity(self.threads, self.start, self.stop, **kwargs)
        asu.calibrate() 
        asu.par.update(params_in)
        asu.inputs()
        asu.build()
        if len(self.rand_params):
            asu.randomize_params(self.rand_params)
    
        return asu  
class Fmin(object):
    
    def __init__(self, label, obj_model_list, start, stop, **kwargs):
        
        self.label=label
        self.obj_model_list=copy.deepcopy(obj_model_list)
        self.obj_model_dic={}
        for mo in self.obj_model_list:
            mo.start=start
            mo.stop=stop
            self.obj_model_dic[mo.label]=mo
        self.xopt=[]
        
        self.x0=[]
        self.x_code_book=[]
        
        for mo in self.obj_model_list:
            
            for key, x0 in mo.variables_key_x0:
                self.x0.append(x0)
                self.x_code_book.append([mo.label, key])
                
            mo.set_rand(kwargs)

               
        
    def add_model(self, lesion_setup, setup, i, threads):
        if len(setup)==5:setup.append({})
        label, flag, variables, start, stop, kwargs=setup
        self.obj_model_list.append(Activity_model_obj(label, flag, variables, start, stop, threads, lesion_setup, **kwargs))
        #self.fmin_variables_set(self.xopt)
    
    def fmin(self, load, save_at):
 
        if not load:
            t=time.time()
            [xopt,fopt, iter, funcalls , warnflag, allvecs] = opt.fmin(self.fmin_error_fun, self.x0, args=(), maxiter=20, maxfun=20,full_output=1, retall=1)
            t=int(time.time()-t)
            s='time opt={0:6} xopt={1:10} fopt={2:6} iter={3:6} funcalls={4:6} warnflag={5:2}'
            s=s.format(str(t), str(xopt), str(fopt), str(iter), str(funcalls), str(warnflag))
            data_handling.pickle_save([s, xopt, fopt, iter, funcalls , warnflag, allvecs], save_at)
        else:
            [s, xopt,fopt, iter, funcalls , warnflag, allvecs]=data_handling.pickle_load(save_at)        
        self.xopt=xopt
        self.fmin_xopt_to_all_set(xopt)
        return s
        
    def fmin_error_fun(self, x, *arg):
        
        e=0    
        self.fmin_variables_set(x)

        for mo in self.obj_model_list:         
            if mo.variables=={}: continue   
            asu=mo.simulate(record_voltage=False)

            
            for target_path_list, method, args, kwargs in mo.variables_eval: 
                target_value=misc.dict_recursive_get(asu.par.dic, target_path_list)
                
                func=getattr(asu, method)
                e+=(target_value-func(*args, **kwargs))**2
        '''
        d=func(*args, **kwargs)
        d=my_nest.GetStatus(asu.units_dic['GPE_A-study'].population.ids, ['V_th'])
        d=[dd[0] for dd in d]   
        pylab.hist(d, bins=10)
        pylab.show()
        '''   
        print e, x
        return e
    
    def fmin_variables_set(self, x):
        
        for x_code_book, val  in zip(self.x_code_book, x):
            label=x_code_book[0]
            variable_param=x_code_book[1]
            mo = self.obj_model_dic[label]
            misc.dict_recursive_add(mo.variables, variable_param, val) 

    def fmin_xopt_to_all_set(self, x):
        
        for x_code_book, val  in zip(self.x_code_book, x):
            variable_param=x_code_book[1]
            for mo in self.obj_model_list:
                misc.dict_recursive_add(mo.variables, variable_param, val) 
            
                    
class Activity_model(object):
    
    def __init__(self, threads, lesion_setup, setup_list_models, setup_list_fmin):
         
        self.label_list=[] 
         
        self.obj_models={}
        self.obj_fmins={}
        for setup in setup_list_models:
            if len(setup)==6: setup.append({})
            label, flag, variables, start, stop, input_name, kwargs=setup
            self.label_list.append(label)
            self.obj_models[label]=Activity_model_obj(label, flag, variables, start, stop, threads,  lesion_setup, input_name, **kwargs)
        
        for setup in setup_list_fmin:
            if len(setup)==4: setup.append({})
            label_fmin, label_list_models, start, stop, kwargs=setup
            obj_model_list=[]
            for label in label_list_models:
                obj_model_list.append(copy.deepcopy(self.obj_models[label]))
                
            self.obj_fmins[label_fmin]=Fmin(label_fmin, obj_model_list, start, stop, **kwargs)

        self.lesion_setup=lesion_setup
        self.path_data=Single_units_activity().path_data
        self.path_pictures=Single_units_activity().path_pictures
        self.sname=sys.argv[0].split('/')[-1].split('.')[0]
        
        self.threads=threads
    
    def add_model_fmins(self, model, setup):
        self.obj_fmins[model].add_model(self.lesion_setup, setup, len(self.obj_models.keys()), self.threads)
    
    def get_models(self, labels):    
        
        if isinstance(labels, str) :
            labels=[labels]
        save_labels=[]
        model_list=[]
        for label in labels:
            
            if label in self.obj_models.keys():
                model_list.append(self.obj_models[label])
                save_labels.append(label)
            else:
                for mo in self.obj_fmins[label].obj_model_list:
                    model_list.append(mo)
                    save_labels.append(label+'-'+mo.label)
                
        
        return model_list, save_labels
        
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
        
        model_list, save_labels=self.get_models(labels)
        if not len(linestyle):linestyle=['-']*len(model_list)
        for i, mo in enumerate(model_list):
            d=mo.data_input_output
            ax.plot(d[0], d[1], colors[i], linestyle=linestyle[i])
            ax.text( coords[i][0], coords[i][1], mo.label, transform=ax.transAxes, 
                     fontsize=pylab.rcParams['font.size'], backgroundcolor = 'w', **{'color': colors[i]})
        
        pylab.setp(ax.lines, linewidth=2.0) # Need to pu ti before  
        ax.set_ylabel('Rate MSN (spikes/s)') 
        ax.set_xlabel('Cortical input (events/s)')
        
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
    def find_params(self, loads, labels_fmin):
        
        
        for load, label_fmin in zip(loads, labels_fmin):
            save_at=self.path_data+self.sname+'/find_param'+label_fmin              
            save_log_at=self.path_data+self.sname+'/find_param.log'  

            s=self.obj_fmins[label_fmin].fmin(load, save_at)
            data_handling.txt_save_to_label(s, label_fmin, save_log_at )
            
            
        
    def simulate_example(self, loads, labels,  rand_setup={}):

        for load, label in zip(loads, labels):
            model_list, save_labels=self.get_models(label)
            
            for mo, sv_label in zip(model_list, save_labels):
                save_at=self.path_data+self.sname+'/simulate_example-'+sv_label
                if not load:
    
                    if rand_setup: mo.set_rand(rand_setup)
                        
                    asu=mo.simulate(record_voltage=True)     
                    asu.get_spikes(asu.study_name)
                    mo.data_mean_rate=asu.get_mean_rate(asu.study_name)
                    mo.data_example=asu.get_voltage_trace(asu.study_name)
                    
                    data_handling.pickle_save([mo.data_example, mo.data_mean_rate], save_at)      
                
                else:
                    mo.data_example, mo.data_mean_rate=data_handling.pickle_load(save_at)
    
    def simulate_example_mm(self, loads, labels,  rand_setup={}, mm_types=['I']):

        for load, label in zip(loads, labels):
            model_list, save_labels=self.get_models(label)
            
            for mo, sv_label in zip(model_list, save_labels):
                save_at=self.path_data+self.sname+'/simulate_example_synaptic_currents-'+sv_label
                if not load:
    
                    if rand_setup: mo.set_rand(rand_setup)
                        
                    asu=mo.simulate(record_mm=mm_types)     
                    
                    mo.data_example_mm=asu.get_mm(asu.study_name, mm_types)
                    
                    data_handling.pickle_save([mo.data_example_mm], save_at)      
                
                else:
                    mo.data_example_mm=data_handling.pickle_load(save_at)      
                                       
    def simulate_input_output(self, loads, labels, input_rates, input_names='', rand_setup={}):
        
        if not isinstance(input_names, list):
            input_names=[input_names]
        if len(input_names)==1:
            input_names=input_names*len(labels)
            
           
        for input_name, load, label in zip(input_names, loads, labels):
            
            model_list, save_labels=self.get_models(label)
            
            for  mo, sv_label in zip( model_list, save_labels):
                save_at=self.path_data+self.sname+'/simulate_input_vs_model-'+sv_label
     
                if not load:
                    output_rates=[]
    
                    for r in input_rates:
                        if not input_name: input_name=mo.input_name
                        misc.dict_recursive_add(mo.variables, ['node', input_name, 'rate'], float(r))
                        
                        if rand_setup: mo.set_rand(rand_setup)

                        
                        asu=mo.simulate(record_voltage=False)
                        mr=asu.get_spike_statistics(asu.study_name)[0]
                        output_rates.append(mr)
                        
                    data_handling.pickle_save([input_rates, output_rates], save_at)
                else:
                    input_rates, output_rates=data_handling.pickle_load(save_at)
                mo.data_input_output=[input_rates, output_rates]
            

    def simulate_variable_population(self, loads, labels, rand_setup={}):
        
        for load, label in zip(loads, labels):
            model_list, save_labels=self.get_models(label)
            
            for mo, sv_label in zip(model_list, save_labels):
                
                save_at=self.path_data+self.sname+'/simulate_variable_population-'+sv_label
                
                if rand_setup: mo.set_rand(rand_setup)
                
                if not load:  
    
                    asu=mo.simulate()  
                    mrs=asu.get_mean_rates(asu.study_name)
                    data_handling.pickle_save(mrs, save_at)
                    
                else: 
                    mrs=data_handling.pickle_load(save_at)
                    
                mo.data_variable_population=mrs
            
    def rheobase_variable_population(self, loads, models, rand_setup={}):
        save_at=self.path_data+self.sname+'/rheobase_variable_population'
        for load, model in zip(loads, models):
            mo=self.obj_models[model]
            save_at=save_at+'-'+mo.label
            if rand_setup: mo.set_rand(rand_setup)
            if not load:  

                asu=mo.build()

                params=my_nest.GetStatus(asu.units_dic[mo.study_name].population.ids, ['k','E_L','V_th','b_2','V_b', 'p_2','C_m'])  

                v_rheo=[]
                I_rheo=[]
                for p in params:
                    k, E_L, V_th, b_2, V_b, p_2, C_m=p
                    f=lambda V:-( k*(V-V_th)*(V-E_L)-b_2*(V-V_b)**p_2)
                    f_v_rheo=lambda V :((2*V-V_th-E_L)*k-(p_2*b_2*(V-V_b)**(p_2-1)))**2
                    [xopt,fopt, iter, funcalls , warnflag, allvecs] = opt.fmin(f_v_rheo, -60, args=(), maxiter=200, maxfun=200,full_output=1, retall=1)
                    v_rheo.append(xopt)
                    I_rheo.append(abs(f(xopt[0])))
                    
                data_handling.pickle_save(I_rheo, save_at)
                
            else: 
                I_rheo=data_handling.pickle_load(save_at)
                
            mo.data_rheobase_variable_population=I_rheo
            
    def show(self, labels_models, labels_fmin, labels_variable_pop):
        
        fig, ax_list=plot_settings.get_figure(self, n_rows=3, n_cols=3, 
                                              w=1000.0, h=800.0, fontsize=12)
        
        colors=['g','b','r', 'm', 'c', 'k']
        coords=[[0.05, 0.9-i*0.1] for i in range(len(colors))]
        #self.plot_example_models(ax_list[0], [labels_models[0]], colors, coords)
        self.plot_input_output(ax_list[0], labels_models[0:6], colors, coords)
        self.plot_example_fmin(ax_list[1], labels_fmin, colors, coords)
        self.plot_variable_population(ax_list[2], labels_variable_pop[0:4], colors, coords)
        self.plot_variable_population(ax_list[3], labels_variable_pop[4:8], colors, coords)
        
        return fig


def main():
        
    pass
if __name__ == "__main__":
    main()
