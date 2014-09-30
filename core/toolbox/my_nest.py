'''
Mikael Lindahl 2010


Module:
mynest

Here my own nest functions can be defined. For example connection functions
setting random weight or delay on connections.


'''



# Imports
import numpy
import numpy.random as rand

import socket
HOST=socket.gethostname().split('.')[0]
# if 'milner-login1.pdc.kth.se'==HOST:
# if 0:
from nest import *
import nest
    
    
import time
from copy import deepcopy
from toolbox.parallelization import comm, Barrier


def _Simulate_mpi(*args, **kwargs):
    with Barrier():
        nest.Simulate(*args, **kwargs)
        
def Simulate(*args, **kwargs):
    if comm.is_mpi_used():
        _Simulate_mpi(*args, **kwargs)    
    else:
        nest.Simulate(*args, **kwargs)   

def _Create_mpi(*args, **kwargs):
    with Barrier():
        return nest.Create(*args, **kwargs)
        
def Create(*args, **kwargs):
    if comm.is_mpi_used():
        return _Create_mpi(*args, **kwargs)    
    else:
        return nest.Create(*args, **kwargs)  


def Connect(*args, **kwargs):
    if comm.is_mpi_used():
        _Connect_mpi(*args, **kwargs)
    else:
        _Connect(*args, **kwargs)
        

def _Connect_mpi(*args, **kwargs):
    with Barrier():
        _Connect(*args, **kwargs)



        

def _Connect(pre, post, *args, **kwargs):
    
    if hasattr(nest, 'OneToOneConnect'):
        nest.OneToOneConnect(pre, post,  *args, **kwargs)
    else:
        nest.Connect(pre, post,  *args, **kwargs)
 
#         if params:
#             syn_list=[{'model':model,
#                        'delay':delay,
#                        'weight':weight} 
#                       for delay, weight in zip(delay, params)]
#         else:
#             syn_list=[{'model':model}]*len(pre)
#         conn_list=[None]*len(pre)
#         
#         
#         pre=[[v] for v in pre] 
#         post=[[v] for v in post]
#         map(nest.Connect, pre, post, conn_list, syn_list) 
#         for source, target, syn_dict in zip(pre,  post, syn_list):
#             nest.Connect([source], [target], syn_spec=syn_dict)

def C(pre, post, params = None, delay = None, model = "static_synapse" ):    
    
    '''
    As NEST Connect, Make one-to-one connections of type model between 
    the nodes in pre and the nodes in post. pre and post have to be lists 
    of the same length. If params is given (as dictionary or list of
    dictionaries), they are used as parameters for the connections. If
    params is given as a single float or as list of floats, it is used
    as weight(s), in which case delay also has to be given as float or
    as list of floats.
    '''
    
    if isinstance(model, str): model = [ model ]  
    if isinstance(pre, int):   pre   = [ pre   ]                                # since NEST needs list objects
    if isinstance(post, int):  post  = [ post  ]                                # since NEST needs list objects
     
    Connect(pre, post, params = params, delay = delay, model = model[ 0 ] ) 
    
    if len( model ) > 1:                                                        # pre -> post with model[1], model[2] etc  
        for m in model[1:]:
            Connect(pre, post, params = params, delay = delay, model = m ) 
                
def CC( pre,  post, 
        params = { 'd_mu'  : [], 'w_mu'  : [], 'd_rel_std' : 0.0, 'w_rel_std' : 0.0 }, 
        model = [ 'static_synapse' ] ):
    '''
    As NEST ConvergentConnect except that it can take severals models and make same
    connections for both models. This can be used to connect same source with 
    target with both AMPA and NMDA. Mean and standard deviation for delays and
    weights can be given. Mean as a list of values one for each model and
    standard deviation as single values applied to all synapse models.
     
    Inputs:
        pre                      - list of source ids       
        post                     - list of target ids  
    	params[ 'd_mu' ]         - mean delay for each model
    	params[ 'w_mu' ]       	 - mean weight for each model
        params[ 'd_rel_std' ]    - relative mean standard deviation delays
        params[ 'w_rel_std' ]    - relative mean standard deviation weights
        model                    - synapse model, can be a list of models
    '''
    
    for key, val in params.iteritems():
        exec '%s = %s' % (key, str(val))                                        # params dictionary entries as variables
    
    
    if isinstance(model, str): model = [model]                                  # if model is a string put into a list   
    if isinstance(pre, int):   pre   = [ pre   ]                                # since NEST needs list objects
    else:                      pre   = pre[ : ]                                 # else retreive list
    if isinstance(post, int):  post  = [ post  ]                                # since NEST needs list objects
    else:                      post   = post[ : ]                               # else retreive list
    
        
    n = len( pre )
    
    d_mu=params['d_mu']
    d_rel_std=params['d_rel_std']
    w_mu=params['w_mu']
    w_rel_std=params['w_rel_std']   
    if not d_mu: d_mu = [ GetDefaults( m )[ 'delay' ] for m in model ]     # get mean delays d_mu
    d_sigma  = [ d_rel_std*mu for mu in d_mu ]                                  # get delay standard deviation
    
    if not w_mu: w_mu = [ GetDefaults( m )[ 'weight' ] for m in model ]    # get mean weights w_mu   
    w_sigma = [ w_rel_std*mu for mu in w_mu ]                                   # get weight standard deviation            
    
    delays  = [ rand.normal( mu, sigma, [ len( pre ), n ] )                     # calculate delays, randomized if sigma != 0
               if sigma else numpy.ones( ( len( pre ), n ) )*mu           
    		   for mu, sigma in zip( d_mu, d_sigma )   ]                                                                
      
    weights = [ rand.normal( mu, sigma, [ len( pre ), n ] )                     # calculate weights, randomized if sigma != 0
               if sigma else numpy.ones( ( len( pre ), n ) )*mu           
    		   for mu, sigma in zip( w_mu, w_sigma ) ]                                                                  
    
    
    for i, post_id in enumerate( post ):                                        # connect with model[0]                                                                            
        d = [ dj for dj in delays[  0 ][ i ] ] 
        w = [ wj for wj in weights[ 0 ][ i ] ]
        ConvergentConnect(  pre,  [ post_id ], weight = w , delay = d, 
                                 model = model[ 0 ] )                        
          
        if len( model ) > 1:                                                    # pre -> post with model[1], model[2] etc  
            for m, dlays, wghts, in zip( model[1:], delays[1:], weights[1:] ):
            	d = [ dj for dj in dlays[ i ] ]
                w = [ wj for wj in wghts[ i ] ]
            	ConvergentConnect( pre, [ post_id ], weight = w, 
                                        delay = d, model = m )                   
    
    
# def Create(*args, **kwargs):
#     try:
#         return nest.Create(*args, **kwargs)
#     except Exception as e:
#         s='\nargs:{} \nkwargs:{} \nAvailable models: \n{}'.format(args, 
#                                                            kwargs, 
#                                                            nest.Models())
#         if len(e.message)>40:
#             e.message=e.message[0:40]+'\n'+e.message[40:]
#         raise type(e)(e.message + s), None, sys.exc_info()[2]
        
def GetConn(soruces, targets):    
    c=[]
    
    for s in soruces:
        for t in targets:
            c.extend(nest.GetStatus(nest.FindConnections([s], [t])))     
    return c

def get_default_module_paths(home):

    if nest.version()=='NEST 2.2.2':
        s='nest-2.2.2'
    if nest.version()=='NEST 2.4.1':
        s='nest-2.4.1'    
    if nest.version()=='NEST 2.4.2':
        s='nest-2.4.2'   
          
    path= (home+'/opt/NEST/module/'
                  +'install-module-130701-'+s+'/lib/nest/ml_module')
    sli_path =(home+'/opt/NEST/module/'
                      +'install-module-130701-'+s+'/share/ml_module/sli')
    
    return path, sli_path

def install_module(path, sli_path, model_to_exist='my_aeif_cond_exp'):
    
    
    if not model_to_exist in nest.Models(): 
        nest.sr('('+sli_path+') addpath')
        nest.Install(path)
        

def GetConnProp(soruces, targets, prop_name, time):
    c=GetConn(soruces, targets)                                        
    x,y={},{}
    
    for conn in c:
        model=conn['synapse_model']
        
        if not model in y.keys():
            y[model]=[]
            x[model]=[]
            
        if prop_name in conn.keys():
            y[model].append(conn[prop_name])
        else:
            y[model].append(numpy.NaN)
            
        x[model].append(time)
    return x,y

# def GetDopamine(soruces, targets):
#     c=GetConn(soruces, targets)                                                 
#     w=[]
#     for conn in c:
#         w.append(conn['n'])
#     return w
# 
#     return w
    
def DC( pre, post,
        params = { 'd_mu'  : [], 'w_mu'  : [], 'd_rel_std' : 0.0, 'w_rel_std' : 0.0 }, 
        model = [ 'static_synapse' ] ):        
    '''
    As NEST DivergentConnect except that it can take severals models and make same
    connections for both models. This can be used to connect same source with 
    target with both AMPA and NMDA.Mean and standard deviation for delays and
    weights can be given. Mean as a list of values one for each model and
    standard deviation as single values applied to all synapse models.
     
    Inputs:
        pre                      - list of source ids       
        post                     - list of target ids  
        params[ 'd_mu' ]         - mean delay for each model
        params[ 'w_mu' ]            - mean weight for each model
        params[ 'd_rel_std' ]    - relative mean standard deviation delays
        params[ 'w_rel_std' ]    - relative mean standard deviation weights
        model                    - synapse model, can be a list of models
    '''
  
    for key, val in params.iteritems():
        exec '%s = %s' % (key, str(val))                                        # params dictionary entries as variables  
  
    if isinstance(model, str): model = [model]                                  # if model is a string put into a list   
    if isinstance(pre, int):   pre   = [ pre   ]                                # since NEST needs list objects
    else:                      pre   = pre[ : ]                                 # else retreive list
    if isinstance(post, int):  post  = [ post  ]                                # since NEST needs list objects
    else:                      post   = post[ : ]                               # else retreive list

    n = len( post )    
    
    d_mu=params['d_mu']
    d_rel_std=params['d_rel_std']
    w_mu=params['w_mu']
    w_rel_std=params['w_rel_std']  
    
    if not d_mu: d_mu =  [GetDefaults( m )[ 'delay' ] for m in model ]     # get mean delays d_mu
    d_sigma  = [ d_rel_std*mu for mu in d_mu ]                                  # get delay standard deviation
    
    if not w_mu: w_mu = [ GetDefaults( m )[ 'weight' ] for m in model ]    # get mean weights w_mu   
    w_sigma = [ w_rel_std*mu for mu in w_mu ]                                   # get weight standard deviation            
    
    delays  = [ rand.normal( mu, sigma, [ len( pre ), n ] )                     # calculate delays, randomized if sigma != 0
               if sigma else numpy.ones( ( len( pre ), n ) )*mu          
			   for mu, sigma in zip( d_mu, d_sigma )   ]                                                                
      
    weights = [ rand.normal( mu, sigma, [ len( pre ), n ] )                     # calculate weights, randomized if sigma != 0
               if sigma else numpy.ones( ( len( pre ), n ) )*mu           
			   for mu, sigma in zip( w_mu, w_sigma ) ]                                                                  

    
    for i, pre_id in enumerate( pre ):                                          # connect with model[0                                    
        d = [ dj for dj in delays[  0 ][ i ] ] 
        w = [ wj for wj in weights[ 0 ][ i ] ]
        DivergentConnect( [ pre_id  ], post, weight = w , delay = d, 
                               model = model[ 0 ] )                          
               
        if len( model ) > 1:                                                    # pre -> post with model[1], model[2] etc   
            for m, dlays, wghts, in zip( model[1:], delays[1:], weights[1:] ):
            	d = [ dj for dj in dlays[ i ] ]
                w = [ wj for wj in wghts[ i ] ]
            	DivergentConnect( [ pre_id  ], post, weight = w, 
                                       delay = d, model = m )                           
             
def RDC(pre, post, n, 
        params = { 'd_mu'  : [], 'w_mu'  : [], 'd_rel_std' : 0.0, 'w_rel_std' : 0.0 },  
        model = 'static_synapse' ):
    '''
    As NEST RandomDivergentConnect, except it can take severals models and 
    make same connections for both models. This can be used to connect same
    source with  target with both AMPA and NMDA.Mean and standard deviation 
    for delays and weights can be given. Mean as a list of values one for 
    each model and standard deviation as single values applied to all 
    synapse models.
     
    Inputs:
        pre                      - list of source ids       
        post                     - list of target ids  
        n                        - number of neurons each source neuron connects to
        params[ 'd_mu' ]         - mean delay for each model
        params[ 'w_mu' ]            - mean weight for each model
        params[ 'd_rel_std' ]    - relative mean standard deviation delays
        params[ 'w_rel_std' ]    - relative mean standard deviation weights
        model                    - synapse model, can be a list of models
    '''

    rn = rand.normal                                                            # copy function for generation of normal distributed random numbers

    for key, val in params.iteritems():
        exec '%s = %s' % (key, str(val))                                        # params dictionary entries as variables
    
    if isinstance(model, str): model = [ model ]                                # if model is a string put into a list   
    
    d_mu=params['d_mu']
    d_rel_std=params['d_rel_std']
    w_mu=params['w_mu']
    w_rel_std=params['w_rel_std']  
    
    if not d_mu: d_mu = [GetDefaults( m )[ 'delay' ] for m in model ]      # get mean delays d_mu
    d_sigma  = [ d_rel_std*mu for mu in d_mu ]                                  # get delay standard deviation
    
    if not w_mu: w_mu = [GetDefaults( m )[ 'weight' ] for m in model ]     # get mean weights w_mu   
    w_sigma = [ w_rel_std*mu for mu in w_mu ]                                   # get weight standard deviation            
    
    delays  = [ rand.normal( mu, sigma, [ len( pre ), n ] )                     # calculate delays, randomized if sigma != 0
               if sigma else numpy.ones( ( len( pre ), n ) )*mu           
			   for mu, sigma in zip( d_mu, d_sigma )   ]                                                                
      
    weights = [ rand.normal( mu, sigma, [ len( pre ), n ] )                     # calculate weights, randomized if sigma != 0
               if sigma else numpy.ones( ( len( pre ), n ) )*mu            
			   for mu, sigma in zip( w_mu, w_sigma ) ]                                                                  
     
    for i, pre_id in enumerate( pre ):                                                                              
        j = 0
        
         
        if d_sigma[ j ]: d = rn( d_mu[ j ], d_sigma[ j ], [ 1, n ] )[ 0 ]       # if sigma len( targets ) randomized delays                               
        else:  d = numpy.ones( ( 1, n ) )[ 0 ]*d_mu[ j ]                        # else no randomized delays  
                                  
        if w_sigma[ j ]: w = rn( w_mu[ j ], w_sigma[ j ], [ 1, n ] )[ 0 ]       # if signa len( targets ) randomized weights                              
        else: w = numpy.ones( ( 1, n ) )[ 0 ]*w_mu[ j ]                         # else no randomized weights
             
        d, w = list( d ), list( w )                                             # as lists             
        RandomDivergentConnect( [ pre_id ], post, n, weight = w ,          # connect with model[0]
                                     delay = d, model = model[ 0 ] )                  
        
        if len( model ) > 1:
            targets = [ conn['target'] for conn in                              # find connections made by RandomDivergentConnect 
                       GetStatus(FindConnections( [ pre_id ] ) ) 
                       if conn[ 'synapse_type' ] == model[ 0 ]]                                                           
            nt      = len( targets )
            
            for j, m in enumerate( model[ 1 : ], start = 1 ):                   # pre -> post with model[1], model[2] etc  
                if d_sigma[ j ]: d = rn( d_mu[ j ], d_sigma[ j ], [ 1, nt ] )[ 0 ] # if sigma len( targets ) randomized delays 
                else:            d = numpy.ones( ( 1, nt ) )[ 0 ]*d_mu[ j ]     # else no randomized delays   
                                         
                if w_sigma[ j ]: w =rn( w_mu[ j ], w_sigma[ j ], [ 1, nt ] )[ 0 ]  # if signa len( targets ) randomized delays 
                else:            w = numpy.ones( ( 1, nt ) )[ 0 ]*w_mu[ j ]      # else not   
                
                d, w = list( d ), list( w )         
                RandomDivergentConnect( [ pre_id ], targets, n, weight = w, 
                                       delay = d, model = m )                           
                            
def RCC(pre, post, n, 
        params = { 'd_mu'  : [], 'w_mu'  : [], 'd_rel_std' : 0.0, 'w_rel_std' : 0.0 },  
        model = 'static_synapse' ):
    '''
    As NEST RandomDivergentConnect, except it can take severals models and 
    make same connections for both models. This can be used to connect same
    source with  target with both AMPA and NMDA.Mean and standard deviation 
    for delays and weights can be given. Mean as a list of values one for 
    each model and standard deviation as single values applied to all 
    synapse models.
     
    Inputs:
        pre                      - list of source ids       
        post                     - list of target ids  
        n                        - number of neurons each source neuron connects to
        params[ 'd_mu' ]         - mean delay for each model
        params[ 'w_mu' ]            - mean weight for each model
        params[ 'd_rel_std' ]    - relative mean standard deviation delays
        params[ 'w_rel_std' ]    - relative mean standard deviation weights
        model                    - synapse model, can be a list of models
    '''

    rn = rand.normal                                                            # copy function for generation of normal distributed random numbers

    for key, val in params.iteritems():
        exec '%s = %s' % (key, str(val))                                        # params dictionary entries as variables
    
    if isinstance(model, str): model = [ model ]                                # if model is a string put into a list   
    d_mu=params['d_mu']
    d_rel_std=params['d_rel_std']
    w_mu=params['w_mu']
    w_rel_std=params['w_rel_std']  
    if not d_mu: d_mu = [GetDefaults( m )[ 'delay' ] for m in model ]      # get mean delays d_mu
    d_sigma  = [ d_rel_std*mu for mu in d_mu ]                                  # get delay standard deviation
    
    if not w_mu: w_mu = [GetDefaults( m )[ 'weight' ] for m in model ]     # get mean weights w_mu   
    w_sigma = [ w_rel_std*mu for mu in w_mu ]                                   # get weight standard deviation            
    
    delays  = [ rand.normal( mu, sigma, [ len( pre ), n ] )                     # calculate delays, randomized if sigma != 0
               if sigma else numpy.ones( ( len( pre ), n ) )*mu           
               for mu, sigma in zip( d_mu, d_sigma )   ]                                                                
      
    weights = [ rand.normal( mu, sigma, [ len( pre ), n ] )                     # calculate weights, randomized if sigma != 0
               if sigma else numpy.ones( ( len( pre ), n ) )*mu            
               for mu, sigma in zip( w_mu, w_sigma ) ]                                                                  
     
    for i, pre_id in enumerate( pre ):                                                                              
        j = 0
        
         
        if d_sigma[ j ]: d = rn( d_mu[ j ], d_sigma[ j ], [ 1, n ] )[ 0 ]       # if sigma len( targets ) randomized delays                               
        else:  d = numpy.ones( ( 1, n ) )[ 0 ]*d_mu[ j ]                        # else no randomized delays  
                                  
        if w_sigma[ j ]: w = rn( w_mu[ j ], w_sigma[ j ], [ 1, n ] )[ 0 ]       # if signa len( targets ) randomized weights                              
        else: w = numpy.ones( ( 1, n ) )[ 0 ]*w_mu[ j ]                         # else no randomized weights
             
        d, w = list( d ), list( w )                                             # as lists             
        RandomDivergentConnect( [ pre_id ], post, n, weight = w ,          # connect with model[0]
                                     delay = d, model = model[ 0 ] )                  
        
        if len( model ) > 1:
            targets = [ conn['target'] for conn in                              # find connections made by RandomDivergentConnect 
                       GetStatus(FindConnections( [ pre_id ] ) ) 
                       if conn[ 'synapse_type' ] == model[ 0 ]]                                                           
            nt      = len( targets )
            
            for j, m in enumerate( model[ 1 : ], start = 1 ):                   # pre -> post with model[1], model[2] etc  
                if d_sigma[ j ]: d = rn( d_mu[ j ], d_sigma[ j ], [ 1, nt ] )[ 0 ] # if sigma len( targets ) randomized delays 
                else:            d = numpy.ones( ( 1, nt ) )[ 0 ]*d_mu[ j ]     # else no randomized delays   
                                         
                if w_sigma[ j ]: w =rn( w_mu[ j ], w_sigma[ j ], [ 1, nt ] )[ 0 ]  # if signa len( targets ) randomized delays 
                else:            w = numpy.ones( ( 1, nt ) )[ 0 ]*w_mu[ j ]      # else not   
                
                d, w = list( d ), list( w )         
                ConvergentConnect( [ pre_id ], targets, weight = w, 
                                       delay = d, model = m )                  

def MyLoadModels( model_setup, models ):
    '''
    Input
        model_setup - list with tuples (base model, new model name, parameters)
                      or ddictionary with  keys  new model name and values
                      tuples  (base model, new model name, parameters)
        models     - new name of models in models to load into nest
    '''  
    if type(model_setup) in [list, tuple]:  
        for setup in model_setup: 
            if setup[ 1 ] in models:
                CopyModel( setup[ 0 ], setup[ 1 ], setup[ 2 ] )   # Create model
    elif type(model_setup)==dict:
        for model in models: 
            setup=model_setup[model]
            if not setup[ 1 ] in nest.Models():
                CopyModel( setup[ 0 ], setup[ 1 ], setup[ 2 ] )   # Create model
              

def MyCopyModel(params, new_name):
    
    params=deepcopy(params)
    type_id=params['type_id']
    del params['type_id']
    if not new_name in nest.Models():
        CopyModel( type_id, new_name, params ) 
    
        
def MySimulate(duration):

    
    start = time.time() 
    Simulate( duration )
    stop = time.time() 
    
    s = stop - start
    m = s // 60
    s = s - m*60
    print 'Rank %i simulation time: %i minutes, %i seconds' % ( Rank(), m, s )    
            
def ResetKernel(threads=1, print_time=False, display=True):

    
    if display:
        print 'Reseting kernel'
    nest.ResetKernel()
    #nest.SetKernelStatus({"resolution": .1, "print_time": print_time})
    nest.SetKernelStatus({ "print_time": print_time})
    
    if comm.is_mpi_used():
        nest.SetKernelStatus({"local_num_threads":  1})
    else:    
        nest.SetKernelStatus({"local_num_threads":  threads})
