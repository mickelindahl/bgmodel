'''
Mikael Lindahl 2010


Module:
misc


'''

# Imports
import collections
import matplotlib
import numpy
import pylab

from NeuroTools.stgen import StGen
from numpy import array, concatenate
import scipy.signal as signal
from copy import deepcopy

import scipy.cluster.vq as clust

import random
import itertools
import numpy as np
import time
import sys,os

class my_slice(object):
    # can do set manipulations with these
    def __init__(self, *a, **k):
        self.slice=slice(*a,**k)
    
    def __repr__(self):
        return (self.__class__.__name__
                +'({},{},{})'.format(self.slice.start, 
                                     self.slice.stop,
                                     self.slice.step))

    
    @property    
    def start(self):
        return self.slice.start
    
    @property    
    def stop(self):
        return self.slice.stop
   
    @property    
    def step(self):
        return self.slice.step
    

    def __hash__(self):
        return self.stop+self.start+self.step
    
    def __eq__(self, other):
        return self.__hash__()==other.__hash__()    
    
    def get_slice(self):
        return self.slice

class Stopwatch():
    def __init__(self, *args):
        self.msg = args[0]
        self.args=args
        self.time=None
    def __enter__(self):
        print self.msg,
        self.time=time.time()
    def __exit__(self, type, value, traceback):
        t=round(time.time()-self.time,)
        print ' {} sec'.format(t)
        if len(self.args)>1:
            self.args[1][self.msg]=t
            
class Stop_stdout():
    def __init__(self, flag):
        self.flag = flag
        self.stdout=None
    
    def __enter__(self):
        if self.flag:
            f = open(os.devnull, 'w')
            self.stdout=sys.stdout
            sys.stdout = f
      
    def __exit__(self, type, value, traceback):
        if self.flag:
            sys.stdout.close()
            sys.stdout=self.stdout
        
def adjust_limit(limit, percent_x=0.03, percent_y=0.04):
    ''' Adjust limit by withdrawing percent of diff from start and
        adding percent of diff to start.  
    
        Input:
            limit      - the limit to manipulate
            percent    - percent margin to add
    
    '''
    
    x=limit[0]
    y=limit[1]
    diff=y-x
    coords=[x-diff*percent_x, y+diff*percent_y]
    
    return coords
    
'''
Method for method to evenly split up a colormap into an RGB colors array
'''
def autocorrelation(spikes, bin=1, max_time=1000):
    N=len(spikes)
    
    N_order_isi=[]
    N_order_isi_histogram=[] # save isi histogram order 1 to n, see Perkel 1967
    top_bins=numpy.arange(1, max_time, bin)
    bottom_bins=numpy.arange(-max_time+1, 0, bin)
    middle_bins=array([bottom_bins[-1],0.001,0.001,top_bins[0]])
    bins=array([])
    bins=numpy.append(bins, bottom_bins)
    bins=numpy.append(bins, middle_bins)
    bins=numpy.append(bins, top_bins)
    isi_matrix=spikes-numpy.transpose(array([spikes]))*numpy.ones([1,N])
    
    # Zero order isi is on the diagonal, first on the first sub diagonal 
    # above an below the diagonal third on the next diagonals, and n 
    # order is the right top value and left bottom value.
    hist_autocorr,xaxis =numpy.histogram(isi_matrix, bins=bins, normed=False)
    
    # Set zero interval bin count to zero
    hist_autocorr[hist_autocorr==max(hist_autocorr)]=0
    
    hist_autocorr=hist_autocorr/float((N*bin))*1000
    
    xaxis=xaxis[0:-1]+bin/2
    
    return hist_autocorr, xaxis

def cluster_connections(pre_ids, post_ids, k, n_cluster):
        n_post=len(post_ids)
        partition=[int(v) for v in numpy.linspace(0,n_post, n_cluster+1)]
        intervals=zip(partition[:-1],partition[1:])
        
        sources=[]
        targets=[]
        
        for i in range(n_cluster):        
            tr=set(post_ids).difference(post_ids[intervals[i][0]:intervals[i][1]])    
            
            spids=pre_ids[intervals[i][0]:intervals[i][1]]
            
            tmp_tr=numpy.array(reduce(lambda x,y: list(x)+list(y), (tr,)*len(spids)))
            tmp_so=numpy.array(reduce(lambda x,y: list(x)+list(y),[(v,)*len(tr) for v in spids]))
            idx=numpy.nonzero(numpy.random.random(len(tmp_tr))<k)
            targets.extend(tmp_tr[idx])
            sources.extend(tmp_so[idx])
        
        return sources, targets

def convert2bin(raw_data, start, stop, clip=False, res=1):

    '''
    Convert a spike train from a time stamp representation to a 
    binned spike train
    
    Inputs:
        id       - id to vector with time stamps
        start    - first spike time
        stop     - last spike time
        clip     . clipping if more than one spike falls in abind
        
    Returns:
        st_binned  - vector that contains resulting binned vetor structure
        st_times   - vector of the resulting corresponding time stamps of
                     each bin
    
    remark: Spikes before start and stop are ignored
    '''
    id_vector=raw_data[1,:]
    spk_vector=raw_data[0,:]
    ids=numpy.unique(id_vector)
    
    ids=sorted(ids)
    output=numpy.zeros((len(ids), numpy.ceil( (stop-start)/res) + 1))
    
    for i, _id in enumerate(ids):
      
        st=spk_vector[id_vector==_id]
         
        validspikes=st[(st>start)*(st<stop)]
        
        if len(validspikes)!=0:
            if clip:
                for j in validspikes:
                    output[i,numpy.int_(numpy.round( (j-start)/res) )]=1
            else:
                for j in validspikes:
                    output[i,numpy.int_(numpy.round( (j-start)/res ))]+=1
                
    return output        
        

def convolve(*args, **kwargs):
    out=[]
    for arg in args:
        out.append(_convolve(arg, **kwargs))
    if len(out)==1:
        return out[0]
    else:
        return out

def _convolve(binned_data, bin_extent=1, kernel_type='triangle', axis=0,
              no_mean=False, params={}, **kwargs):
    ''' 
    Convolve data with a specific kernel. (Low pass filtering)
        Inputs:
            binned_data - binned spike count
            bin_extent  - extent of kernel
            kernel_type - kernel to filter by   
            no_mean     - subtract mean 
            
    '''
    if axis==1:
        binned_data=binned_data.transpose()
        
    binned_data=deepcopy(binned_data)
    
    if len(binned_data.shape)==1:
        single=True
    else:
        single=False
    
    if kernel_type=='triangle':
        kernel=numpy.arange(1,numpy.ceil(bin_extent/2.)+1)
        kernel=numpy.append(kernel, numpy.arange(numpy.floor(bin_extent/2.),0,-1))
        kernel=kernel/float(numpy.sum(kernel))
    
    if kernel_type=='rectangle':
        kernel=numpy.ones(bin_extent+1)
        #kernel=kernel/float(numpy.sum(kernel))
    
    if kernel_type=='gaussian':
        '''
        params['std_ms'] - standard deviation in ms
        params['fs']     - sampling frequency in hz
        '''
        kernel = signal.gaussian(bin_extent, params['std_ms']*params['fs']/1000.0)
    if len(kernel)>binned_data.shape[-1]:
        print 'j'   ,len(kernel),binned_data.shape[-1]
    assert len(kernel)<binned_data.shape[-1], 'kernel to big for data set'
    
    conv_data=[]
    import time
    if not single:
        if no_mean==True:
          
            for i in xrange(binned_data.shape[0]):
                binned_data[i, :] = np.convolve(binned_data[i,:], kernel/sum(kernel),'same')
                binned_data[i, :] -= np.mean(binned_data[i, :])

        else:
            for i in xrange(binned_data.shape[0]):
                binned_data[i, :] = np.convolve(binned_data[i,:], kernel/sum(kernel),'same')
             
            
        conv_data=binned_data      
    
    else:
        
        conv_data=numpy.convolve(kernel, binned_data, mode='same')     

    if axis==1:
        conv_data=conv_data.transpose()
    return conv_data


def dict_depth(d, depth=1):
    if not isinstance(d, dict) or not d:
        return depth
    return max(dict_depth(v, depth+1) for k, v in d.iteritems())

def dict_haskey(dic, keys):
        
        '''recursively merges dict's. not just simple a['key'] = b['key'], if
        both a and b have a key who's value is a dict then dict_merge is called
        on both values and the result stored in the returned dictionary.'''
        if len(keys)==0:
            return True
        
        #dic=deepcopy(dic)
        if keys[0] in dic.keys():
            val=dict_haskey(dic[keys[0]], keys[1:])
        else:
            val=False
        
        return val 

def dict_merge(a, b):
    '''recursively merges dict's. not just simple a['key'] = b['key'], if
    both a and b have a key who's value is a dict then dict_merge is called
    on both values and the result stored in the returned dictionary.'''
    if not isinstance(b, dict):
        return b
    result = deepcopy(a)
    for k, v in b.iteritems():
        if k in result and isinstance(result[k], dict):
                result[k] = dict_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result           

def dict_recursive_add(dic, keys, val):
        
        '''recursively merges dict's. not just simple a['key'] = b['key'], if
        both a and b have a key who's value is a dict then dict_merge is called
        on both values and the result stored in the returned dictionary.'''
        if len(keys)==0:
            return val
        
        #dic=deepcopy(dic)
        
        if keys[0] in dic.keys():
            if isinstance(dic[keys[0]], dict):
                dic[keys[0]].update(dict_recursive_add( dic[keys[0]],keys[1:], val))
            else:
                # Cope with updating existing parameter
                dic[keys[0]]=dict_recursive_add( dic[keys[0]] ,keys[1:], val)
        else:
            dic[keys[0]]={}
            dic[keys[0]]=dict_recursive_add(dic[keys[0]], keys[1:], val )
        
        return dic   

def dict_recursive_get(dic, keys):
        
        '''recursively merges dict's. not just simple a['key'] = b['key'], if
        both a and b have a key who's value is a dict then dict_merge is called
        on both values and the result stored in the returned dictionary.'''
        if len(keys)==0:
            return dic
        
        #dic=deepcopy(dic)
        if keys[0] in dic.keys():
            val=dict_recursive_get(dic[keys[0]], keys[1:])
        else:
            #val=None
            raise KeyError('key {} do not exist'.format(keys))
        return val   

def dict_apply_operation(d, keys, val, op='='):
        
    '''recursively merges dict's. not just simple a['key'] = b['key'], if
    both a and b have a key who's value is a dict then dict_merge is called
    on both values and the result stored in the returned dictionary.'''

    if len(keys)==0:
        if op=='=': 
            return val
        if op=='*':  
            return d*val
        if op=='+':  
            return d+val
    
    if keys[0] in d.keys():
        d[keys[0]]=dict_apply_operation(d[keys[0]], keys[1:], val, op )
    else:
        raise Exception('Entry '+keys[keys[0]]+' do not exist in  the dictionary')
    
    return d   



def dict_update(d, u, skip=False, skip_val=None, no_mapping_change=False):
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = dict_update(d.get(k, {}), v, skip, skip_val, no_mapping_change)
                    
            d[k] = r
        else:
            if skip and(d[k]==skip_val):
                pass

            elif no_mapping_change and (d=={}):
                #if d=={}:# or isinstance(d[k], collections.Mapping):
                pass
            else:
             #   try:
                    d[k] = u[k]
              #  except:
               #     print 'h'
    return d

def dict_reduce(dic_in, dic_out, s='', deliminator='.'):
    '''
    put dic_out={}
    '''
    
    
    if isinstance(dic_in, dict):
        s_pre=s
        for key in dic_in.keys():          
            s_post=s_pre+key+deliminator
            dic_out=dict_reduce(dic_in[key], dic_out, s_post, deliminator)
        return dic_out
    else: 
        s=s[:-1]
        dic_out.update({s:dic_in})  
        return dic_out   

def dict_slice(d, keys):
    m=map(lambda x:d[x], keys)
    return dict(zip(keys,m)) 


def dict_iter(d):
    stack = d.items()
    while stack:
        s=stack.pop()
        k=list(s[:-1])
        v=s[-1]
        if isinstance(v, dict):
            e=[tuple(k+[kk,vv]) for kk,vv in v.iteritems()]
            stack.extend(e)
        else:
            yield k,v

def sigmoid(p,x):
        x0,y0,c,k=p
        y = c / (1 + np.exp(-k*(x-x0))) + y0
        return y
    
def fit_sigmoid():
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.optimize


    def residuals(p,x,y):
        return y - sigmoid(p,x) 

    def resize(arr,lower=0.0,upper=1.0):
        arr=arr.copy()
        if lower>upper: lower,upper=upper,lower
        arr -= arr.min()
        arr *= (upper-lower)/arr.max()
        arr += lower
        return arr
    
def import_class(cl):
    d = cl.rfind(".")
    classname = cl[d+1:len(cl)]
    m = __import__(cl[0:d], globals(), locals(), [classname])
    return getattr(m, classname)   

def import_module(cl):
    d = cl.rfind(".")
    module = cl[d+1:len(cl)]
    m = __import__(cl[0:d], globals(), locals(), [module])
    return getattr(m, module)   
   
def inh_poisson_spikes(rate, t, t_stop, n_rep=1 ,seed=None):   
    '''
    Returns a SpikeTrain whose spikes are a realization of an inhomogeneous 
    poisson process (dynamic rate). The implementation uses the thinning 
    method, as presented in the references (see neurotools).
    
    From Professor David Heeger 2000 - "Poisson Model of Spike Generation" 
    
    Generating Poisson Spike Trains
    There are two commonly used procedures for numerically generating Poisson 
    spike trains. The first approach is based on the approximation in Eq. 2 
   
    P{1 spike during the interval(t-dt/2,t+dt/2)}=r(t)dt                     (2) 
    
    for the probability of a spike occurring during a short time interval. For 
    the homogeneous Poisson process, this expression can be rewritten
    (removing the time dependence) as
    
    P{1 spike during dt} ~ rdt
    
    This equation can be used to generate a Poisson spike train by first 
    subdividing time into a bunch of short intervals, each of duration dt. Then 
    generate a sequence of random numbers x[i] , uniformly distributed between 
    0 and 1. For each interval, if x[i] <=rdt, generate a spike. Otherwise, 
    no spike is generated. This procedure is appropriate only when dt is very 
    small, i.e, only when rdt<<1. Typically, dt = 1 msec should suffice. The 
    problem with this approach is that each spike is assigned a discrete 
    time bin, not a continuous time value.
    
    The second approach for generating a homogeneous Poisson spike train, 
    that circumvents this problem, is simply to choose interspike intervals 
    randomly from the exponential distribution. Each successive spike time is 
    given by the previous spike time plus the randomly drawn interspike 
    interval . Now each spike is assigned a continuous time value instead of a 
    discrete time bin. However, to do anything with the simulated spike train 
    (e.g., use it to provide synaptic input to another simulated neuron), it is
     usually much more convenient to discretely sample the spike train
    (e.g., in 1 msec bins), which makes this approach for generating the spike 
    times equivalent to the first approach described above.
    
    I use neruotools 
    
    
    Inputs:
             rate   - an array of the rates (Hz) where rate[i] is active on interval 
                     [t[i],t[i+1]] (if t=t.append(t_stop)
            t      - an array specifying the time bins (in milliseconds) at which to 
                     specify the rate
            t_stop - length of time to simulate process (in ms)
            n_rep  - number times to repeat spike pattern  
            seed   - seed random generator
    
    Returns: 
           spike  - array of spikes
     '''       

    times=[]
    rates=[]
    t=numpy.array(t)
    for i in range(n_rep):
        times=concatenate((times,t + t_stop*i/n_rep) ,1)
        rates=concatenate((rates,rate),1)

    stgen=StGen(seed=seed) 
    spikes=stgen.inh_poisson_generator(rate = rates, t=times, t_stop=t_stop, array=True)
    
    return spikes   

def kmean_cluster( data, k, iter = 10):
    '''
    The k-means algorithm takes as input the number of clusters to generate, 
    k, and a set of observation vectors to cluster. It returns a set of 
    centroids, one for each of the k clusters. An observation vector is
    classified with the cluster number or centroid index of the centroid  
    closest to it.
    
    Inputs:
        data          - A M by N array of M observations in N dimensions or a 
                         length M array of M one-dimensional observations.
        k             - The number of clusters to form as well as the number 
                         of centroids to generate. If minit initialization 
                         string is 'matrix', or if a ndarray is given instead, 
                         it is interpreted as initial cluster to use instead.
        iter          - Number of iterations of the k-means algrithm to run. 
                        Note that this differs in meaning from the iters 
                        parameter to the kmeans function.
    
    Returns: 
        centroids      - A k by N array of k centroids. The i'th centroid 
                        codebook[i] is represented with the code i. The 
                        centroids and codes     generated represent the lowest 
                        distortion seen, not necessarily the globally minimal
                         distortion.
        distortion    - The distortion between the observations passed and the 
                        centroids generated.
        labels          - A length N array holding the code book index for each 
                        observation.
        dist          - The distortion (distance) between the observation and 
                        its nearest code.
                       
    Examples:
        >>> from misc import kmean_cluster
        >>> features  = array([[ 1.9,2.3],
        ...                    [ 1.5,2.5],
        ...                    [ 0.8,0.6],
        ...                    [ 0.4,1.8],
        ...                    [ 0.1,0.1],
        ...                    [ 0.2,1.8],
        ...                    [ 2.0,0.5],
        ...                    [ 0.3,1.5],
        ...                    [ 1.0,1.0]])
        >>> kmean_cluster( features, 2, iter = 10):
        (array([[ 2.3110306 ,  2.86287398],
           [ 0.93218041,  1.24398691]]), 0.85684700941625547)
    '''

    centroids, labels = clust.kmeans2(data, k, minit ='points', iter = iter)

    #whitened = whiten(data)
    #centroids, distortion = kmeans( whitened, k, iter = iter)
    #labels, dist    =  vq( whitened, codebook )

    #return centroids, distortion, labels, dist   
    
    return centroids, labels
    
def kmean_image( data, code, times = False, ids = False, display=False, kwargs={} ):
        '''
        Plots the kmean data accoringly to clusters
        Inputs:
            data      - A M by N array of M observations in N dimensions or a 
                        length M array of M one-dimensional observations.
            code       - A length N array holding the code book index for each 
                        observation.
            times      - vector with the times of the sliding window
        '''
        
        if not display: ax = pylab.axes()
        else:           ax = display
        
        if not ids:
            ids=range(data.shape[0])
        
        if not times:
            times=range(data.shape[1])
        
        sorted_index = numpy.argsort(code)
        sorted_data  = numpy.array([data[i] for i in sorted_index])
               
        kwargs.update( { 'origin' : 'lower', } )
        if any(times) and any(ids): image = ax.imshow(sorted_data, extent=[times[0],times[-1], ids[0], ids[-1]], **kwargs)
        else: image = ax.imshow(sorted_data, **kwargs)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron #')
        ax.set_aspect(aspect='auto')

        return image
      
def kmean_raster_plot( data_spk, code, display=False, kwargs={} ):
        '''
        Plots the kmean data accoringly to clusters
        Inputs:
            data_spk   - List with spike data for each id
            code       - A length N array holding the code book index for each 
                        observation.
            times      - vector with the times of the sliding window
        '''
        
        if not display: ax = pylab.axes()
        else:           ax = display
        
        
        sorted_index = numpy.argsort(code)
        sorted_data  = [data_spk[i] for i in sorted_index]
           
        ids = []
        spike_times = [] 
   
        for i, d in enumerate(sorted_data):
            ids.extend( (i,)*len( d ))
            spike_times.extend(d)
                
               
        if len(spike_times) > 0:
           ax.plot(spike_times, ids, ',', **kwargs)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron #')
        ax.set_ylim(min(ids), max(ids))
    
def make_N_colors(cmap_name, N):
     #! `cmap_name` is a string chosen from the values in cm.datad.keys().You 
     #! will want N to be one minus number of colors you want.
     cmap = matplotlib.cm.get_cmap(cmap_name, N)
     
     
     #! Return list with rgb colors. With return cmap(np.arange(N) a list of 
     #! RGBA values is returned, actually an Nx4 ndarray.
     return cmap(numpy.arange(N))[:, :-1]

def mean(*args, **kwargs):
    d=[]
    for a in args:
        d.append(numpy.mean(a, **kwargs))
    return d

def fano_factor(d):
    
    var=numpy.var(d, axis=0)
    mean=numpy.mean(d, axis=0)
    ff=var/mean
    return ff
    
def mutual_information(count_sr_list):
    '''
    
    Arguments:
        p_sr        - list with the joint probability
                      count of stimulus and response.
                      Each position in count_sr_list[i] contains a matrix 
                      n x m matrix where n is number of stimulus 
                      and m is the number of responses.

    '''    
    mi=[]
        
    for p_sr in count_sr_list:
        p_sr=numpy.array(p_sr)/float(numpy.sum(numpy.sum((p_sr)))) # Probabilities needs to sum to one
        p_s=numpy.sum(p_sr, axis=1) #Marginal distribution p_s
        p_r=numpy.sum(p_sr, axis=0) #Marginal distribution p_r
        
        
        
        tmp=0
        for i in range(p_sr.shape[0]):
            for j in range(p_sr.shape[1]):
                if p_sr[i,j]>0:
                    tmp+=p_sr[i,j]*numpy.log2(p_sr[i,j]/(p_s[i]*p_r[j]))
                if numpy.isnan(tmp):
                    print tmp
        mi.append(tmp)
    
    return numpy.array(mi)


        
    

def PRC(sim_time):
    '''
    Function that calculates excitatory and inhibitory phase response curves
    
    '''
    
    if isinstance( id, int ): id =[ id ] 
    
    dt_pre   = 1/5.*sim_time                                            # preceding time without stimulation
    dt_stim  = sim_time                                                 # stimulation time in ms
    dt_post  = 1/5.*sim_time                                            # post time without stimulation
    dt_dead  = 1000.0                                                   # dead time between runs such that membrane potential settles
    
    traces_v = []                                                       # save voltage traces
    v_ss     = []                                                       # steady state voltage
                                                                                    # between dt_sub and dt )
    dt       = dt_pre + dt_stim + dt_post + dt_dead
    T        = 0                                                        # accumulated simulation time



def vector2matrix(*args, **kwargs):
    n=kwargs.get('n',2)
    d=[]
    for v in args:
        m=len(v)/n            
        d.append(numpy.reshape(v, (n,m)))
    return d
    
def my_glue(*args, **kwargs):
    d=[]
    for v in args:
        if type(v[0]) ==list:
            z=zip(v)
            
            c=reduce(lambda a,b:list(a)+list(b), v)
        if type(v[0])==numpy.ndarray:
            c=numpy.concatenate(*v)

        d.append(c)     
    return d

def sample_wr(pop, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(pop)
    eps=np.finfo(float).eps
    _random, _int = random.random, int  # speed hack
    return [pop[_int(_random() * n-eps)] for _ in itertools.repeat(None, k)]         
    
def slice_line(line, xlim=False, ylim=False):
        
    if xlim:
        x=line.get_xdata()
        y=line.get_ydata()
        
        y=y[x>=xlim[0]]
        x=x[x>=xlim[0]]
        
        y=y[x<=xlim[1]]
        x=x[x<=xlim[1]]
        line.set_xdata(x)
        line.set_ydata(y)
        
    if ylim:
        x=line.get_xdata()
        y=line.get_ydata()
        
        x=x[y>=ylim[0]]
        y=y[x>=ylim[0]]
        
        x=x[y<=ylim[1]]
        y=y[y<=ylim[1]]    
        
        line.set_ydata(y)
        line.set_ydata(x)

def time_resolved_rate(binned_data, bin_extent, kernel_type, res):
    '''
        The rate of at each bin is calculated by a kernel which uses 
        nearby data points to estimate the bin rate. 
                
        Inputs:
            binned_data - binned spike data, n x m where
                          n is number of traces and m is 
                          number of samples
            bin_extent  - extent of kernel
            kernel      - kernel to filter by
            res         - binned data resolution
    
    '''   
    
    if kernel_type=='triangle':
        kernel=numpy.arange(1,numpy.ceil(bin_extent/2.)+1)
        kernel=numpy.append(kernel, numpy.arange(numpy.floor(bin_extent/2.),0,-1))
        kernel=kernel/float(numpy.sum(kernel))
    
    rate_traces=[]
    for i in range(binned_data.shape[0]):
        rate_traces.append(signal.lfilter(kernel, 1/res, binned_data[i,:])*1000.0)     
    rate_traces=numpy.array(rate_traces)
    return rate_traces
    
def wrap(ids, start, stop):
    if stop > len(ids): return ids[ start : ] + ids[ : stop - len(ids) ] 
    else:               return ids[ start : stop ] 

def zero_lag_correlation_matrix(spikes, lenGaussKernel = 1,  samplingW =1):
    '''
    zero_lag_correlation_matrix(steps, MSN_layer, FSI_layer,
CortexX_layer, CortexY_layer, meter, nestResetting,
pathLength = 10, seedNest = randomSeedNest, problemType =
0, tau = 23, initialPositions = None, clustRuntime =
runtime, stepSize = numCortexX/50, lenGaussKernel =
4*filterDecay, stdGaussKernel = filterDecay, samplingW =
100)

    Gets  cell-cell zero time lag cross correlation matrix
from the liquid states of striatal micro circuit generated
when receiving cortical input encoding positions on a 2D
surface

    Input:
            lenGaussKernel    Length of Gaussian kernel used for the 
                              convolution of trains before calculating 
                              correlation
            stdGaussKernel    standard deviation of Gaussian kernal
            samplingW         Width of sampling window for spike trains
    Output:
            zlxc         squared matrix with cell-cell
                         zero time lag cross correlation
            distances    squared matrix with cell-cell distance
            connd        squared matrix with cell-cell
                         distance only for connected neurons, zero otherwise
            times        spike times
            neurons      indices of neurons per spike
            visited      visited positions
    '''
    stdGaussKernel = lenGaussKernel/4.
    import time
    n_spike_trains=spikes.shape[0]
 
    # Here is where the trains are convolved with the spike trains stored in 2D array spikes
    kernel = signal.gaussian(lenGaussKernel/samplingW, stdGaussKernel/samplingW)
    for ii in xrange(n_spike_trains):
        spikes[ii, :] = np.convolve(spikes[ii,:], kernel/sum(kernel),'same')


    
    #convloveld_data=convolve(spikes, lenGaussKernel, 'gaussian', axis=0, single=False,
    #         std_gauss_kernel=stdGaussKernel, sampling_window=1, no_mean=True)

    
    ##ef corr_coef(spk):
    #    norm0 = np.sqrt(numpy.mean(spikes[ii,:]**2)*numpy.mean(spikes[jj, :]**2))
    #   aux = numpy.mean(spikes[ii,:]*spikes[jj, :])/norm0
    #vpolyval = np.vectorize(mypolyval, excluded=['p'])
    
    #Inside this double for the cross correlation matrix is stored in zlxc
    zlxc = np.zeros((n_spike_trains, n_spike_trains))
    
    for ii in xrange(n_spike_trains):
        start=time.time()
        for jj in xrange(n_spike_trains):
            #            aux = np.corrcoef(spikes[ii,:],spikes[jj,:])[0,1]
            norm0 = np.sqrt(numpy.mean(spikes[ii,:]**2)*numpy.mean(spikes[jj, :]**2))
            aux = numpy.mean(spikes[ii,:]*spikes[jj, :])/norm0
            if numpy.isnan(aux): zlxc[ii, jj] = 0.
            else: zlxc[ii, jj] = aux
        stop=time.time()
        print ii, stop-start
            
            #if ii <= jj:
            #    distances[ii, jj] =numpy.sqrt(numpy.sum(numpy.power(numpy.array(positions[ii][:]) -numpy.array(positions[jj][:]), 2.)))
            #    distances[jj, ii] = distances[ii, jj]
    
    #connd = numpy.zeros((n_spike_trains, n_spike_trains))
    #minIndex = min(indices)
    #targets = topp.GetTargetNodes(indices,topp.GetLayer([indices[0]]))
    #for ii, tt in enumerate(targets):
    #    if len(tt)>0:
    #        for jj, target in enumerate(tt):
    #            connd[ii, target - minIndex] = 1.
    #            neurons -= min(neurons)
    
    #return zlxc, distances, connd
    
    #To cluster then I just do:

    
    
    #return zlxc, distances, connd
    return zlxc



import unittest



class TestModule_functions(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_misc_update(self):
        d1={1:{1:1,
               2:{1:1}},
            2:{1:1,
               2:{1:1}}}
        d2={1:{2:{2:1}}}
        d3=deepcopy(d1)
        d3[1][2].update({2:1})
        self.assertDictEqual(dict_update(d1,d2), d3)

        d2={1:{2:{1:3}}}
        d3=deepcopy(d1)
        d3[1][2].update({1:3})
        self.assertDictEqual(dict_update(d1,d2), d3)

        
        d1[1][2][1]=None
        d4=deepcopy(d1)
        self.assertDictEqual(dict_update(d1,d2, skip=True), d4)

        d1[1][2][1]=1
        d4=deepcopy(d1)
        d4[1][2][1]=3
        self.assertDictEqual(dict_update(d1,d2, skip=True), d4)
        
        d2={1:None}
        self.assertDictEqual(dict_update(d1,d2, no_mapping_change=True), d1)
        
        d2={1:None, 2:None}
        self.assertDictEqual(dict_update(d1,d2, no_mapping_change=False), d2)   

        d2={1:None, 2:None}
        self.assertDictEqual(dict_update(d2, d1, no_mapping_change=True), d2)   
             
    def test_dict_apply_operation(self):
        d1={1:{1:2}}
        
        d2=dict_apply_operation(deepcopy(d1), [1,1], 5,'=')   
        d3=dict_apply_operation(deepcopy(d1), [1,1], 5,'*')    
        d4=dict_apply_operation(deepcopy(d1), [1,1], 5,'+')   
        d5=dict_apply_operation(d1, [1,1], 5,'+')  
        
        self.assertDictEqual(d2, {1:{1:5}})
        self.assertDictEqual(d3, {1:{1:10}})
        self.assertDictEqual(d4, {1:{1:7}})
        self.assertDictEqual(d5, d1)


    def test_dict_slice(self):
        d1={1:1,2:2,3:3,4:4}
        d2=dict_slice(d1, [1,3])
        self.assertDictEqual(d2, {1:1,3:3})

    def test_dict_iter(self):
        d={'a':{'b':{'c':1, 'd':2}, 
                'e':{'f':3, 'g':{'h':{'i':{'bu':5}, 'j':6}}}}, 
           'k':{'l':5, 'm':6}}
             

    def test_dict_recursive_get(self):
        d={'a':{'b':{'c':1, 'd':2}, 
                'e':{'f':3, 'g':{'h':{'i':{'bu':5}, 'j':6}}}}, 
           'k':{'l':5, 'm':6}}
        
        val=dict_recursive_get(d, ['a','b','c'])
        self.assertEqual(val, 1)
        self.assertRaises(KeyError, dict_recursive_get, d, ['a','b','z'])
        
if __name__ == '__main__':
    
    test_classes_to_run=[
                            TestModule_functions,
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)



