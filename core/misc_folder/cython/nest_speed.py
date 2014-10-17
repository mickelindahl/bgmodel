from nest import NESTError, broadcast
import nest.pynestkernel as _kernel  


def fun_pre_post(s,d,m):
    pushsli=_kernel.pushsli
    runsli=_kernel.runsli

    pushsli(s) #s
    pushsli(d) #d
    runsli('/{} Connect'.format(m))    
    
def fun_pre_post_params(*args):
    pushsli=_kernel.pushsli
    runsli=_kernel.runsli
    pushsli(args[0]) #s
    pushsli(args[1]) #d
    pushsli(args[2]) #p
    runsli('/%s Connect' % args[3])  
    
def fun_pre_post_weight_delay(*args):
    pushsli=_kernel.pushsli
    runsli=_kernel.runsli
    pushsli(args[0]) #s
    pushsli(args[1]) #d
    pushsli(args[2]) #w
    pushsli(args[3]) #dl
    runsli('/%s Connect' % args[4])                  


def Connect(pre, post, params=None, delay=None, model="static_synapse"):
    """
    Make one-to-one connections of type model between the nodes in
    pre and the nodes in post. pre and post have to be lists of the
    same length. If params is given (as dictionary or list of
    dictionaries), they are used as parameters for the connections. If
    params is given as a single float or as list of floats, it is used
    as weight(s), in which case delay also has to be given as float or
    as list of floats.
    """


    if len(pre) != len(post):
        raise NESTError("pre and post have to be the same length")

    # pre post Connect
    if params == None and delay == None:
        map(fun_pre_post, pre, post, [model]*len(pre))


    # pre post params Connect
    elif params != None and delay == None:
        params = broadcast(params, len(pre), (dict,), "params")
        if len(params) != len(pre):
            raise NESTError("params must be a dict, or list of dicts of length 1 or len(pre).")

        map(fun_pre_post_params, pre, post, params, [model]*len(pre))

    # pre post w d Connect
    elif params != None and delay != None:
        params = broadcast(params, len(pre), (float,), "params")
        if len(params) != len(pre):
            raise NESTError("params must be a float, or list of floats of length 1 or len(pre) and will be used as weight(s).")
        delay = broadcast(delay, len(pre), (float,), "delay")
        if len(delay) != len(pre):
            raise NESTError("delay must be a float, or list of floats of length 1 or len(pre).")
        
        map(fun_pre_post_weight_delay, pre, post, params, delay, [model]*len(pre))
        

    else:
        raise NESTError("Both 'params' and 'delay' have to be given.")