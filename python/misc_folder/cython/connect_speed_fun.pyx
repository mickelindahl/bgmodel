cimport nest.pynestkernel as _kernel  

def fun_pre_post(int s, int d, char* m):
    pushsli=_kernel.pushsli
    runsli=_kernel.runsli

    pushsli(s) #s
    pushsli(d) #d
    runsli('/{} Connect'.format(m)) 