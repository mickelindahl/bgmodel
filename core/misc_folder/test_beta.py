'''
Created on Mar 3, 2015

@author: mikael
'''
import numpy
import pylab
params=range(1,9)
change=numpy.array([6, 2, 2, 1.5, 1.25, 2, 1.5, 100]); 
# %f_beta=@
d0=0.8
# shift=(1-(1-0.8)).*(change-1); %mV Relative max
f_beta=lambda f: (1-f)/(d0+f*(1-d0)); #%Relarive max
# %shift=params.*(change-1); %mV Relative normal
# betas=shift./params;
betas=f_beta(change);

tata=numpy.linspace(0, 1, 11)

for i in range(8):
    f=(1+betas[i]*(tata-0.8)) #*params[i];
    pylab.subplot(4,2,+i+1)
    pylab.plot(tata,f)
#     hold on
    pylab.plot(0.8,params[i], 'o')
    pylab.title(' ch min/max {}/{}'.format(100.*f[0]/f[-1], 100*f[0]/f[-3]))
    pylab.tight_layout()
pylab.show()