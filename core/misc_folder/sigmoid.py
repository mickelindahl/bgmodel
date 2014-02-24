import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def sigmoid(p,x):
    x0,y0,c,k=p
    y = c / (1 + np.exp(-k*(x-x0))) + y0
    return y

def residuals(p,x,y):
    return y - sigmoid(p,x)

def resize(arr,lower=0.0,upper=1.0):
    arr=arr.copy()
    if lower>upper: lower,upper=upper,lower
    arr -= arr.min()
    arr *= (upper-lower)/arr.max()
    arr += lower
    return arr

# raw data

y = np.array([29.34254837,  20.7612381,   14.56746674,   8.71971893,   4.94809484],dtype='float')
x = np.array([1.,    13.25,  25.5,   37.75,  50. ],dtype='float')

p=np.polyfit(x, y, 3, rcond=None, full=False)
print 'p', p
f=lambda x: p[0]*x**3+p[1]*x**2+p[2]*x+p[3]
df=lambda x: 3*p[0]*x**2+2*p[1]*x+p[2]

xr=np.linspace(min(x),max(x), 50)
print f(xr)
#plt.plot(xr,f(xr))
#plt.plot(xr,df(xr),'r')
#plt.plot(x,y,'g*')

#plt.show()

#x = np.array([821,576,473,377,326],dtype='float')
#y = np.array([255,235,208,166,157],dtype='float')

x=resize(x,lower=0.0)
y=resize(y,lower=0.0)
print(x)
print(y)
p_guess=(np.median(x),np.median(y),1.0,1.0)
print p_guess
p, cov, infodict, mesg, ier = scipy.optimize.leastsq(
    residuals,p_guess,args=(x,y),full_output=1,warning=True)  

x0,y0,c,k=p
print('''\
x0 = {x0}
y0 = {y0}
c = {c}
k = {k}
'''.format(x0=x0,y0=y0,c=c,k=k))

xp = np.linspace(0, 1.1, 1500)
pxp=sigmoid(p,xp)

# Plot the results
plt.plot(x, y, '.', xp, pxp, '-')
plt.xlabel('x')
plt.ylabel('y',rotation='horizontal') 
plt.grid(True)
plt.show()