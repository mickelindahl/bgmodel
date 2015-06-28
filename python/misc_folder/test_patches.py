'''
Created on Apr 30, 2015

@author: mikael
'''
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import pylab
import pprint
pp=pprint.pprint

patches=[]
patches += [
    Wedge((1,1), .2, 0, 300, facecolor ='b', edgecolor='k',  linewidth=2.),             # Full circle
    Wedge((1,1), .2, 300, 360, facecolor='r', edgecolor='k', linewidth=2.), # Full ring
    Wedge((-1,1), .2, 300, 360, facecolor='r', edgecolor='k',linewidth=2.), # Full ring
    Wedge((-1,-1), .2, 0, 45, facecolor='b', edgecolor='k',  linewidth=2.),              # Full sector
    Wedge((-1,-1), .2, 45, 360,facecolor='g', edgecolor='k', linewidth=2.), # Ring sector
]

# for i in range(N):
#     polygon = Polygon(np.random.rand(N,2), True)
#     patches.append(polygon)
linewidth=0.5
colors = ['w','k','b','r']
fig, ax = pylab.subplots()
p = PatchCollection(patches)
# ax.add_col1lection(p)a
# for p in patches:
#     ax.add_patch(p)
# # plt.colorbar(p)
# 
# ax.set_xlim([-2,2])
# ax.set_ylim([-2,2])
# 
# # pylab.show()

res=10
import numpy
step=2./res
pos=numpy.meshgrid(numpy.linspace(1+step/2,3-step/2,res), 
                   numpy.linspace(1+step/2,3-step/2,res))

pos=zip(pos[0].ravel(),pos[1].ravel())


def plot_pies(ax, X, Y, Z, radious, colors, **kw):
    '''
    X,Y two dimensional grids [n,m]
    Z three dimensional [n,m,l]
    colors list with l colors
    radoius of pie 
    
    '''
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x=X[i][j]
            y=Y[i][j]
            for k in range(len(Z[i][j])):
                lower=0 if k==0 else Z[i][j][k-1]
                upper=Z[i][j][k]
                c=colors[k]
                w=Wedge([x,y], radious, 360*lower, 360*upper, 
                        facecolor =c, **kw)
                patches += [w] 
    
    for p in patches:
        ax.add_patch(p)

y=[]
for _ in range(res**2):
    v=numpy.random.randint(low=1, high=3,size=4)
    v=v/float(sum(v))
    v=numpy.cumsum(v)
    v=[0]+list(v)
    y.append(numpy.array(v))
    
    
pp(y)
data=[]
for e0,e1 in zip(y,pos):
    data+=[{'y':e0, 'colors':colors, 'pos':e1,
            'linewidth':linewidth,
            'radious':2./(2.*res+5)}]
    
patches=[]
for k in data:  
    for i ,c, in enumerate(k['colors']):
        lower=k['y'][i]
        upper=k['y'][i+1]
        if lower==upper:continue
        print lower, upper
        w=Wedge(k['pos'], 
                k['radious'], 
                360*lower, 360*upper, 
                facecolor =c, edgecolor='k',  linewidth=2.)
        patches += [w] 
for p in patches:
    ax.add_patch(p)
# plt.colorbar(p)

ax.set_xlim([1,3])
ax.set_ylim([1,3])
pylab.show()

pp(data) 

   