'''
Created on Nov 3, 2014

@author: mikael
'''
import numpy
import pylab

from toolbox import plot_settings as ps
import matplotlib.gridspec as gridspec


def gs_builder(*args, **kwargs):
    
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.05 ), 
              hspace=kwargs.get('hspace', 1. / n_cols ))

    iterator = [[slice(1,4),slice(1,5)],
                [slice(1,4),5]]
    
    return iterator, gs, 

n=10
z=numpy.random.random((n,n))
z[1,:]+=1
z[8,:]+=2
_vmin=0
_vmax=4
stepx=1
stepy=1
startx=0
starty=0
stopy=10
stopx=10
res=10

fig, axs=ps.get_figure2(n_rows=5, n_cols=6, w=700, h=500, fontsize=24,
                        frame_hight_y=0.5, frame_hight_x=0.7, title_fontsize=20,
                        gs_builder=gs_builder)        

pos=numpy.linspace(0.5,9.5,10)
axs[1].barh(pos,numpy.mean(z,axis=1)[::-1], align='center')


# ax=pylab.subplot(111)
nets=['Net_'+str(i) for i in range(10)]
x1,y1=numpy.meshgrid(numpy.linspace(startx, stopx, res+1),
                   numpy.linspace(stopy, starty, res+1))
# x2,y2=numpy.meshgrid(numpy.linspace(startx+stepx/2, 
#                                     stopx-stepx/2, res),
#                      numpy.linspace(stopy+stepy/2, 
#                                     starty-stepy/2, res))

print z

im = axs[0].pcolor(x1, y1, z, cmap='coolwarm', 
                   vmin=_vmin, vmax=_vmax)
axs[0].set_yticks(pos)
axs[0].set_yticklabels(nets)
axs[0].set_xticks(pos)
axs[0].set_xticklabels(nets, rotation=70)
axs[1].my_remove_axis(xaxis=False, yaxis=True)
axs[1].my_set_no_ticks(xticks=2)
axs[1].set_xticks([1,2])


box = axs[0].get_position()
axColor=pylab.axes([box.x0+0.1*box.width, 
                    box.y0+box.height+box.height*0.3, 
                    box.width*0.8, 
                    0.05])
#     axColor = pylab.axes([0.05, 0.9, 1.0, 0.05])
cbar=pylab.colorbar(im, cax = axColor, orientation="horizontal")
cbar.ax.set_title('MSE Control vs lesion')#, rotation=270)
from matplotlib import ticker

tick_locator = ticker.MaxNLocator(nbins=5)
cbar.locator = tick_locator
cbar.update_ticks()

pylab.show()
