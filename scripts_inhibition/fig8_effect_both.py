'''
Created on Nov 13, 2014

@author: mikael



'''

import numpy
import pprint
pp=pprint.pprint
import toolbox.plot_settings as ps
import matplotlib.gridspec as gridspec

from effect_conns import (_plot_conn, nice_labels, 
                          separate_M1_M2, set_colormap)
from toolbox.network.manager import get_storage
from simulate import save_figures

def extract_d(file_names, attr, y_entries):
    d_list = []
    for file_name in file_names:
        sd = get_storage(file_name, '')
        dd = sd.load_dic()
        d_list.append(dd['d_gradients_control'][attr])
        d_list.append(dd['d_gradients_lesion'][attr])
    
    print len(d_list)
    for d in d_list:
        pp(d.keys())
    
    z = []
    for d in d_list:
        z.append(d['z'])
    

    v = [[v1, v2] for v1, v2 in zip(z[0].ravel(), z[1].ravel())]
    w = [[v1, v2] for v1, v2 in zip(z[2].ravel(), z[3].ravel())]
    v = numpy.reshape(numpy.array(v).ravel(), (y_entries, 23 * 2))
    w = numpy.reshape(numpy.array(w).ravel(), (y_entries, 23 * 2))
    z_new = [list(v1) + list(v2) for v1, v2 in zip(v, w)]
    z_new = numpy.reshape(numpy.array(z_new).ravel(), 
                          (y_entries * 2, 23 * 2))
    
    d['z'] = z_new
    d['labelsy_meta'] = d_list[0]['labelsy']
    d['labelsx_meta'] = d_list[0]['labelsx_meta']
    d['labelsx'] = range(23 * 2)
    d['labelsy'] = range(y_entries * 2)
    
    return d

def gs_builder_conn(*args, **kwargs):

    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.02 ), 
              hspace=kwargs.get('hspace', 0.1 ))

    iterator = [[slice(0,1),slice(0,15)],
                [slice(1,4),slice(0,15)],
                [slice(4,6),slice(0,15)],]
    
    return iterator, gs, 

file_names=[('/home/mikael/results/papers/inhibition/network/'
                    +'supermicro/fig8_effect_beta_conn/data'),
            ('/home/mikael/results/papers/inhibition/network/'
                    +'supermicro/fig8_effect_slow_wave_conn/data')]
       
attr='firing_rate'
y_entries=8
d = extract_d(file_names, attr, y_entries)
d1= extract_d(file_names, 'mean_coherence_max', 4)
kw={'n_rows':8, 
        'n_cols':16, 
        'w':int(72/2.54*11.6), 
        'h':int(72/2.54*22)/3, 
        'fontsize':7,
        'title_fontsize':7,
        'gs_builder':gs_builder_conn}

fig, axs=ps.get_figure2(**kw) 

l=separate_M1_M2(*[d],**{'z_key':'z',
                         'z_sep_at':4,
                         'labelsy_sep_at':4} )
l.append(d1)
pp(l)

images=[]
for ax, d in zip(axs[0:3], l):

    kwargs={
            'ax':ax,
            'd':d,
            'images':images,
            'fontsize_x':7,
            'fontsize_y':7,
            'z_key':'z',
            'startx':0,
            'starty':0,
            'vertical_lines':True, 
            'horizontal_lines':True, 
            'nice_labels_x':nice_labels(version=0),
            'nice_labels_y':nice_labels(version=1),
            'cmap':'coolwarm',
            'color_line':'k',
            'x_sep_linewidth':0.5,
            'y_sep_linewidth':0.5}

    _plot_conn(**kwargs)




for i, clim in enumerate([[-2,2], [-50,50], [-0.6,0.6]]):
    images[i].set_clim(clim)

args=[[axs[0],axs[1],axs[2]], 
      ['', '', 'Coher./weight'],
      [images[0],images[1],images[2]],
      [{'nbins':2, 'x_scale':1.03},
       {'nbins':3, 'x_scale':1.03},
       {'nbins':3, 'x_scale':1.03}]]

for ax, label, im, k in zip(*args):
        set_colormap(ax, im, label,**k)

axs[1].text(1.12, 0.9, 'Rate/weight', 
            transform=axs[1].transAxes,
            rotation=270)    
    
axs[0].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True) 
axs[1].my_remove_axis(xaxis=True, yaxis=False,keep_ticks=True)
    
save_figures([fig], (__file__.split('/')[-1][0:-3]
                                           +'/data'), dpi=200)

import pylab
pylab.show()



