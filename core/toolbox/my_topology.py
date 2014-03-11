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
from nest.topology import *
import nest.topology
import time
import pylab
import nest
import misc


    
def _draw_extent(ax, xctr, yctr, xext, yext):
    """Draw extent and set aspect ration, limits"""

    import matplotlib.pyplot as plt

    # thin gray line indicating extent
    llx, lly = xctr - xext/2.0, yctr - yext/2.0
    urx, ury = llx + xext, lly + yext
    ax.add_patch(plt.Rectangle([llx, lly], xext, yext, fc='none', ec='0.5', lw=1, zorder=1))
    
    # set limits slightly outside extent
    ax.set(aspect='equal', 
           xlim=[llx - 0.05*xext, urx + 0.05*xext],
           ylim=[lly - 0.05*yext, ury + 0.05*yext],
           xticks=[], yticks=[])

def MyPlotLayer(layer, ax=None, nodecolor='b', nodesize=20):
    """
    Plot nodes in a layer.
    
    This function plots only top-level nodes, not the content of composite nodes.
    
    Note: You should not use this function in distributed simulations.
    
    Parameters
    ----------
    layer         GID of layer to plot (as single-element list)
    fig           Matplotlib figure to plot to. If not given, a new figure is created [optional].
    nodecolor     Color for nodes [optional].
    nodesize      Marker size for nodes [optional].

    Returns
    -------
    Matplotlib figure.
    
    See also
    --------
    PlotTargets
    """

    import matplotlib.pyplot as plt
    
    if len(layer) != 1:
        raise ValueError("layer must contain exactly one GID.")

    # get layer extent and center, x and y
    xext, yext = nest.GetStatus(layer, 'topology')[0]['extent'][:2]
    xctr, yctr = nest.GetStatus(layer, 'topology')[0]['center'][:2]
    
    # extract position information, transpose to list of x and y positions
    xpos, ypos = zip(*GetPosition(nest.GetChildren(layer)))

    if not ax:
        ax = pylab.subplot(111)


    ax.scatter(xpos, ypos, s=nodesize, facecolor=nodecolor, edgecolor='none')
    _draw_extent(ax, xctr, yctr, xext, yext)

def MyConnectLayers(pre, post, projections):
   
    
    if 'n_cluster' in projections.keys():

        n_cluster=projections['n_cluster']
        k=projections['kernel']
        syn_model=projections['synapse_model']
        k_new=k*1/(1-1/n_cluster)
        
        pre_ids=nest.GetLeaves(pre)[0]
        post_ids=nest.GetLeaves(post)[0]
        
        sources, targets=misc.cluster_connections(pre_ids, post_ids, k_new, n_cluster)
         
        nest.Connect(list(sources),list(targets), model=syn_model)
    else:
        ConnectLayers(pre, post, projections)
        
    
def MyPlotTargets(src_nrn, tgt_layer, tgt_model=None, syn_type=None, ax=None,
                mask=None, kernel=None,
                src_color='red', src_size=50, tgt_color='blue', tgt_size=20,
                mask_color='red', kernel_color='red'):
    """
    Plot all targets of src_nrn in a tgt_layer.
    
    Note: You should not use this function in distributed simulations.

    Parameters
    ----------
    src_nrn      GID of source neuron (as single-element list)
    tgt_layer    GID of tgt_layer (as single-element list)
    tgt_model    Show only targets of a given model [optional].
    syn_type     Show only targets connected to with a given synapse type [optional].
    fig          Matplotlib figure to plot to. If not given, new figure is created [optional].
    
    mask         Draw topology mask with targets; see PlotKernel for details [optional].
    kernel       Draw topology kernel with targets; see PlotKernel for details [optional].
    
    src_color    Color used to mark source node position [default: 'red']
    src_size     Size of source marker (see scatter for details) [default: 50]
    tgt_color    Color used to mark target node positions [default: 'blue']
    tgt_size     Size of target markers (see scatter for details) [default: 20]
    mask_color   Color used for line marking mask [default: 'red']
    kernel_color Color used for lines marking kernel [default: 'red']

    Returns
    -------
    Matplotlib figure.
    
    See also
    --------
    PlotLayer, GetTargetPositions
    matplotlib.pyplot.scatter
    """

    import matplotlib.pyplot as plt

    # get position of source
    srcpos = GetPosition(src_nrn)[0]

    # get layer extent and center, x and y
    xext, yext = nest.GetStatus(tgt_layer, 'topology')[0]['extent'][:2]
    xctr, yctr = nest.GetStatus(tgt_layer, 'topology')[0]['center'][:2]
    
    if not ax:
        ax = pylab.subplot(111)


    # get positions, reorganize to x and y vectors
    tgtpos = GetTargetPositions(src_nrn, tgt_layer, tgt_model, syn_type)
    if tgtpos:
        xpos, ypos = zip(*tgtpos[0])
        ax.scatter(xpos, ypos, s=tgt_size, facecolor=tgt_color, edgecolor='none')

    ax.scatter(srcpos[:1], srcpos[1:], s=src_size, facecolor=src_color, edgecolor='none',
               alpha = 0.4, zorder = -10)
    
    _draw_extent(ax, xctr, yctr, xext, yext)

    if mask or kernel:
        try:
            PlotKernel(ax, src_nrn, mask, kernel, mask_color, kernel_color)
        except: 
            'Do not support plotting of provided kernel type'

    plt.draw() 
    n_targets=len(tgtpos[0])
    return ax, n_targets
