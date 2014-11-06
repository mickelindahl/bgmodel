# coding:latin
from matplotlib.axes import Axes 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

import numpy

import pylab


# import toolbox.plot_settings as pl
import unittest

class Mixin(object):
    def legend_box_to_line(self):
    
        handels, labels=self.get_legend_handles_labels()
        artist=[]
        for hh in handels:
            color=hh._edgecolor
            linestyle=hh._linestyle
            obj=pylab.Line2D((0,1),(0,0), color=color, linestyle=linestyle)
            artist.append(obj)

        self.legend(artist, labels)    
          
    def my_push_axis(self, xaxis=True, yaxis=True ):
        '''
        Push x and/or y axis  out from plot 10 points. 
        
        Inputs:
            self        - axis handel
            xaxis     - True/False
            yaxis     - True/False
        '''
        
      
        xlabel   = self.get_xlabel()                      # Get xlabel
        ylabel   = self.get_ylabel()                      # Get ylabel
        
    
        push   = []
        if xaxis: push.append('bottom')
        if yaxis: push.append('left')
        
        # Push axis outward by 10 points
        for loc, spine in self.spines.iteritems():
            if loc in push:     spine.set_position( ( 'outward', 10 ) ) 
            #else: raise ValueError( 'unknown spine location: %s'%loc )            
    
        if xaxis: 
            self.xaxis.set_ticks_position( 'bottom' )        
            self.set_xlabel(xlabel,position=( 0.5, -0.2 ) )  # Reposition x label
        
    
        if yaxis: 
            self.yaxis.set_ticks_position( 'left' )          
            self.set_ylabel(ylabel,position=( -0.2, 0.5 ) )  # Reposition y label
    
    
    def my_remove_spine(self, left=False,  bottom=False, right=True, top=True):
        '''
        Removes frames around plot
        '''
        
        remove   = []
        if left:   remove.append('left')
        if bottom: remove.append('bottom')
        if right:  remove.append('right')
        if top:    remove.append('top')
        
        # Remove upper and right axes frames and move the bottom and left 
        for loc, spine in self.spines.iteritems():
            if loc in remove: spine.set_color( 'none' )          # don't draw spine
            #else: raise ValueError( 'unknown spine location: %s'%loc )
    
    def my_remove_axis(self, xaxis=False, yaxis=False , keep_ticks=False):
        '''
        Remove axis.
        
        Inputs:
            self        - axis handel
            xaxis     - True/False
            yaxis     - True/False
        '''
        if xaxis:      
            self.set_xticklabels( '' )                    # turn of x ticks
            if not keep_ticks:
                self.xaxis.set_ticks_position( 'none' )       # turn of x tick labels
            self.set_xlabel( '' )                         # turn of x label
        
        if yaxis:
            self.set_yticklabels( '' )                    # turn of y ticks
            if not keep_ticks:
          
                self.yaxis.set_ticks_position( 'none' )       # turn of y tick labels
                self.set_ylabel( '' )                         # turn of y label
        
    def my_set_no_ticks(self, xticks=None, yticks=None, zticks=None):
        '''
        set_no_ticks(self, xticks, yticks)
        Set number of ticks on axis
            self     - axis handel
            xticks - number of xticks to show
            yticks - number of yticks to show 
        '''
        if xticks: self.xaxis.set_major_locator( MaxNLocator( xticks ) )  
        if yticks: self.yaxis.set_major_locator( MaxNLocator( yticks ) )                    
        if hasattr(self ,'zaxis'):
            if zticks: self.zaxis.set_major_locator( MaxNLocator( zticks ) )  

        
    def twinx(self):
        """
        call signature::

          ax = twinx()

        create a twin of Axes for generating a plot with a sharex
        x-axis but independent y axis.  The y-axis of self will have
        ticks on left and the returned axes will have ticks on the
        right
        """
        ax=super( self.my_class, self ).twinx()
        
        ax=convert_super_to_sub_class(ax)
        
        return ax


class MyAxes3D_base(Axes3D):
    def __init__(self, fig, rect=None, *args, **kwargs):
        '''
        Build an :class:'Axes3D' instance in
        :class:'~matplotlib.figure.Figure' *fig* with
        *rect=[left, bottom, width, height]* in
        :class:'~matplotlib.figure.Figure' coordinates
        
        Optional keyword arguments:
        
          ================   =========================================
          Keyword            Description
          ================   =========================================
          *azim*             Azimuthal viewing angle (default -60)
          *elev*             Elevation viewing angle (default 30)
          *zscale*           [%(scale)s]
          ================   =========================================
        '''
        super( MyAxes3D_base, self ).__init__( fig, rect, *args, **kwargs)
        
        # In order to be able to convert super clas object to subclass object
        self._init_extra_attributes(fig)
       
            
    
    def _init_extra_attributes(self, fig=None):  
        if fig: fig.add_axes(self)  
        self.my_class=MyAxes3D_base
        

class MyAxes3D(MyAxes3D_base, Mixin):
    pass
       
class MyAxes_base(Axes):
    
    def __init__(self, fig, rect,
                 axisbg = None, # defaults to rc axes.facecolor
                 frameon = True,
                 sharex=None, # use Axes instance's xaxis info
                 sharey=None, # use Axes instance's yaxis info
                 label='',
                 xscale=None,
                 yscale=None,
                 **kwargs
                 ):
        """
        Build an :class:`Axes` instance in
        :class:`~matplotlib.figure.Figure` *fig* with
        *rect=[left, bottom, width, height]* in
        :class:`~matplotlib.figure.Figure` coordinates

        Optional keyword arguments:

          ================   =========================================
          Keyword            Description
          ================   =========================================
          *adjustable*       [ 'box' | 'datalim' ]
          *alpha*            float: the alpha transparency
          *anchor*           [ 'C', 'SW', 'S', 'SE', 'E', 'NE', 'N',
                               'NW', 'W' ]
          *aspect*           [ 'auto' | 'equal' | aspect_ratio ]
          *autoscale_on*     [ *True* | *False* ] whether or not to
                             autoscale the *viewlim*
          *axis_bgcolor*     any matplotlib color, see
                             :func:`~matplotlib.pyplot.colors`
          *axisbelow*        draw the grids and ticks below the other
                             artists
          *cursor_props*     a (*float*, *color*) tuple
          *figure*           a :class:`~matplotlib.figure.Figure`
                             instance
          *frame_on*         a boolean - draw the axes frame
          *label*            the axes label
          *navigate*         [ *True* | *False* ]
          *navigate_mode*    [ 'PAN' | 'ZOOM' | None ] the navigation
                             toolbar button status
          *position*         [left, bottom, width, height] in
                             class:`~matplotlib.figure.Figure` coords
          *sharex*           an class:`~matplotlib.axes.Axes` instance
                             to share the x-axis with
          *sharey*           an class:`~matplotlib.axes.Axes` instance
                             to share the y-axis with
          *title*            the title string
          *visible*          [ *True* | *False* ] whether the axes is
                             visible
          *xlabel*           the xlabel
          *xlim*             (*xmin*, *xmax*) view limits
          *xscale*           [%(scale)s]
          *xticklabels*      sequence of strings
          *xticks*           sequence of floats
          *ylabel*           the ylabel strings
          *ylim*             (*ymin*, *ymax*) view limits
          *yscale*           [%(scale)s]
          *yticklabels*      sequence of strings
          *yticks*           sequence of floats
          ================   =========================================
        """
        
        # About the super() function 
        # (from python programming - Michael Dawson, page 277) 
        # Incorporate the superclass ConductanceList method's functionality. 
        # To add a new attribute ids i need to override the inherited 
        # constructor method from ConductanceList. I also want my new 
        # constructor to create all the attributes from ConductanceList. 
        # This can be done with the function super(). It lets you invoke the 
        # method of a base class(also called a superclass). The first argument 
        # in the function call, 'MyConductanceList', says I want to invoke a 
        # method of the superclass (or base class) of MyConductanceList which 
        # is ConductanceList. The next argument. seöf, passes a reference to 
        # the object so that ConductanceList can get to the object and add 
        # its attributes to it. The next part of the statement __init__(
        # signals, id_list, dt, t_start, t_stop, dims) tells python I want to
        # invoke the constructor method of ConductanceList and a want to pass 
        # it the values of signals, id_list, dt, t_start, t_stop and dims.
        
        # Invoke __init__ of base class ConductanceList which has 
        # AnalogSignalList as base class where the method exist.
        super( MyAxes_base, self ).__init__(fig, rect,
                 axisbg , # defaults to rc axes.facecolor
                 frameon ,
                 sharex, # use Axes instance's xaxis info
                 sharey, # use Axes instance's yaxis info
                 label,
                 xscale,
                 yscale,
                 **kwargs)
        
        # In order to be able to convert super clas object to subclass object
        self._init_extra_attributes(fig)
       
            
    
    def _init_extra_attributes(self, fig=None):  
        if fig: fig.add_axes(self)  
        self.my_class=MyAxes_base
    
class MyAxes(MyAxes_base, Mixin):
    pass
 
    
         
def convert_super_to_sub_class(superClass, into=MyAxes):
        ''' Convert a super class object into a sub class object'''
        subClass = superClass
        del superClass
        subClass.__class__ = into
        subClass._init_extra_attributes()
        
        return subClass        
    
def convert(ax, **kwargs):
    return convert_super_to_sub_class(ax,**kwargs)




class TestAxes(unittest.TestCase):
    
    
    def test_legend_box_to_line(self):
#         _, axs=pl.get_figure(n_rows=1, n_cols=1, w=1000.0, h=800.0, fontsize=16) 
        fig = pylab.figure( facecolor = 'w' )
        ax=MyAxes(fig, [ 0.1,  0.1,  0.5,0.5 ] )
  
        r=numpy.random.random(100)
        ax.hist(r, **{'histtype':'step', 'label':'test 1'})
        r=numpy.random.random(100)
        ax.hist(r, **{'histtype':'step', 'label':'test 2', 'linestyle':'dashed'})
        
        ax.legend_box_to_line()
#         pylab.show()
 

class TestAxes3D(unittest.TestCase):
    
    def setUp(self):
        import numpy as np
        X= np.arange(-5, 5, 0.25)
        Y=np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.sin(R)
        self.X,self.Y,self.Z=X,Y,Z
        
        self.fig = pylab.figure( facecolor = 'w' )
        self.ax=MyAxes3D(self.fig, [ 0.1,  0.1,  0.5,0.5 ] )
    
    def test_surface(self):
        
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter        

        surf = self.ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap='coolwarm',
                linewidth=0, antialiased=False)
        self.ax.set_zlim(-1.01, 1.01)
        
        self.ax.zaxis.set_major_locator(LinearLocator(5))
        self.ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        self.fig.colorbar(surf, shrink=0.5, aspect=5)
#         pylab.show()


    def test_my_set_no_ticks(self):
        surf = self.ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, 
                                    cmap='coolwarm', 
#                                     shade=True,
                linewidth=0, antialiased=True)
        self.ax.my_set_no_ticks(None, None, zticks=10)
#         pylab.show()
        
        
        
    def test_elev(self):
        # Rotate up
        for ii in xrange(0,15,5):
            ax=MyAxes3D(pylab.figure( facecolor = 'w' ), [ 0.1,  0.1,  0.5,0.5 ] )
            print ii
            ax.view_init(elev=ii)
        pylab.show()
            
#     def test_azim(self):
#         # Rotate left
#         for ii in xrange(0,15,5):
#             ax=MyAxes3D(pylab.figure( facecolor = 'w' ), [ 0.1,  0.1,  0.5,0.5 ] )
#             
#             ax.view_init(azim=ii)
#         pylab.show()           
            
if __name__ == '__main__':
    test_classes_to_run=[
#                          TestAxes,
                         TestAxes3D
                         
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    unittest.TextTestRunner(verbosity=2).run(big_suite)
    
    #unittest.main() 
