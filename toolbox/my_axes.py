# coding:latin
from matplotlib.axes import Axes 
from matplotlib.ticker import MaxNLocator


class MyAxes(Axes):
    
    def __init__(self,fig, rect,
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
        super( MyAxes, self ).__init__(fig, rect,
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
            else: raise ValueError( 'unknown spine location: %s'%loc )
    
    def my_remove_axis(self, xaxis=False, yaxis=False ):
        '''
        Remove axis.
        
        Inputs:
            self        - axis handel
            xaxis     - True/False
            yaxis     - True/False
        '''
        if xaxis:      
            self.set_xticklabels( '' )                    # turn of x ticks
            self.xaxis.set_ticks_position( 'none' )       # turn of x tick labels
            self.set_xlabel( '' )                         # turn of x label
        
        if yaxis:
            self.set_yticklabels( '' )                    # turn of y ticks
            self.yaxis.set_ticks_position( 'none' )       # turn of y tick labels
            self.set_ylabel( '' )                         # turn of y label
        
    def my_set_no_ticks(self, xticks=None, yticks=None):
        '''
        set_no_ticks(self, xticks, yticks)
        Set number of ticks on axis
            self     - axis handel
            xticks - number of xticks to show
            yticks - number of yticks to show 
        '''
        if xticks: self.xaxis.set_major_locator( MaxNLocator( xticks ) )  
        if yticks: self.yaxis.set_major_locator( MaxNLocator( yticks ) )                    
    
    def my_twinx(self):
        """
        call signature::

          ax = twinx()

        create a twin of Axes for generating a plot with a sharex
        x-axis but independent y axis.  The y-axis of self will have
        ticks on left and the returned axes will have ticks on the
        right
        """
        ax=super( MyAxes, self ).twinx()
        
        ax=convert_super_to_sub_class(ax)
        
        return ax
        
def convert_super_to_sub_class(superClass):
        ''' Convert a super class object into a sub class object'''
        subClass = superClass
        del superClass
        subClass.__class__ = MyAxes
        subClass._init_extra_attributes()
        
        return subClass        

