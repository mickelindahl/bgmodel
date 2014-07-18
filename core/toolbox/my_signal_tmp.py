'''
Created on Jul 17, 2014

@author: mikael
'''


class Data_phases_diff_with_cohere_base(object):
    def hist(self, ax,  num=100.0, **k):
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
            
#         k['histtype']=k.get('histtype','step')
#         k['normed']=1
        k['linestyle']='--'
#         k['linewidth']=0
        bins=numpy.linspace(-numpy.pi, numpy.pi, num)
#         y=reduce(lambda x,y:list(x)+list(y),self.y)
#         h=ax.hist(numpy.array(self.y.ravel()), bins,  **k)
        
        a=numpy.mean(self.y_bins, axis=0)[0:-1]
        step=numpy.diff(a)[0]
        a=numpy.array([[aa, aa+step] for aa in a]).ravel()
        b=numpy.array([[bb,bb] for bb in numpy.sum(self.y, axis=0)],
                      dtype=numpy.float).ravel()
        h=ax.plot(a, b/sum(b)*100., **k)
        color=pylab.getp(h[0], 'color')   
        
        k['linestyle']='-'
        k['color']=color
        idx=self.idx_sorted[self.coherence[self.idx_sorted]>self.p_conf95]
        y=self.y[idx,:]
        y_bins=self.y_bins[idx,:]
        
        a=numpy.mean(y_bins, axis=0)[0:-1]
        a=numpy.array([[aa, aa+step] for aa in a]).ravel()
        b=numpy.array([[bb,bb] for bb in numpy.sum(y, axis=0)],
                      dtype=numpy.float).ravel()
        h=ax.plot(a, b/sum(b)*100., **k)
#         ax.hist(numpy.array(self.y[idx,:].ravel()), bins,  **k)
        
#         ax.set_xlim(-numpy.pi, numpy.pi)
        ax.set_xlabel('Angle (Rad)') 
        ax.set_ylabel('Count') 
        ax.my_set_no_ticks(xticks=10, yticks=6)
        ax.legend()
        
    def Factory_phases_diff_with_cohere(self, *args, **kwargs):
        '''
        Returns the phase of the population firing rate filters in the band
        lowcut to highcut. 
        '''
        fs=kwargs.get('fs')
        low=kwargs.get('lowcut', 0.5)
        high=kwargs.get('highcut', 1.5)
        time_bin=int(1000/fs)
        other=kwargs.get('other', None)
        sample=kwargs.get('sample',10)
        
        assert other!=None, 'need to provide other spike list'
        
        ids1, ids2=shuffle(*[self.id_list, other.id_list],
                           **{'sample':sample})
               
        sl1=self.id_slice(ids1)
        sl2=other.id_slice(ids2)
        
        signals1=sl1.firing_rate(time_bin, average=False, **kwargs)
        signals2=sl2.firing_rate(time_bin, average=False, **kwargs)       
        
#         args=[lowcut, highcut, order, fs]
        y=sp.phases_diff(signals1, signals2, **kwargs)
        vals=[]
        bins=[]
        bins0=numpy.linspace(3*-numpy.pi, 3*numpy.pi, 3*100)
        for yy in y:
            val, bin=numpy.histogram(yy, bins0)
            vals.append(val)
            bins.append(bin)
            
#             hg.append(numpy.histogram(yy, 100))
        
#         y=numpy.array(y, dtype=numpy.float16)
#         y=hg
        x2, y2=sp.coherences(signals1, signals2, **kwargs)
        
        idx, v =sp.sort_coherences(x2, y2, low, high)
 
        L=float(len(signals1[0]))/kwargs.get('NFFT')
        p_conf95=numpy.ones(len(x2))*(1-0.05**(1/(L-1)))  
 
        
        d= {'ids1':self.id_list,
            'ids2':other.id_list,
#             'x':self.time_axis_centerd(time_bin) , 
#             'y':y,
            'y':numpy.array(vals),
            'y_bins':numpy.array(bins),
            'coherence':v,
            'idx_sorted':idx,
            'p_conf95':p_conf95}

        return Data_phases_diff_with_cohere(**d)