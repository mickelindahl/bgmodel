'''
Created on Mar 19, 2014

@author: lindahlm
'''
def main():    
    IV=build_cases(**{'lesion':True, 'mm':True})
    IF=build_cases(**{'lesion':True})
    FF=build_cases(**{'lesion':False})
    
    curr_IV=range(-200,300,100)
    curr_IF=range(0,500,100)
    rate_FF=range(100,1500,100)
    _, axs=pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16)     
    
    plots('plot_IV_curve', IV, 1, **{'ax':axs[0],'curr':curr_IV, 'node':'FS'})
    plots('plot_IF_curve', IF, 1, **{'ax':axs[1],'curr':curr_IF, 'node':'FS'})
    plots('plot_FF_curve', FF, 1, **{'ax':axs[2],'rate':rate_FF, 'node':'FS',
                                     'input':'CFp'})    
    beautify(axs)
    pylab.show()
    
if __name__ == "__main__":
    main()  