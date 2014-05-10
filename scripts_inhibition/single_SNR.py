'''
Created on Mar 19, 2014

@author: lindahlm
'''



# from single import (beautify, build_general, creat_dic_specific, create_nets,  
#                     create_list_dop, do, set_optimization_val)
# import pylab
# import toolbox.plot_settings as pl
# from toolbox import misc
# 
# import pprint
# pp=pprint.pprint
# 
# def build_cases(**kwargs):
#     su=kwargs.get('single_unit', 'SN')
#     
#     
#     inputs=kwargs.get('inputs',['GIp', 'STp', 'ESp', 'M1p']) 
#     d=build_general(**kwargs)
#     d=misc.dict_update(d, creat_dic_specific(kwargs, su, inputs))
#     
#     l = create_list_dop(su)
#     
#     names=['$SNr_{+d}$',
#            '$SNr_{-d}$',
# ]
#     
#     nets = create_nets(l, d, names)
#     
#     return nets
# 
# def main():   
#     node='SN' 
#     IV=build_cases(**{'lesion':True, 'mm':True})
#     IF=build_cases(**{'lesion':True,  'size':50})
#     FF=build_cases(**{'lesion':False})
#     opt=build_cases(**{'lesion':False})
#     hist=build_cases(**{'lesion':False, 'size':200})
#         
#     curr_IV=range(-200,300,100)
#     curr_IF=range(0,500,100)
#     rate_FF=range(1000,2000,100)
#     _, axs=pl.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16)     
#     
#     do('plot_IV_curve', IV, 1, **{'ax':axs[0],'curr':curr_IV, 'node':node})
#     do('plot_IF_curve', IF, 1, **{'ax':axs[1],'curr':curr_IF, 'node':node})
#     do('plot_FF_curve', FF, 1, **{'ax':axs[2],'rate':rate_FF, 'node':node,
#                                      'input':'ESp'})    
#     d=do('optimize', opt, 1, **{'ax':axs[3], 'x0':2000.0,'f':[node],
#                                    'x':['node.ESp.rate']})
#     
#     set_optimization_val(d, hist, **{'x':['node.ESp.rate'], 'node':node})
#     do('plot_hist_rates', hist, 0, **{'ax':axs[3], 'node':node})
#     
#     beautify(axs)
#     
#     pylab.show()
#     
# if __name__ == "__main__":
#     main()  