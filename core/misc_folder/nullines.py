'''
Created on Aug 21, 2013

@author: lindahlm
'''
import numpy
import pylab

p_aeif={
             'a':2.5,    # I-V relation # I-V relation
             'b':70.,   # I-F relation
             'C':40.,  # t_m/R_in
             'delta_T':1.7,                      
             'E_L':-55.1, # v_t    = -56.4                                                               # 
             'gleak':1.,
             'tau_w':20.,# I-V relation, spike frequency adaptation
             'v_peak':15.,  # Cooper and standford
             'V_reset':-60.,  # I-V relation
             'v_T':-54.7,
             'v_a':-55.1} # I-V relation


I_null_aeif= lambda V,V_th: -p_aeif['gleak']*(V-p_aeif['E_L'])+p_aeif['gleak']*p_aeif['delta_T']*numpy.exp((V-V_th)/p_aeif['delta_T'])-p_aeif['a']*(V-p_aeif['E_L'])

V_vec=range(-70,-30, 1) 
V_th_vec=numpy.linspace(p_aeif['v_T']-10., p_aeif['v_T']+10., 5)


VV=[]
II=[]
for V_th in V_th_vec:
    II.append([I_null_aeif(V,V_th) for V in V_vec])
    VV.append(V_vec)


VV=numpy.array(VV)    
II=numpy.array(II)    

pylab.plot(VV.transpose(),II.transpose())
pylab.ylim([-50,200])
pylab.show()