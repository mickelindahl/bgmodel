'''
Created on Aug 12, 2013

@author: lindahlm
'''
from toolbox.network.default_params import Perturbation_list as pl

import numpy

import pprint

from core.toolbox import misc

pp=pprint.pprint



from oscillation_perturbations8 import get_solution_final_beta2



def get():

    

    

    l=[]
    d = {}
    solution=get_solution_final_beta2()

    misc.dict_update(d, solution['mul']) 

    l+=[pl(d, '*', **{'name':''})]

    d = {}
    misc.dict_update(d, solution['equal']) 


    l[-1]+=pl(d, '=', **{'name':'myname'}) 
      



    print l[0]

    for e in sorted(l[0]):

        print e

        



    return l





get()
