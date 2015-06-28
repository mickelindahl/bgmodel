'''
Created on Sep 19, 2014

@author: mikael
'''
from toolbox import data_to_disk
import sys
# print sys.argv
_, path, data=sys.argv
data = eval(data)

sd=data_to_disk.Storage_dic(path+'data')
sd.save_dic({'data1':data})