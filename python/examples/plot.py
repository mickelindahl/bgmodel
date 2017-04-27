# Create by Mikael Lindahl on 4/27/17.

from core import data_to_disk

path = '/home/mikael/git/bgmodel/results/example/simulate/data'
sd = data_to_disk.Storage_dic.load(path, ['Net_0'])
print sd