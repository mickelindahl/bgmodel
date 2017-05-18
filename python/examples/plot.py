# Create by Mikael Lindahl on 4/27/17.

from core import data_to_disk

path = '/Users/Mohammad/Documents/PhD/Projects/BGmodel/bgmodel/results/example/simulate/data'
sd = data_to_disk.Storage_dic.load(path, ['Net_0'])
print sd
