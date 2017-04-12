# Create by Mikael Lindahl on 4/12/17.

from core.network.parameters.eneuro import EneuroPar

eneuro = EneuroPar()
dic = eneuro.dic
print dic
# netw: global network parameters which is used by other keys. Dont look at the as simulation parameters
# node: parameters related to the population; network size, nuclei short names which corresponds to model name in 'nest' key
# conn: connection rules for node
# simu: simulation parameters; result directory, simulation time, ...
# nest: all parameters for nest


