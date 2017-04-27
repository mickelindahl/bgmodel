# Create by Mikael Lindahl on 4/12/17.

from core.network.parameters.eneuro import EneuroPar
import pprint

pp = pprint.pprint
eneuro = EneuroPar()

eneuro.set({
    'simu': {
        'local_num_threads': 4,
        'path_data': eneuro.dic['simu']['path_data'] + 'project/data/simulation',
        'path_figure': eneuro.dic['simu']['path_data'] + 'project/fig/simulation',
        'path_nest': eneuro.dic['simu']['path_data'] + 'project/nest/simulation',
        'sd_params': {
            'to_file': True,
            'to_memory': False
        }
    }
})

dic = eneuro.dic
pp(dic)
# netw: global network parameters which is used by other keys. Dont look at the as simulation parameters
# node: parameters related to the population; network size, nuclei short names which corresponds to model name in 'nest' key
# conn: connection rules for node
# simu: simulation parameters; result directory, simulation time, ...
# nest: all parameters for nest
