# Create by Mikael Lindahl on 4/12/17.

from core.network import structure
from core.network import engine
from core import my_nest, data_to_disk
from core.network.parameters.eneuro_beta import EneuroBetaPar
# from core.network.default_params import Beta
from core.network.parameters.eneuro import EneuroPar
from scripts_inhibition.base_oscillation import add_GI, add_GPe
import pprint
pp=pprint.pprint

def build(par):

    # ******
    # Build
    # ******
    surfs, pops = structure.build(par.get_nest(),
                                  par.get_surf(),
                                  par.get_popu())

    return surfs, pops


def connect(par, surfs, pops):


    args = [pops, surfs, par.get_nest(), par.get_conn(), True]

    structure.connect(*args)

def postprocessing(pops):

    d = {}

    signal_type='spike_signal'

    d_signals = pops.get(signal_type)

    for name, signal in d_signals.items():
        engine.fill_duds_node(d, name, signal_type, signal)

    add_GI(d)
    add_GPe(d)

    return d

def main():

    my_nest.ResetKernel()

    # Get parameters
    par = EneuroBetaPar(other=EneuroPar())


    # Configure simulation parameters
    par.set({
        'simu': {
            'local_num_threads': 8,
            'path_data': par.dic['simu']['path_data'] + 'example/simulate/data',
            'path_figure': par.dic['simu']['path_data'] + 'example/simulate/fig',
            'path_nest': par.dic['simu']['path_data'] + 'example/simulate/nest',
            'print_time':True,
            'sd_params': {
                'to_file': True,
                'to_memory': False
            }
        }
    })

    par.nest_set_kernel_status()

    # Clear nest data directory
    par.nest_clear_data_path({'display':True})

    # Show kernel status
    pp(my_nest.GetKernelStatus())

    # Create news populations and connections structures
    surfs, pops = build(par)

    # Connect populations accordingly to connections structure
    connect(par, surfs, pops)

    # Simulate
    my_nest.Simulate(500)

    # Create spike signals
    d = postprocessing(pops)

    # Save
    sd = data_to_disk.Storage_dic.load(par.dic['simu']['path_data'], ['Net_0'])
    sd.save_dic({'Net_0':d}, **{'use_hash': False})



if __name__ == '__main__':
    main()
