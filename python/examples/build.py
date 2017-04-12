# Create by Mikael Lindahl on 4/12/17.

from core.network import structure
from core import my_nest
from core.network.parameters.eneuro_beta import EneuroBetaPar
# from core.network.default_params import Beta
from core.network.parameters.eneuro import EneuroPar

def build():
    par = EneuroBetaPar(other=EneuroPar())

    my_nest.ResetKernel(local_num_threads=2, print_time=True)

    # ******
    # Build
    # ******
    surfs, pops = structure.build(par.get_nest(),
                                  par.get_surf(),
                                  par.get_popu())

    return surfs, pops


def connect(pops, surfs):


    args = [pops, surfs, par.get_nest(), par.get_conn(), True]

    structure.connect(*args)

def main():

    surfs, pops = build()

    connect(surfs, pops)

    my_nest.Simulate(1000)


if __name__ == '__main__':
    surfs, pops = do()
    print 'surfs', surfs
    print 'pops', pops
