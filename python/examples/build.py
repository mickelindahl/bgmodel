# Create by Mikael Lindahl on 4/12/17.

from core.network import structure
from core.network.parameters.eneuro import EneuroPar

def do():
    par = EneuroPar()

    # ******
    # Build
    # ******
    surfs, pops = structure.build(par.get_popu(),
                                  par.get_surf(),
                                  par.get_popu())

    return surfs, pops


if __name__ == '__main__':
    surfs, pops = do()
    print 'surfs', surfs
    print 'pops', pops