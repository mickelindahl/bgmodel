# Module

This module contains the following models in addition to the standard ones 
 that `drop-odd-spike-connections`, `pif-psc-alpha` and `step-pattern-builder` MyModule in nest includes
 
- **my-aeif-cond-exp** A extended aeif-cond-exp with more synapse connections 
- **izhik_cond_exp** A implmentation of [izhikevich simple neuron model](http://www.izhikevich.org/publications/spikes.htm)
- **my_poisson_generator** Test implementation of nest poisson_generator model 
- **poisson_generator_dynamic** Poisson generator where a list of timings can be set together with rates that
 the poisson generator should change to at those timings
- **poisson_generator_periodic** Poisson generator that can be used to simulate periodic change in poisson firing rate

Note! Remember to have GSL installed
install C++ build-essential package