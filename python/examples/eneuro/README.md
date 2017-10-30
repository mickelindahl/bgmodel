# Eneuro example
 
This examples run the four types of input scenarios activation control, 
activation dopamine depletion (beta added), slow wave and slow wave dopamine depletion 
from ["Untangling basal ganglia network dynamics and function – role of dopamine depletion and inhibition investigated in a spiking network model"](http://eneuro.org/content/early/2016/12/22/ENEURO.0156-16.2016). The population size can be specified.  

Output data
 * firing-rate/* - firing rate plots for each nuclei
 * parameters.pkl - pickled dictionary of  parameters that defined the
 model. `{'netw':{network pars}, 'simu':{simulation pars}, 'nest':{nest pars}, 'node':{node pars}, 'conn':{connection pars}}, `
 * randomized-params.json - for each nuclei and node the values of the randomized neuron parameters
 * randomized-params.png - scatter plots of randomized parameters for each 
 * scatter/* - Scatter plots for each nuclei 
 * statistics.json - rate and CV data for nuclei
 * statistics.png - rate and CV data plot for nuclei 
 
Note! environment variable `BGMODEL_HOME={model directory}` need to be set

## Run
To run the example simply do
```
python main {size}
```

The results can be found in 
```
{bgmodel root}/results/example/{size}
```

