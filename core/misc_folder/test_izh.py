import nest
import pylab
MODULE_PATH=  '/afs/nada.kth.se/home/w/u1yxbcfw/tools/NEST/dist/install-nest-2.2.2/lib/nest/ml_module'
nest.Install(MODULE_PATH) # Change ml_module to your module name

n=nest.Create('izhik_cond_exp')
mm=nest.Create('multimeter')
nest.SetStatus(mm, {'interval': 0.1, 'record_from': ['V_m']})
                  
# Params from Izhikevich regular spiking
a=0.03
b=-2.0
c=-50.
C=100.0
d=100.0
k=0.7
v_b=-60.
v_r=-60.
v_t=-40.
v_peak=35.
I_e=0.0
params = { 'a' : a, 'b_1' : b, 'b_2' : b, 'c' : c, 'C_m' : C,
           'd' : d,  'k' : k, 'V_b' : v_b, 'V_peak' : v_peak, 
           'E_L' : v_r, 'V_th' : v_t, 'I_e' : I_e}


pg=nest.Create('poisson_generator', params={'rate':10.0})
nest.SetStatus(n, params)  
nest.Connect(mm,n)
nest.Connect(pg, n, params={'weight':10., 'receptor_type':1})
nest.Connect(pg, n, params={'weight':-10., 'receptor_type':1})

nest.Simulate(1000)


status_mm=nest.GetStatus(mm)

v_m=status_mm[0]['events']['V_m']
times=status_mm[0]['events']['times']
pylab.plot(times, v_m)
pylab.show()


