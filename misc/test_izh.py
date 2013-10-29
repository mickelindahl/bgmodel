import nest
import pylab
nest.Install('/usr/local/lib/nest/ml_module') # Change ml_module to your module name

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
I_e=60.0
params = { 'a' : a, 'b_1' : b, 'b_2' : b, 'c' : c, 'C_m' : C,
           'd' : d,  'k' : k, 'V_b' : v_b, 'V_peak' : v_peak, 
           'V_r' : v_r, 'V_t' : v_t, 'I_e' : I_e}



nest.SetStatus(n, params)  
nest.Connect(mm,n)

nest.Simulate(1000)


status_mm=nest.GetStatus(mm)

v_m=status_mm[0]['events']['V_m']
times=status_mm[0]['events']['times']
pylab.plot(times, v_m)
pylab.show()


