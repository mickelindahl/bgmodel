# Create by Mikael Lindahl on 4/2/17.

import nest
import pylab
import time


class Stopwatch():
    def __init__(self, *args, **kwargs):
        self.msg = args[0]
        self.args = args
        self.time = None

    def __enter__(self):

        self.time = time.time()
        print self.msg,


    def __exit__(self, type, value, traceback):
        t = round(time.time() - self.time, )
        msg_out = '... finnish {} {} sec '.format(self.msg, t)
        print msg_out

def show(multimeter, spikedetector):
    dmm = nest.GetStatus(multimeter)[0]

    Vms = dmm["events"]["V_m"]
    ts = dmm["events"]["times"]

    pylab.figure(1)
    pylab.plot(ts, Vms)

    dSD = nest.GetStatus(spikedetector,keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]
    pylab.figure(2)
    pylab.plot(ts, evs, ".")
    pylab.show()