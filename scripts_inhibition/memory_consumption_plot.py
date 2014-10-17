'''
Created on Sep 12, 2013

@author: lindahlm
'''
import csv, numpy, pylab
add='memory_consumption_logs/'
file_name=add+'stat141002105923.dat'
file_name=add+'stat141002114940_simulate_beta_ZZZ112.py.dat'
file_name=add+'stat141006200929_simulate_beta_ZZZ112.py_.dat'
# file_name=add+'stat141013112617_simulate_beta_ZZZ112.py_.dat'
# file_name=add+'stat141013114525_simulate_beta_ZZZ112.py_.dat'
# file_name=add+'stat141013125013_simulate_beta_ZZZ112.py_.dat'
# file_name=add+'stat141014180811_simulate_beta_ZZZ112.py_okt14.dat'
# file_name=add+'stat141014181430_simulate_beta_ZZZ112.py_okt14.dat'
file_name=add+'stat141015134146_simulate_beta_ZZZ112.py_come_on.dat'
file_name=add+'stat141015232809_simulate_beta_ZZZ112.py_okt15.dat'
# file_name=add+'stat141016001250_simulate_beta_ZZZ112.py_okt16.dat'
# file_name=add+'stat141016130856_simulate_beta_ZZZ112.py_okt16.dat'

def plot_csv(file_name):

    with open(file_name, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        l=[]
        for row in spamreader:
            l.append(row)
            print  row
    lengend_names=l[6][0].split(',')[1:]
    l=l[7:]
    data=[]
    for row in l:
        data.append([float(d)/10**6 for d in row[1].split(',')[1:]])
    
    print l
    print numpy.array(data)
    pylab.plot(numpy.array(data))
    pylab.legend(lengend_names)
    pylab.ylabel('Memory (MB)')
    pylab.ylabel('Time (sec)')
    pylab.show()
    
def plot_dat(file_name):

    with open(file_name, 'rb') as datfile:
        l=[]
        for row in datfile:
            if len(row.split('|')[-1].split()):
                l.append(row.split('|')[-1].split())
                print  row
    lengend_names=l[1]
    l=l[2:]
    data=[]
    for row in l:
        for i in range(len(row)):
            try:
                type=row[i][-1]
                row[i]=float(row[i][:-1])
                if type=='G':
                    row[i]*=1000.0
            except:
                print i
                row[i]=0.
        data.append(row)
    data=zip(*data)
    data=numpy.array(data)
    print data.transpose().shape
    shape=data.transpose().shape
    print numpy.mgrid[0:shape[0]*10:10,0:4][0].shape
    pylab.plot(numpy.mgrid[0:shape[0]*10:10,0:4][0],
               data.transpose())
    pylab.legend(lengend_names)
    pylab.ylabel('Memory (MB)')
    pylab.xlabel('Time (sec)')
    pylab.show()
plot_dat(file_name)
#print csv.list_dialects()       
