'''
Created on Sep 12, 2013

@author: lindahlm
'''
import csv, numpy, pylab
add='memory_consumption_logs/'
file_names=[
            add+'stat141006122938_test_model2.py_three.dat',
            ]
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
    
def plot_dat(ax, file_name):

    with open(file_name, 'rb') as datfile:
        l=[]
        for row in datfile:
            if len(row.split('|')[-1].split()):
                l.append(row.split('|')[-1].split())
#                 print  row
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
#                 print i
                row[i]=0.
        data.append([row[0]])
    data=zip(*data)
    data=numpy.array(data)
    shape=data.transpose().shape

    
    ax.plot(numpy.mgrid[0:shape[0]*10:10,0:1][0],
               100*(data.transpose()-data.transpose()[0,0])/(1533.0+59900.0))
    pylab.legend([lengend_names[0]])
    pylab.ylabel('Memory (MB)')
    pylab.xlabel('Time (sec)')
    pylab.show()
 
color=['b', 'g', 'r', 'm']
for file_name, c in zip(file_names, color):
    ax=pylab.subplot(111)
    plot_dat(ax,file_name)


#print csv.list_dialects()       
