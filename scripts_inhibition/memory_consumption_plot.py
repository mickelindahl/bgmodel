'''
Created on Sep 12, 2013

@author: lindahlm
'''
import csv, numpy, pylab

#file_name='memory_consumption_logs/stat130912174515.dat'
#file_name='memory_consumption_logs/stat130913085147.dat'
file_name='memory_consumption_logs/stat130913092123.dat'
#file_name='memory_consumption_logs/stat130912174515.csv'
#file_name='memory_consumption_logs/stat130913115358.dat'
#file_name='memory_consumption_logs/stat130913170305.dat'
#file_name='memory_consumption_logs/stat130913170316.dat'
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
    
    pylab.plot(data.transpose())
    pylab.legend(lengend_names)
    pylab.ylabel('Memory (MB)')
    pylab.ylabel('Time (sec)')
    pylab.show()
plot_dat(file_name)
#print csv.list_dialects()       
