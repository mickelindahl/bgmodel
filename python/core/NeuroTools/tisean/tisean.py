# tisean wrapper

import sys, os, shutil, string, re, tempfile
import urllib, random
from numpy import *


def list_tisean_functions():

    func = '''
    addnoise     choose         henon      lyap_spec  poincare                        rbf         upo
    arima-model  cluster        histogram  lzo-gm     polyback                        recurr      upoembed
    ar-model     compare        ikeda      lzo-run    polynom                         resample    wiener1
    ar-run       corr           intervals  lzo-test   polynomp                        rescale     wiener2
    autocor      d2             lazy       makenoise  polypar                         rms         xc2
    av-d2        delay          lfo-ar     mem_spec   predict                         sav_gol     xcor
    boxcount     endtoend       lfo-run    mutual     project                         spectrum    xrecur
    c1           events         lfo-test   notch      randomize_auto_exp_random       spikeauto   xzero
    c2d          extrema        lorenz     nrlazy     randomize_autop_exp_random      spikespec
    c2g          false_nearest  low121     nstat_z    randomize_spikeauto_exp_random  stp
    c2naive      fsle           lyap_k     pc         randomize_spikespec_exp_event   surrogates
    c2t          ghkss          lyap_r     pca        randomize_uneven_exp_random     timerev
    '''
    print func


# functions to generate dataseries
def generate_fetchrandnum(num=100, min=0, max=1, col=1):
    """ returns random numbers from www.random.org webinterface
        num: max. 10.000
        min: min. -1.000.000.000
        max: max.  1.000.000.000
        col: format in these columns
    """
    randomnumbers = []
    # put command together and open url
    cmd = 'num='+str(num)+'&min='+str(min)+'&max='+str(max)+'&col='+str(col)
    randomobject = urllib.urlopen('http://www.random.org/cgi-bin/randnum?'+cmd)
    # get each line and close url
    for row in randomobject.readlines():
        randomnumbers.append(map(float, string.split(row)))
    randomobject.close()
    # create new array
    return array(randomnumbers, Float)


def generate_henon(number = 1000,
                    parameter_a = 1.4, parameter_b = 0.3,
                    initial_x = 0.2, initial_y = 0.3):
    """ generates a henon map
    """
    # check parameters
    parameter_a = checkparameterarray(number,parameter_a)
    parameter_b = checkparameterarray(number,parameter_b)
    # create dataseries
    res = array(zeros((number,2)),Float)
    res[0,0] = initial_x
    res[0,1] = initial_y
    for i in range(0,number-1):
        res[i+1,0] = 1 - parameter_a[i] * (res[i,0]**2) + parameter_b[i] * res[i,1]
        res[i+1,1] = res[i,0]
    return res

def generate_henonx(maps = 1, number = 1000,
                    parameter_a = 1.4, parameter_b = 0.3,
                    initials = [[0.2, 0.3]]):
    """ generates a multiple of henon maps
    """
    initials = array(initials)
    res = array(zeros((number,2)),Float)
    for i in range(maps):
        res[:] = res[:] + generate_henon(number, parameter_a, parameter_b, initials[i,0], initials[i,1])[:]
    return res

def generate_arma(number=10000, coeffs_a = [0.2,0.98], values_x = [0,0], coeffs_b = [0.2,0.98], stddev = 1, rndfile = None):
    """ generates an arma process of length 'number'

        number: number of values to create
        coefficients_a: array of coefficients a, determines the order of the arma process, order (a[n-1], a[n-2], ..., a[1])
        values_x      : start values of variable x, order (x[n-1], x[n-2], ... , x[1])
        coefficients_b: compare coefficients_a, order(b[n], b[n-1], ... , b[1]), this is not allowed to be longer than coefficients_a
    """
    # check parameters and create parameter arrays
    length_a = len(coeffs_a)
    coefficients_a = array(zeros([number,length_a]),Float)
    for i in range(length_a):
        coefficients_a[:,i] = checkparameterarray(number,coeffs_a[i])[:]
    length_b = len(coeffs_b)
    coefficients_b = array(zeros([number,length_b]),Float)
    for i in range(length_b):
        coefficients_b[:,i] = checkparameterarray(number,coeffs_b[i])[:]
    # prepare prevalues_x
    res = array(zeros(number),Float)
    for i in range(min(len(values_x),length_a)):
        res[length_a-1-i] = values_x[i]
    # prepare random numbers
    if rndfile == None:
        rnd = array(zeros(number),Float)
        for i in range(number):
            rnd[i] = random.gauss(0,stddev)
    else:
        rnd = data_fload(rndfile)[:,0]
        rnd[:] = rnd[:]
    # create dataseries
    for i in range(length_a,number):
        # AR part
        firstsum = 0
        for j in range(length_a):
            firstsum = firstsum + coefficients_a[i,j] * res[i-1-j]
        # MA part
        secondsum = 0
        for j in range(length_b):
            secondsum = secondsum + coefficients_b[i,j] * rnd[i-j]
        # put AR and MA together
        res[i] = firstsum + secondsum
    return res

def generate_monoton(number=1000, start = 0, stop = 1, step = 100):
    """ generates a monotonically increasing or decreasing series of values, at each step increased
        
        number : number of values to create
        start  : start value for increase
        stop   : stop value for increase
        step   : make a change after 'step' number of values
    """
    res = array(zeros((number)), Float)
    increase = float((stop - start)) / float((number / step))
    for i in range(number):
        if (i / step) == 0:
            res[i] = start
        else:
            res[i] = start + (i / step) * increase
    return res

def generate_tent(number=1000, start = 0, peak = 1, step = 100):
    """ generates a tent map from 'start' to 'peak' and back to 'start'
        
        number : number of values to create
        start  : start value for tent map
        peak   : peak value for tent map
        step   : make a change after 'step' number of values
    """
    res = array(zeros((number)), Float)
    increase = float(peak - start) / float((number / 2)/ step)
    for i in range(number / 2):
        res[i] = start + (i / step) * increase
    for i in range((number / 2) + (number % 2)):
        # (number % 2) to respect odd 'number'
        res[i+(number / 2)] = peak - (i / step) * increase
    return res   

### methods to simply modify the dataseries

def modify_log(arr):
    """ takes the log
        arr: array for operation
        retruns a new array
    """
    resultarr = array(log(arr[:]))
    return resultarr

def modify_logdif(arr):
    """ takes the log and then the diff between following values, last element = 0.0
        arr: array for operation
        returns a new array
    """
    resultarr = modify_log(arr)
    resultarr[:-1] = resultarr[1:] - resultarr[:-1]
    resultarr[-1:] = 0.0
    return resultarr

def modify_exp(arr, exp):
    """ takes the exponent of the named column
        arr: array for operation
        exp: the exponent to take
        returns a new array
    """
    resultarr = array(arr[:]**exp)
    return resultarr

### helper functions

def checkparameterarray(number,parameter):
    """ checks 'parameter' to be an array or not and returns an array anyway
    """
    try:
        len(parameter)
    except StandardError:
        # convert to constant array
        parameter = array(zeros(number)+parameter, Float)
    else:
        # check length
        if len(parameter) < number:
            tempparameter = zeros(number,Float)
            tempparameter[:len(parameter)] = parameter[:len(parameter)]
            tempparameter[len(parameter):] = parameter[len(parameter)-1]
            parameter = tempparameter
    return parameter

def etedataparser(filename):
    """ parses the endtoend datafile - a horrible task
    """
    
    # read endtoend data
    etedaten = []
    file  = open(filename,'r')
    for line in file.readlines():
        if string.find(str(line),'length:') <> -1:
            etedaten.append(line)

    # parse endtoend data, actually quite ugly
    index = []
    for line in etedaten:
        hilfe = string.split(line, ':')
        length = string.split(hilfe[1], 'offset')
        offset = string.split(hilfe[2], 'lost')
        index.append(int(length[0]), int(offset[0]))
    length = index[-1][0]
    offset = index[-1][1]

    return length, offset

def rmsvalues(inputfile):
    """ returns the mean, rms, min and max value from a file produced by rms
    """
    listarray = []
    file =open(inputfile,'r')
    for line in file.readlines():
        listarray.append(string.split(line))
    file.close

    mean = float(listarray[0][0])
    rms = float(listarray[0][1])
    min = float(listarray[0][2])
    max = float(listarray[0][3])

    return mean,rms,min,max

def resultprederror(inputfile, key = 'data', surr=19):
    """ checks, whether test rejected null hypothesis or not
    
        return value: -1 = rejected lower bound, 1 = rejected upper bound, 0 = not rejected
    """
    myarray,off,off = tisean.prederrorarray(inputfile, key=key , surr=surr)
    maxarray = argmax(myarray,0)
    minarray = argmin(myarray,0)
    if  maxarray[0] == maxarray[1]:
        return 1
    elif minarray[0] == maxarray[1]:
        return -1
    else:
        return 0

def prederrorarray(inputfile, key='', found=2, notfound=1, rms=1, surr=19):
    """ creates an array with the prediction error in the first column and 2 or 1 in the second depending on whether key was found or not

        return values: array of formatted data, mean and std deviation of surrogates (notfound)
    """
    # read file
    listarray = []
    file = open(inputfile, 'r')
    for line in file.readlines():
        listarray.append(string.split(line))
    file.close()
    # find key
    surrarray = []
    dataarray = []
    surrcount = 1
    for line in listarray:
        if line[2] == key:
            dataarray.append([float(line[1]),found])
            dataarray[-1][0] = dataarray[-1][0]/rms
        elif surrcount <= surr:
            surrarray.append([float(line[1]),notfound])
            surrarray[-1][0] = surrarray[-1][0]/rms
            surrcount = surrcount + 1
    # calculate mean and standard deviation
    m = mean(surrarray)[0]
    s = std(surrarray)[0]
    # merge the two arrays
    for line in dataarray:
        surrarray.append(line)
    # return array
    return surrarray, m, s

### methods to access original tisean programs

def foperation(cmd, options, inputfile, outputfile):
    """ performs any operation on the inputfile given as an argument and stores the output in outputfile
        cmd: executable to call
        options: append these options to call
        inputfile: datafile with input data
        outputfile: datafile with results of operation
        no return value
    """
    cmdstr = cmd+' '+options+' '+inputfile+' -o '+outputfile
    result = os.system(cmdstr) # FIXME: result abfange

def operation(arr, cmd, options):
    """ performs a tisean operation on the array
        arr: the input data as array
        cmd: executable to call
        options: append these options to call
        returns the resulting array value (read from stdout)
        """
    # create a temporary file and write array content to it
    tmpfile = string.replace(tempfile.mktemp(),"\\","/")
    data_save(arr, tmpfile)
    # process the operation on the file
    foperation(cmd, options, tmpfile, tmpfile)
    #cmdstr = cmd+' '+options+' '+tmpfile+' -o '+tmpfile
    #result = os.system(cmdstr) # FIXME: result abfangen
    # read it back and delete it
    arr = data_fload(tmpfile)
    os.remove(tmpfile)
    return arr

### some helper functions (IO)

def data_fload(inputfile):
    """ fast array loader, uses TableIO
        inputfile: name of file to read data from
        returns array (Numeric)
    """
    # open file, transfer filtered content into list listarray
    try:
        import TableIO
    except ImportError:
        return data_load(inputfile, check=1)
    else:
        return TableIO.readTableAsArray(inputfile,'#')

def data_load(inputfile, check=1):
    """ slower, truely python version to load a file
        inputfile:
        check: 1 -> check file, 0 -> just read in (faster, if data is correct)
        returns array (Numeric)
    """
    listarray = []
    if (check == 1):
        p1 = re.compile('[^\s.+-eE0-9]') # something else than numbers
        p2 = re.compile('[^\s]') # something else than empty lines
        file = open(inputfile, 'r')
        for line in file.readlines():
            # dropt line if something else than numbers or only spaces appear
            if not p1.search(line) and p2.search(line):
                listarray.append(map(float, string.split(line)))
        file.close()
    else:
        file = open(inputfile, 'r')
        for line in file.readlines():
            listarray.append(map(float,string.split(line)))
        file.close()
    # convert list listarray into an array (Numpy)
    return array(listarray, Float)
 
def data_fsave(arr, outputfile):
    """ fast, but unprecise way to store arrays to disk
        arr: array to store
        outputfile:
        no return value
        suitable for integer and short float numbers.
    """
    try:
        import TableIO
    except ImportError:
        data_save(arr,outputfile)
    else:
        TableIO.writeArray(outputfile, arr)
        
def data_save(arr, outputfile):
    """ slowly, but precise way to store data to disk
        arr: array to store
        outputfilename: 
    """
    # open file, format array and save it to disk
    file = open(outputfile, 'w')
    # if array's shape is only 1 we mustnt join the dataline
    if len(shape(arr)) == 1:
        for line in arr:
            file.write(str(line)+'\n')
    if len(shape(arr)) == 2:    
        for line in arr:
            file.write(string.join(map(str, line))+'\n')
    file.close()

