'''
Created on Jul 11, 2013

@author: lindahlm
'''
from __future__ import division
import numpy
from numpy import mean, hanning, cov
from numpy import zeros, ones, diagonal, transpose, matrix
from numpy import resize, sqrt, divide, array, concatenate
from numpy import convolve, dot, conjugate, absolute, arange, reshape #matrixmultiply, \
Float=numpy.float64 
Complex=numpy.complex 
     
from numpy.fft import fft

from scipy.signal import butter, lfilter, hilbert, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    
    # Nyquist frequency is pi radians / sample.
    nyq = 0.5 * fs
    # Low and high are the normalized frequencies
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_lfilter(data, lowcut, highcut, fs, order=5):
    '''
    fs         - sampling frequency
    lowcut     - low cut frequency
    highcut    - high cut frequency
    '''
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    
    #y = filtfilt(b, a, data)
    return y

def butter_bandpass_filtfilt(data, lowcut, highcut, fs, order=5):
    '''
    fs         - sampling frequency
    lowcut     - low cut frequency
    highcut    - high cut frequency
    '''
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    y = lfilter(b, a, y[::-1])
    return y[::-1]

    #y = filtfilt(b, a, data)
    #return y
    
def my_hilbert(x, N=None, axis=-1):
    return hilbert(x,N,axis)

def norm(x):
    return sqrt(dot(x,x))

def window_hanning(x):
    return hanning(len(x))*x

def window_none(x):
    return x

def detrend_mean(x):
    return x - mean(x)

def detrend_none(x):
    return x

def detrend_linear(x):
    """Remove the best fit line from x"""
    # I'm going to regress x on xx=range(len(x)) and return
    # x - (b*xx+a)
    xx = arange(len(x), typecode=x.dtype)
    X = transpose(array([xx]+[x]))
    C = cov(X)
    b = C[0,1]/C[0,0]
    a = mean(x) - b*mean(xx)
    return x-(b*xx+a)


def psd(x, NFFT=256, Fs=2, detrend=detrend_none,
        window=window_hanning, noverlap=0):
    """
    The power spectral density by Welches average periodogram method.
    The vector x is divided into NFFT length segments.  Each segment
    is detrended by function detrend and windowed by function window.
    noperlap gives the length of the overlap between segments.  The
    absolute(fft(segment))**2 of each segment are averaged to compute Pxx,
    with a scaling to correct for power loss due to windowing.  Fs is
    the sampling frequency.

    -- NFFT must be a power of 2
    -- detrend and window are functions, unlike in matlab where they are
       vectors.
    -- if length x < NFFT, it will be zero padded to NFFT
    

    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """

    if NFFT % 2:
        raise ValueError, 'NFFT must be a power of 2'

    # zero pad x up to NFFT if it is shorter than NFFT
    if len(x)<NFFT:
        n = len(x)
        x = resize(x, (NFFT,))
        x[n:] = 0
    

    # for real x, ignore the negative frequencies
    if x.dtype==Complex: numFreqs = NFFT
    else: numFreqs = NFFT//2+1
        
    windowVals = window(ones((NFFT,),x.dtype))
    step = NFFT-noverlap
    ind = range(0,len(x)-NFFT+1,step)
    n = len(ind)
    Pxx = zeros((numFreqs,n), Float)

    # do the ffts of the slices
    for i in range(n):
        thisX = x[ind[i]:ind[i]+NFFT]
        thisX = windowVals*detrend(thisX)
        fx = absolute(fft(thisX))**2
        Pxx[:,i] = fx[:numFreqs]

    # Scale the spectrum by the norm of the window to compensate for
    # windowing loss; see Bendat & Piersol Sec 11.5.2
    if n>1: Pxx = mean(Pxx,1)
    Pxx = divide(Pxx, norm(windowVals)**2)
    freqs = Fs/NFFT*arange(0,numFreqs)
    return Pxx, freqs

def csd(x, y, NFFT=256, Fs=2, detrend=detrend_none,
        window=window_hanning, noverlap=0):
    """
    The cross spectral density Pxy by Welches average periodogram
    method.  The vectors x and y are divided into NFFT length
    segments.  Each segment is detrended by function detrend and
    windowed by function window.  noverlap gives the length of the
    overlap between segments.  The product of the direct FFTs of x and
    y are averaged over each segment to compute Pxy, with a scaling to
    correct for power loss due to windowing.  Fs is the sampling
    frequency.

    NFFT must be a power of 2

    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """

    if NFFT % 2:
        raise ValueError, 'NFFT must be a power of 2'

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x)<NFFT:
        n = len(x)
        x = resize(x, (NFFT,))
        x[n:] = 0
    if len(y)<NFFT:
        n = len(y)
        y = resize(y, (NFFT,))
        y[n:] = 0

    # for real x, ignore the negative frequencies
    if x.dtype==Complex: numFreqs = NFFT
    else: numFreqs = NFFT//2+1
        
    windowVals = window(ones((NFFT,),x.dtype))
    step = NFFT-noverlap
    ind = range(0,len(x)-NFFT+1,step)
    n = len(ind)
    Pxy = zeros((numFreqs,n), Complex)

    # do the ffts of the slices
    for i in range(n):
        thisX = x[ind[i]:ind[i]+NFFT]
        thisX = windowVals*detrend(thisX)
        thisY = y[ind[i]:ind[i]+NFFT]
        thisY = windowVals*detrend(thisY)
        fx = fft(thisX)
        fy = fft(thisY)
        Pxy[:,i] = fy[:numFreqs]*conjugate(fx[:numFreqs])

    # Scale the spectrum by the norm of the window to compensate for
    # windowing loss; see Bendat & Piersol Sec 11.5.2
    if n>1: Pxy = mean(Pxy,1)
    Pxy = divide(Pxy, norm(windowVals)**2)
    freqs = Fs/NFFT*arange(0,numFreqs)
    return Pxy, freqs

def cohere(x, y, NFFT=256, Fs=2, detrend=detrend_none,
           window=window_hanning, noverlap=0):
    """
    cohere the coherence between x and y.  Coherence is the normalized
    cross spectral density

    Cxy = |Pxy|^2/(Pxx*Pyy)

    The return value is (Cxy, f), where f are the frequencies of the
    coherence vector.  See the docs for psd and csd for information
    about the function arguments NFFT, detrend, windowm noverlap, as
    well as the methods used to compute Pxy, Pxx and Pyy.

    """

    
    Pxx,f = psd(x, NFFT=NFFT, Fs=Fs, detrend=detrend,
              window=window, noverlap=noverlap)
    #print 'Pxx'
    Pyy,f = psd(y, NFFT=NFFT, Fs=Fs, detrend=detrend,
              window=window, noverlap=noverlap)
    #print 'Pyy'
    Pxy,f = csd(x, y, NFFT=NFFT, Fs=Fs, detrend=detrend,
              window=window, noverlap=noverlap)
    
    #print 'Pxy'
    Cxy = divide(absolute(Pxy)**2, Pxx*Pyy)
    
    #Null nans
    Cxy[numpy.isnan(Cxy)]=0
    return Cxy, f

def corrcoef(*args):
    """
    
    corrcoef(X) where X is a matrix returns a matrix of correlation
    coefficients for each row of X.
    
    corrcoef(x,y) where x and y are vectors returns the matrix or
    correlation coefficients for x and y.

    Numeric arrays can be real or complex

    The correlation matrix is defined from the covariance matrix C as

    r(i,j) = C[i,j] / (C[i,i]*C[j,j])
    """

    if len(args)==2:
        X = transpose(array([args[0]]+[args[1]]))
    elif len(args==1):
        X = args[0]
    else:
        raise RuntimeError, 'Only expecting 1 or 2 arguments'

    
    C = cov(X)
#    d = resize(diagonal(C), (2,1))
    d = matrix(resize(diagonal(C), (2,1)))
    r = divide(C,sqrt(d*transpose(d)))[0,1]
    try: return r.real
    except AttributeError: return r



def get_coherence(data1, data2, sampling_freq=1000, NFFT=2048, diagonal=False, noverlap=0):
    results_Cxy=[]
    #samplingfreq=1000.0
    
    #NFFT determines how many frequencies coherence can be calculated for
    for i, d1 in enumerate(data1):
        for j, d2 in enumerate(data2):
            if not i==j or diagonal:
                Cxy, f=cohere(d1,d2, NFFT=NFFT, Fs=sampling_freq, noverlap=noverlap)
                results_Cxy.append(Cxy)
            
    mean_Cxy=numpy.mean(numpy.array(results_Cxy), axis=0) 
    
    return f, mean_Cxy 


def get_cross_spectral_density(data1, data2, sampling_freq=1000, NFFT=2048, diagonal=False, noverlap=0):
    results_Pxy=[]
    f=[]
    #samplingfreq=1000.0
    
    #NFFT determines how many frequencies coherence can be calculated for
    for i, d1 in enumerate(data1):
        for j, d2 in enumerate(data2):
            if not i==j or diagonal:
                Pxy, f=csd(d1,d2, NFFT=NFFT, Fs=sampling_freq, noverlap=noverlap)
                results_Pxy.append(Pxy)
            
    mean_Pxy=numpy.mean(numpy.array(results_Pxy), axis=0) 
    return f, mean_Pxy 
    
    