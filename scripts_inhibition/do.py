'''
Created on Mar 19, 2014

@author: lindahlm

'''
import sys

from os.path import expanduser
HOME = expanduser("~")
sys.path.append(HOME+'/tmp/') #for testing

def do(attr, s, *args):
    module=__import__(attr)
    fun=module.main
    try:
        print 'running '+s
        fun(*args)
        print 'success'
    except Exception as e:
        
        print 'fail '+e.message
     
do(*sys.argv [1:])


