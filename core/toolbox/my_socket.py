'''
Created on Sep 28, 2014

@author: mikael
'''

from socket import *
import socket
def determine_host():

    HOST=socket.gethostname().split('.')
    if (len(HOST)==1 and HOST[0]!='supermicro'):
        return 'milner'
    if HOST[0][0:6]=='milner':
        return 'milner_login'
    else:
        return HOST[0]
