'''
Created on May 21, 2014

@author: mikael
'''
import sys, time
temp = sys.stdout #store original stdout object for later
sys.stdout = open('log.txt','w', 0) #redirect all prints to this log file
print("testing123") #nothing appears at interactive prompt
# sys.stdout.flush()
time.sleep(15)
print("another line") #again nothing appears. It is instead written to log file
sys.stdout.close() #ordinary file object
sys.stdout = temp #restore print commands to interactive prompt
print("back to normal") #this shows up in the interactive prompt