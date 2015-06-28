'''
Created on May 12, 2014

@author: mikael
'''
import random
import time
import itertools
def test1():
    print "for loop with no multiproc: "
    m = 10000000
    t = time.time()
    for i in range(m):
        pick = random.choice(['on', 'off', 'both'])
    print time.time()-t

def test2():
    print "map with no multiproc: "
    m = 10000000
    t = time.time()
    map(lambda x: random.choice(['on', 'off', 'both']), range(m))
    print time.time()-t

def rdc(x):
    return random.choice(['on', 'off', 'both'])

def test3():
    from multiprocessing import Pool

    pool = Pool(processes=8)
    m = 10000000

    print "map with multiproc: "
    t = time.time()

    r = pool.map(rdc, range(m))
    print time.time()-t



if __name__ == "__main__":
#     test1()
#     test2()
#     test3()
    
    a_args = [1,2,3]
    second_arg = 1
    for i in itertools.izip(a_args, itertools.repeat(second_arg)):
        print i