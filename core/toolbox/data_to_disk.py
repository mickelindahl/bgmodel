'''
Created on Jun 18, 2013

@author: lindahlm
'''

import csv
import cPickle as pickle # Can be 1000 times faster than pickle
import numpy
import os
import unittest

from toolbox import misc
from toolbox.misc import Base_dic

from os.path import expanduser
HOME = expanduser("~")

class Storage(object):
    def __init__(self, *args, **kwargs):
        
        self.main_path=args[0]
        self.data_path=''
    
    def __eq__(self, other):
        b1=self.main_path==other.main_path
        b2=self.data_path==other.data_path
        return b1*b2
    
    def __getstate__(self):
        return self.__dict__
        
    def __setstate__(self, d):
        self.__dict__ = d    

    def get_data_path(self):
        return self.data_path

    def load_data(self):
        return pickle_load(self.data_path)
             
    def save_data(self, name, data):
        hash_code=str(hash(data))
        self.data_path=self.main_path+name+'-'+hash_code+'.pkl'
        pickle_save(data, self.data_path)

    def set_main_path(self, val):
        self.main_path=val

    def values(self):
        return self.data_paths.values()
                       
class Storage_dic(Base_dic):
    '''dictionary of storage objects'''
    
    def __setitem__(self, key, val):
        raise AttributeError('Can set item for Storage_dic, use add_storage')
    
    def _init_extra_attributes(self, *args, **kwargs):
        self.allowed=[]
        self.file_name=args[0] 
        self.directory= self.file_name+'/'   
        self.file_name_info=self.directory+'info.txt'
           

    def Factory_storage(self):
        return Storage(self.directory)

    def add_info(self, info):
        txt_save( info, self.file_name_info )

    def add_storage(self, keys):
        val=self.Factory_storage()
        self.dic=misc.dict_recursive_add(self.dic, keys, val)
    
        
    def clear(self):
        if os.path.isfile(self.file_name+'.pkl'):
            os.remove(self.file_name+'.pkl')        
        
        path=self.directory
        if os.path.isdir(path):
            l=os.listdir(path)
            l=[path+ll for ll in l]
            for p in l:
                os.remove(p)
            os.rmdir(path)
    
    def garbage_collect(self):
        
        files1=[self.file_name_info]
        for _, storage in misc.dict_iter(self.dic):
            files1.append(storage.get_data_path())
        
        mypath=self.directory
        if not os.path.isdir(mypath):
            return
            
        files2=[ os.path.join(mypath,f) for f in os.listdir(mypath) 
               if os.path.isfile(os.path.join(mypath,f)) ]
        
        for f in files2:
            if f in files1:
                continue
            else:
                os.remove(f)
                
    
    @classmethod
    def load(cls, file_name):
        '''It makes perfect sense that you should use foo=Foo.load(), 
        and not foo=Foo();foo.load(). for example, if Foo has some 
        variables that MUST be passed in to the init, you would need 
        to make them up for foo=Foo(); or if the init does some heavy 
        calculations of variables stored in the instance, 
        it would be for nothing. '''
        
        if os.path.isfile(file_name+'.pkl'):        
            return pickle_load(file_name)
        else:
            return Storage_dic(file_name)
    
#     @classmethod
    def load_dic(self, *filt):
              
        d={}
        for keys, storage in misc.dict_iter(self):
            if filt==():
                pass
            a=False
            for key in keys:
                if key not in filt:
                    a=True
            if a:
                continue
                       
            val=storage.load_data()
            d=misc.dict_recursive_add(d, keys, val)
                
        return d
            
    
    def save(self):
        pickle_save(self, self.file_name)
    
    def save_dic(self, d):

        for keys, data in misc.dict_iter(d):
            if not misc.dict_haskey(self.dic, keys):
                self.add_storage(keys)
        
            s=misc.dict_recursive_get(self.dic, keys)
            s.save_data('-'.join(keys), data)
        
        self.save()
        

    
def nest_sd_load(file_names):
    data=[]

    for name in file_names: 
        c=0
        while c<2:
            try:
                with open(name, 'rb') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter='\t')
                    for row in csvreader:
                        data.append([float(row[0]), float(row[1])])
                c=2
            except:
                name_split=name.split('-')
                name=name_split[0]+'-0'+name_split[1]+'-'+name_split[2]
                c+=1
    data=numpy.array(data)
    if len(data):
        return data[:,0], data[:,1]
    else:
        return numpy.array([]), numpy.array([])
     
        #with open(name,'rb') as f:
        #    f.read()


def mkdir(path):
    
    # If a directory does not exist where a file is suppose to be stored  it is created    
    path=path.split('/')
    i=len(path)
    while not os.path.isdir('/'.join(path[0:i])):
        i-=1
        
    while i!=len(path):
        if not os.path.isdir('/'.join(path[0:i+1])):
            os.mkdir('/'.join(path[0:i+1])) 
        i+=1
        
def dic_save(d, fileName):
    depth=misc.dict_depth(d)
    dr=misc.dict_reduce(d, deliminator=';')
    lines=[key+';'+str(val) for key, val in zip(dr.keys(), dr.values())]
    lines.sort()
    for line in lines:
        print(line)
    for i in range(len(lines)):
        lines[i]=lines[i]+';-'*(depth-len(lines[i].split(';')))
    
    s='\n'.join(lines)
    heading=';'.join(['p'+str(i) for i in range(depth)])
    s=heading+'\n'+s
    txt_save(s, fileName, 'csv')
    
    
    
    
    
def txt_save(text, fileName, file_extension='txt' ):
    
    fileName=fileName.split('/')
    if  '~' in fileName:
        fileName[fileName.index('~')]=expanduser("~")
    fileName='/'.join(fileName)    


    mkdir('/'.join(fileName.split('/')[0:-1]))    
    
    if 4<len(fileName) and fileName[-4:]!='.txt':
        fileName=fileName+'.txt'
    f=open(fileName, 'wb') #open in binary mode
     
#     f=open(fileName,'w')
    f.write(text)
    f.close()    

    
def txt_save_to_label(text, label_in, fileName ):
    mkdir('/'.join(fileName.split('/')[0:-1]))  
    if 4>len(fileName)or fileName[-4:]!='.txt':
        fileName=fileName+'.txt'
    
    if not os.path.isfile(fileName):  
        f=open(fileName,'w')
        f.close
    f=open(fileName,'r')
    lines=f.readlines()
    f.close()
    labels_list=[]
    for line in lines:
        labels_list.append(line.split(';')[-1].rstrip('\n'))
    
    i=0
    row_id=len(labels_list)
    for label in labels_list: 
        if label==label_in:
            row_id=i
        i+=1
    text=text +';'+label_in        
    txt_save_to_row(text, row_id, fileName)        
            
        
def txt_save_to_row(text, row_id, fileName):
    mkdir('/'.join(fileName.split('/')[0:-1]))    
    if 4>len(fileName) or fileName[-4:]!='.txt':
        fileName=fileName+'.txt'
    
    if not os.path.isfile(fileName):  
        f=open(fileName,'w')
        f.close
    
    f=open(fileName,'r')
    
    lines=f.readlines()
    write_text=["\n"]*(row_id+1)
    write_text[row_id]=text+'\n'
    i=0
    while i<len(lines):
        if i!=row_id:
            if i<len(write_text):
                write_text[i]=lines[i]
            else:
                write_text.append(lines[i])
        i+=1
    f.close()
    
    f=open(fileName, 'w')
    f.write(''.join(write_text))
    f.close()
         


def pickle_save(data, fileName):
    '''
    
    Arguments:
        data        - data to be pickled
        fileName    - full path or just file name
    '''

    fileName=fileName.split('/')
    if  '~' in fileName:
        fileName[fileName.index('~')]=expanduser("~")
    fileName='/'.join(fileName)    


    mkdir('/'.join(fileName.split('/')[0:-1]))    
    
    if 4<len(fileName) and fileName[-4:]!='.pkl':
        fileName=fileName+'.pkl'
    f=open(fileName, 'wb') #open in binary mode
    
    
    # With -1 pickle the list using the highest protocol available (binary).
    pickle.dump(data, f, -1)
    #cPickle.dump(data, f, -1)
    
    f.close()
    
def pickle_load(fileName):
    '''
    
    Arguments:
        fileName    - full path or just file name
    '''
    if 4<len(fileName) and fileName[-4:]!='.pkl':
        fileName=fileName+'.pkl'
    fileName=os.path.expanduser(fileName)
        
    f=open(fileName, 'rb') # make sure file are read in binary mode
    data=pickle.load(f)

    f.close()
    return data


def text_save(data, fileName):
    '''
    
    Arguments:
        data        - data to be pickled
        fileName    - full path or just file name
    '''
    try:
        f=open(fileName, 'wb') #open in binary mode
    except:
        parts=fileName.split('/')
        os.mkdir('/'.join(parts[0:-1]))    
        f=open(fileName, 'wb') 
    
    f.write(data)
    f.close()
    
def text_load(fileName):
    '''
    
    Arguments:
        data        - data to be pickled
        fileName    - full path or just file name
    '''
    f=open(fileName, 'r')
    data=f.read()
    f.close()
    return data


def read_f_name(data_path, contain_string = None):
    '''
    read in file names in a directory. Files can be filtered by extension
    with extension
    '''
    
    # Get file names in directory    
    file_list=os.listdir(data_path)
    
    # Get files with certain extension 
    if contain_string:
        new_file_list=[]
        for file_path in file_list:
            
            if contain_string in file_path.split('/')[-1]:
                
                new_file_list.append(file_path)
        
        file_list = new_file_list        
            
    return file_list        


def dummy_data():
    d={'net1': {'dummy1':{'string':('s'), 
                          'number':(1)},
                'dummy2':{'string':('t'), 
                          'number':(2)}},
       'net2': {'dummy1':{'statistics':{'mean':10,
                                        'CV':1.5}}}}
    return d

def dummy_data2():
    d={'net1': {'dummy1':{'string':('s')},
                'dummy2':{'string':('t')}}}
    return d   

def dummy_data3():
    d={'net1':{'dummy1':{'string':('s'), 
                         'number':(1)}}}
    return d


def dummy_data4():
    d={'net1': {'dummy1':{'string':('s'), 
                          'number':(1)},
                'dummy2':{'string':('u'), 
                          'number':(3)}},
       'net2': {'dummy1':{'statistics':{'mean':20,
                                        'CV':3.}}}}
    return d

class TestStorage(unittest.TestCase):
    def setUp(self):
        self.data=[('s'), (1)]
        self.name_data=['string', 'number']
        self.main_path=HOME+'/tmp/data_to_disk_unittest/'
        
    def test_create(self):
        s=Storage(self.main_path)
        self.assertTrue(s.main_path==self.main_path)
    
    def test_load_data(self):
        self.test_sava_data()
        for s, data in zip(self.s, self.data):
            d=s.load_data()
            self.assertEqual(d, data)
    
    def test_sava_data(self):
        self.s=[]
        for name, data in zip(self.name_data, self.data):
            s=Storage(self.main_path)
            s.save_data(name, data)
            self.s.append(s)
            
        for name, data in zip(self.name_data, self.data):
            s=str(hash(data))
            self.assertTrue(os.path.isfile(self.main_path+name+'-'+s+'.pkl'))
            
    def tearDown(self):
        if os.path.isdir(self.main_path):
            l=os.listdir(self.main_path)
            l=[self.main_path+ll for ll in l]
            for path in l:
                os.remove(path)
            os.rmdir(self.main_path)
        
class TestStorage_dic(unittest.TestCase):
    def setUp(self):
        self.data=dummy_data()
        self.file_name=HOME+'/tmp/data_to_disk_unittest'
        self.directory=HOME+'/tmp/data_to_disk_unittest/'
        self.s=None
        
    def test_add_info(self):
        sd=Storage_dic(self.file_name)
        info='test'
        sd.add_info(info)
        self.assertTrue(os.path.isfile(self.directory+'info.txt'))
        
        
    def test_create(self):
        s=Storage_dic(self.file_name)
        self.assertTrue(s.file_name==self.file_name)
        
    def test_clear(self):
        self.test_save()
        self.assertTrue(os.path.isfile(self.file_name+'.pkl'))
        self.sd.clear()
        self.assertFalse(os.path.isfile(self.file_name+'.pkl'))
        
    def test_garbage_collect(self):
        d1=dummy_data()
        d2=dummy_data4()
        self.s=Storage_dic(self.file_name)
        self.s.add_info('test')
        self.s.save_dic(d1)
        self.s.save_dic(d2)
        
        d3=self.s.load_dic(self.file_name)
        
        self.assertDictEqual(d2, d3)
        
        mypath=self.s.directory
        files=[ f for f in os.listdir(mypath) 
               if os.path.isfile(os.path.join(mypath,f)) ]     
        self.assertEqual(len(files), 11)
        self.s.garbage_collect()
        files=[ f for f in os.listdir(mypath) 
               if os.path.isfile(os.path.join(mypath,f)) ]   
        self.assertEqual(len(files),7)  
        

    def test_load(self):
        self.test_save()
        sd=Storage_dic.load(self.file_name)
        self.assertEqual(self.sd,sd)

    def test_load_dic(self):
        self.test_save()
        d1=self.sd.load_dic(self.file_name, *['net1', 'net2',
                                            'dummy1', 'dummy2',
                                            'string','number','statistics'])
        self.assertDictEqual(d1, self.data)
        
        d2=dummy_data2()
        d3=self.sd.load_dic(self.file_name, *['net1', 
                                            'dummy1', 'dummy2',
                                            'string'])
        self.assertDictEqual(d2, d3)
        
        d4=dummy_data3()
        d5=self.sd.load_dic(self.file_name, *['net1', 'dummy1',
                                             'string','number'])
        self.assertDictEqual(d4, d5)
        
        
        
    def test_save(self):
        self.sd=Storage_dic(self.file_name)
        for keys, val in misc.dict_iter(self.data):
            
            #Add storage objects
            self.sd.add_storage(keys)
            s=misc.dict_recursive_get(self.sd, keys)
            key='-'.join(keys)
            s.save_data(key,val)
        
        self.sd.save()
        self.assertTrue(os.path.isfile(self.file_name+'.pkl'))
        self.assertTrue(os.path.isdir(self.file_name+'/'))
        
    def test_save_dic(self):
        self.s=Storage_dic(self.file_name)
        self.s.save_dic(self.data)
        self.assertTrue(os.path.isfile(self.file_name+'.pkl'))
    
        

    def tearDown(self):
        if os.path.isfile(self.file_name+'.pkl'):
            os.remove(self.file_name+'.pkl')        
        
        path=self.file_name+'/'
        if os.path.isdir(path):
            l=os.listdir(path)
            l=[path+ll for ll in l]
            for p in l:
                os.remove(p)
            os.rmdir(path)
            
            
if __name__ == '__main__':
    
    test_classes_to_run=[
                         TestStorage,
                         TestStorage_dic
                        ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)  