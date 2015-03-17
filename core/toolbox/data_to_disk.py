'''
Created on Jun 18, 2013

@author: lindahlm
'''


import cPickle as pickle # Can be 1000 times faster than pickle
import numpy
import os
import subprocess
import unittest

import toolbox.misc as misc
import toolbox.my_nest as my_nest # need to have it othervice imprt crashes somtimes
from toolbox.misc import Base_dic
from toolbox.parallelization import comm, Barrier
from os.path import expanduser
HOME = expanduser("~")


# if comm.is_mpi_used():
#     import mpi4py
#     open=mpi4py.MPI.File.Open


# def open(*args):
#     
#     if comm.is_mpi_used():    
#         f=comm.open(*args)
#     else:
#         f=__builtins__.open(*args)
#     
#     
#     return f

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
             
    def save_data(self, name, data, use_hash=False):
        hash_code=''
        if use_hash:
            hash_code='-'+str(hash(data))
        
        self.data_path=self.main_path+name+hash_code+'.pkl'
        
        pickle_save(data, self.data_path)


    def set_main_path(self, val):
        self.main_path=val
    
    def set_data_path(self, val):
        self.data_path=val

    def values(self):
        return self.data_paths.values()

class Storage_dic2(Base_dic):
    '''dictionary of storage objects'''
    
    def __setitem__(self, key, val):
        raise AttributeError('Can set item for Storage_dic, use add_storage')
    
    def _init_extra_attributes(self, *args, **kwargs):
        self.allowed=[]
        self.file_name=args[0] 
        self.nets=args[1]
        self.directory= self.file_name+'/'   
        self.file_name_info=self.directory+'info.txt'
           

    @property
    def file_names(self):
        return [self.filename+net for net in self.nets]

    @property
    def directories(self):
        return [self.directory+net+'/' for net in self.nets]

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
    
    def delete(self, k):
        del self.dic[k]
    
    def force_update(self, file_name):
        
        self.file_name=file_name
        self.directory=file_name+'/'
        
        for _, storage in misc.dict_iter(self.dic):
            storage.set_main_path(self.directory)
            file_name=storage.get_data_path().split('/')[-1]
            storage.set_data_path(self.directory+file_name)
    

    def _garbage_collect_mpi(self):
        with Barrier():
            if comm.rank()==0:
                self._garbage_collect()
                
            
    def garbage_collect(self):
        if comm.is_mpi_used():
            self._garbage_collect_mpi()
        else:
            self._garbage_collect()
            
    def _garbage_collect(self):
        
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

    def save_fig(self, *args):
        if comm.is_mpi_used():
            with Barrier():
                if comm.rank()==0:
                    self._save_fig(*args)
        else:
            self._save_fig(*args)
                    

     
    def _save_fig(self, fig, extension='', format='svg'):
        
        with misc.Stopwatch('Saving figure...'):
            fig.savefig( self.file_name +extension+'.'+format, 
                         format = format) 
    
    def save_figs(self, figs, extension='', format='svg', in_folder=''):
        
        path='/'.join(self.file_name .split('/')[0:-1])+'/'
        name=self.file_name .split('/')[-1]
        
        if in_folder:
            path+=in_folder+'/'
            
        if not os.path.isdir(path):
            mkdir(path)
        
        with misc.Stopwatch('Saving figures...'):
            for i, fig in enumerate(figs):
                fig.savefig( path+name +extension+'_{0:0>4}'.format(i)+'.'+format, 
                             format = format)              
    
    def set_file_name(self, val):
        self.file_name=val

    def set_directory(self, val):
        self.directory=val
    @classmethod
    def load_mpi(cls, file_name, nets):
        '''It makes perfect sense that you should use foo=Foo.load(), 
        and not foo=Foo();foo.load(). for example, if Foo has some 
        variables that MUST be passed in to the init, you would need 
        to make them up for foo=Foo(); or if the init does some heavy 
        calculations of variables stored in the instance, 
        it would be for nothing. '''
        
        if os.path.isfile(file_name+'.pkl'):   

                
#             cls.file_name=file_name
#             cls.directory=file_name+'/'
            out={}
            for net in nets:   
                file_name=file_name+'/'+net
                d= pickle_load(file_name) 
                d.force_update(file_name)
                misc.dict_update(out, d)
            return out
        else:
            return Storage_dic2(file_name, nets)
                    
#     @classmethod
    def load_dic(self, *filt):
              
        d={}
        for keys, storage in misc.dict_iter(self):
            if filt==():
                pass
            else:
                a=False
                i=0
                for key in keys:
                    if key not in filt:
                        a=True
                    i+=1
                    if i==3:
                        break
                if a:
                    continue
                       
            val=storage.load_data()
            d=misc.dict_recursive_add(d, keys, val)
                
        return d
            
    
    def save(self):
        pickle_save(self, self.file_name)
    
    def save_dic(self, d, **k):


        for keys, data in misc.dict_iter(d):
            
            if not misc.dict_haskey(self.dic, keys):
                self.add_storage(keys)
        
            s=misc.dict_recursive_get(self.dic, keys)
            s.save_data('-'.join(keys), data, **k)
        
        self.save()
                  
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
    
    def delete(self, k):
        del self.dic[k]
    
    def force_update(self, file_name):
        
        self.file_name=file_name
        self.directory=file_name+'/'
        
        for _, storage in misc.dict_iter(self.dic):
            storage.set_main_path(self.directory)
            file_name=storage.get_data_path().split('/')[-1]
            storage.set_data_path(self.directory+file_name)
    

    def _garbage_collect_mpi(self):
        with Barrier():
            if comm.rank()==0:
                self._garbage_collect()
                
            
    def garbage_collect(self):
        if comm.is_mpi_used():
            self._garbage_collect_mpi()
        else:
            self._garbage_collect()
            
    def _garbage_collect(self):
        
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

    def save_fig(self, *args):
        if comm.is_mpi_used():
            with Barrier():
                if comm.rank()==0:
                    self._save_fig(*args)
        else:
            self._save_fig(*args)
                    

     
    def _save_fig(self, fig, extension='', format='svg'):
        
        with misc.Stopwatch('Saving figure...'):
            fig.savefig( self.file_name +extension+'.'+format, 
                         format = format) 
    
    def save_figs(self, figs, extension='', format='svg', in_folder='', dpi=None):
        
        path='/'.join(self.file_name .split('/')[0:-1])+'/'
        name=self.file_name .split('/')[-1]
        
        if in_folder:
            path+=in_folder+'/'
            
        if not os.path.isdir(path):
            mkdir(path)
        
        with misc.Stopwatch('Saving figures...'):
            for i, fig in enumerate(figs):
                if not dpi:
                    fig.savefig( path+name +extension+'_'+str(i)+'.'+format, 
                                 format = format)    
                elif dpi:
                    fig.savefig( path+name +extension+'_'+str(i)+'.'+format, 
                                 format = format,
                                 dpi=dpi)                            
    
    def set_file_name(self, val):
        self.file_name=val

    def set_directory(self, val):
        self.directory=val
    
    @classmethod
    def load(cls, file_name, nets=None):
        '''It makes perfect sense that you should use foo=Foo.load(), 
        and not foo=Foo();foo.load(). for example, if Foo has some 
        variables that MUST be passed in to the init, you would need 
        to make them up for foo=Foo(); or if the init does some heavy 
        calculations of variables stored in the instance, 
        it would be for nothing. '''
        
        if os.path.isfile(file_name+'.pkl'):   

                
#             cls.file_name=file_name
#             cls.directory=file_name+'/'   
            d= pickle_load(file_name)


            if nets:
                for k in d.keys():
                    if k in nets:
                        continue
                    d.delete(k)
                    
            d.force_update(file_name)

            return d
        else:
            return Storage_dic(file_name)


    @classmethod
    def load_mpi(cls, file_name, nets):
        '''It makes perfect sense that you should use foo=Foo.load(), 
        and not foo=Foo();foo.load(). for example, if Foo has some 
        variables that MUST be passed in to the init, you would need 
        to make them up for foo=Foo(); or if the init does some heavy 
        calculations of variables stored in the instance, 
        it would be for nothing. '''
        
        if os.path.isfile(file_name+'.pkl'):   

                
#             cls.file_name=file_name
#             cls.directory=file_name+'/'
            out={}
            for net in nets:   
                file_name=file_name+'/'+net
                d= pickle_load(file_name) 
                d.force_update(file_name)
                misc.dict_update(out, d)
            return out
        else:
            return Storage_dic(file_name)
                    
#     @classmethod
    def load_dic(self, *filt):
              
        d={}
        for keys, storage in misc.dict_iter(self):
            if filt==():
                pass
            else:
                a=False
                i=0
                for key in keys:
                    if key not in filt:
                        a=True
                    i+=1
                    if i==len(keys):
                        break
                if a:
                    continue
                       
            val=storage.load_data()
            d=misc.dict_recursive_add(d, keys, val)
                
        return d
            
    
    def save(self):
        pickle_save(self, self.file_name)
    
    def save_dic(self, d, **k):
        print 'Saving '+self.file_name
        for keys, data in misc.dict_iter(d):
            
            if not misc.dict_haskey(self.dic, keys):
                self.add_storage(keys)
        
            s=misc.dict_recursive_get(self.dic, keys)
            s.save_data('-'.join(keys), data, **k)
        
        self.save()
        


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
    
    
    
def get_full_file_name(fileName, file_extension='' ):
    
    
#     fileName=fileName.split('/')
#     if  '~' in fileName:
#         fileName[fileName.index('~')]=expanduser("~")
#     fileName='/'.join(fileName)    
# 
#     if 4<len(fileName) and fileName[-4:]!=file_extension:
#         fileName=fileName+file_extension

    if 4<len(fileName) and fileName[-len(file_extension):]!=file_extension:
        fileName=fileName+file_extension
    fileName=os.path.expanduser(fileName)
    
    return fileName    
 



def make_bash_script(path_bash0, path_bash, **kwargs):
    s=text_load(path_bash0)
    s=s.format(**kwargs)
    text_save(s, path_bash)
    p=subprocess.Popen(['chmod', '777',path_bash])
    p.wait()

def _mkdir(path):
    
    # If a directory does not exist where a file is suppose to be stored  it is created    
    path=path.split('/')
    i=len(path)
    while not os.path.isdir('/'.join(path[0:i])):
        i-=1
        
    while i!=len(path):
        if not os.path.isdir('/'.join(path[0:i+1])):
            os.mkdir('/'.join(path[0:i+1])) 
        i+=1

def _mkdir_mpi(*args):
    with Barrier():
        if comm.rank()==0:
            _mkdir(*args)
         
#     comm.barrier()
     
def mkdir(*args, **kwargs):
    '''
     
    Arguments:
        data        - data to be pickled
        fileName    - full path or just file name
    '''
     
    if comm.is_mpi_used() and not kwargs.get('all_mpi', False):
        _mkdir_mpi(*args)
    else:
        _mkdir(*args)
     

             
def _pickle_save(data, fileName):

    f=open(fileName, 'wb') #open in binary mode
    
    # With -1 pickle the list using the highest protocol available (binary).
    pickle.dump(data, f, -1)
    #cPickle.dump(data, f, -1)
    
    f.close()

def _pickle_save_mpi(*args):
    with Barrier():
        if comm.rank()==0:
            # OBS!!! watch out for having barriers inside here.
            # Will stall program
            _pickle_save(*args)
        
#     comm.barrier()
    
def pickle_save(data, fileName, file_extension='.pkl', **kwargs):
    '''
    
    Arguments:
        data        - data to be pickled
        fileName    - full path or just file name
    '''

    fileName=get_full_file_name(fileName, file_extension)  
    mkdir('/'.join(fileName.split('/')[0:-1])) 
    
    if comm.is_mpi_used() and not kwargs.get('all_mpi', False):
        _pickle_save_mpi(data, fileName)
    else:
        _pickle_save(data, fileName)
    
def _pickle_load_mpi(f):
    with Barrier():
        if comm.rank()==0:
            data=pickle.load(f)
        else:
            data=None
    
    with Barrier():        
        data=comm.bcast(data, root=0)
    
    return data
            
 
    
def pickle_load(fileName, file_extension='.pkl', **kwargs):
    '''
    
    Arguments:
        fileName    - full path or just file name
    '''
#     if 4<len(fileName) and fileName[-4:]!=file_extension:
#         fileName=fileName+file_extension
#     fileName=os.path.expanduser(fileName)

    fileName=get_full_file_name(fileName, file_extension)      
    f=open(fileName, 'rb') # make sure file are read in binary mod
    
    if comm.is_mpi_used()and not kwargs.get('all_mpi', False):
        data=_pickle_load_mpi(f)
    else:  
        data=pickle.load(f)

    f.close()
    return data



def _txt_save(text, fileName):
        
    f=open(fileName, 'wb') #open in binary mode
     
    f.write(text)
    f.close()    

def _txt_save_mpi(*args):
    with Barrier():
        if comm.rank()==0:
            _txt_save(*args)
        
def txt_save(text, fileName, file_extension='.txt', **kwargs ):
    '''
    
    Arguments:
        data        - data to be pickled
        fileName    - full path or just file name
    '''
       
    fileName=get_full_file_name(fileName, file_extension)  
    mkdir('/'.join(fileName.split('/')[0:-1])) 

    if comm.is_mpi_used() and not kwargs.get('all_mpi', False):
        _txt_save_mpi(text, fileName)
    else:
        _txt_save(text, fileName)

def text_save(text, fileName, **kwargs ):
    '''
    
    Arguments:
        data        - data to be pickled
        fileName    - full path or just file name
    '''
        
    mkdir('/'.join(fileName.split('/')[0:-1])) 

    if comm.is_mpi_used() and not kwargs.get('all_mpi', False):
        _text_save_mpi(text, fileName)
    else:
        _text_save(text, fileName)

def _text_save_mpi(*args):
    with Barrier():
        if comm.rank()==0:
            _text_save(*args)

def _text_save(data, fileName):
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
    
def _text_load(fileName):
    '''
    
    Arguments:
        data        - data to be pickled
        fileName    - full path or just file name
    '''
#     print fileName
    f=open(fileName, 'rb')
    data=f.read()
    f.close()
    return data

def _text_load_mpi(f):
    with Barrier():
        if comm.rank()==0:
            data=_text_load(f)
        else:
            data=None
    
    with Barrier():        
        data=comm.bcast(data, root=0)
    
    return data

def text_load(fileName, **kwargs):
    '''
    
    Arguments:
        fileName    - full path or just file name
    '''

    if comm.is_mpi_used()and not kwargs.get('all_mpi', False):
        data=_text_load_mpi(fileName)
    else:  
        data=_text_load(fileName)


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


class TestModuleFunctions(unittest.TestCase):     
    def setUp(self):
        self.data=[('s'), (1)]
    
    def test_picke_save_MPI(self):
        import subprocess
        s = expanduser("~")
        data_path= s+'/results/unittest/data_to_disk/save_dic_MPI/'
        script_name=os.getcwd()+'/test_scripts_MPI/data_to_disk_save_dic_mpi.py'
        threads=4
        p=subprocess.Popen(['mpirun', '-np', str(threads),  'python', 
                         script_name, data_path, str(self.data)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
         
        out, err = p.communicate()
#         print out
#         print err
         
        self.assertTrue(os.path.isfile(data_path+'data.pkl'))

    def test_picke_load_MPI(self):
        import subprocess
        s = expanduser("~")
        data_path= s+'/results/unittest/data_to_disk/pickle_load_MPI/'
        script_name=os.getcwd()+'/test_scripts_MPI/data_to_disk_pickle_load_mpi.py'
    
    
        pickle_save([1],data_path+'data')
        threads=2
    
        p=subprocess.Popen(['mpirun', '-np', str(threads), 'python', 
                         script_name, data_path],
#                         stdout=subprocess.PIPE,
#                         stderr=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        )
        
        out, err = p.communicate()
#         print out
#         print err
        for i in range(threads):
            self.assertTrue(os.path.isfile(data_path+'data'+str(i)+'.pkl'))


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
            s.save_data(name, data, use_hash=True)
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
        self.s.save_dic(d1, use_hash=True)
        self.s.save_dic(d2, use_hash=True)
        
        d3=self.s.load_dic()
        
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
        
    def test_save_fig(self):
        import pylab
        fig=pylab.figure()
        self.s=Storage_dic(self.file_name)
        self.s.save_fig(fig)
        self.assertTrue(os.path.isfile(self.file_name+'.svg'))
        os.remove(self.file_name+'.svg')  

    def test_save_figs(self):
        import pylab
        fig=pylab.figure()
        self.s=Storage_dic(self.file_name)
        self.s.save_figs([fig])
        self.assertTrue(os.path.isfile(self.file_name+'.svg'))
        os.remove(self.file_name+'.svg') 

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
                         TestModuleFunctions,
                        TestStorage,
                        TestStorage_dic
                        ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)  