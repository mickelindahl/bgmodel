'''
Created on Jun 18, 2013

@author: lindahlm
'''
import csv
import nest
import numpy
import cPickle # Can be 1000 times faster than pickle
import os
from toolbox import misc

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
    
    mkdir('/'.join(fileName.split('/')[0:-1]))  
    if 4>len(fileName) or fileName[-4:]!='.'+file_extension:
        fileName=fileName+'.'+file_extension
    
    f=open(fileName,'w')
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

    mkdir('/'.join(fileName.split('/')[0:-1]))    
    if 4<len(fileName) or fileName[-4:]!='.pkl':
        fileName=fileName+'.pkl'
    f=open(fileName, 'wb') #open in binary mode
    
    
    # With -1 pickle the list using the highest protocol available (binary).
    #pickle.dump(data, f, -1)
    cPickle.dump(data, f, -1)
    
    f.close()


def pickle_save_groups(group_list, fileName):
    fileName=fileName+'-'+str(nest.Rank())
    pickle_save(group_list, fileName)

    
def pickle_load(fileName):
    '''
    
    Arguments:
        fileName    - full path or just file name
    '''
    if 4<len(fileName) and fileName[-4:]!='.pkl':
        fileName=fileName+'.pkl'
    fileName=os.path.expanduser(fileName)
    f=open(fileName, 'rb') # make sure file are read in binary mode
    #data=pickle.load(f)
    data=cPickle.load(f) 
    f.close()
    return data


def pickle_load_groups(file_name):
    path='/'.join(file_name.split('/')[0:-1])
    name=file_name.split('/')[-1]
    fileNames = read_f_name( path, contain_string=name )
    
    check_parts=pickle_load(path+'/'+fileNames[0])
    parts=[]   
    for name in fileNames:
        if isinstance(check_parts[0],list):
            parts.append(pickle_load(path+'/'+name))
        else:
            # Put it in a list
            parts.append([pickle_load(path+'/'+name)])
    
    # To put merge parts groups in 
    groups=parts[0]   

    # Find what data is recorded from    
    recorded_from=parts[0][0][0].signals.keys() 
  
    # Iterate over groups
    for j in range(len(parts[0])):
        for k in range(len(parts[0][0])) :   
            group=parts[0][j][k]
                       
            # For each recordable
            for recordable in recorded_from:                                           
                
                if 'spikes' == recordable:
                    
                    spk_signal=group.signals[recordable]
                    
                    # Iterate over parts and merge spike data
                    # Only happens for mpi data
                    
                    for i_part in range(1, len(parts[1:])+1):
                        group_k = parts[i_part][j][k]
                        add_spk_signal = group_k.signals[recordable]
                        spk_signal.merge(add_spk_signal)
                        
                    group.signals[recordable] = spk_signal
                
                # Must be analog signal    
                else:
                    
                    ag_signal=group.signals[recordable]
                    
                    # Iterate over parts and merge analog data
                    # Only happens for mpi data
                    for i_part in range(1, len(parts[1:])+1):
                        group_k = parts[i_part][j][k]
                        ag_signal_k = group_k.signals[recordable]
                        
                        for id, signal in ag_signal_k.analog_signals.iteritems():
                            ag_signal.append(id, signal)
                    
                    group.signals[recordable] = ag_signal
                    
                groups[j][k]=group
    
    return groups         


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