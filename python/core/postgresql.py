'''
Created on Feb 17, 2015

@author: mikael
'''
import cPickle
import numpy
import psycopg2
import subprocess
import time
import unittest
import warnings

from core import data_to_disk
from core.network import default_params
from core.misc import my_slice
from core import misc
from core import my_socket


import pprint
from psycopg2.errorcodes import DATATYPE_MISMATCH
pp=pprint.pprint

#ssh port forwarding
#-p 23 bcause it is ssh port 23 that we should connect ot
PORT_FORWARD_CMD=['ssh',
                  '-p',
                  '23',
                  '-L',
                  '1234:localhost:5432',
                  '-N',
                  'mikael@mikaellindahl.se']

'''
ssh -p 23 -L 1234:localhost:5432 -N mikael@mikaellindahl.se
psql -U mikael -d test -h localhost -p 1234
'''

'''
From http://stackoverflow.com/questions/13668735/
how-to-unpickle-binary-data-stored-in-postgresql-by-psycopg2-module-in-python
'''
def cast_pickle(data, cur):
    if data is None: return None
    return cPickle.loads(str(psycopg2.BINARY(data, cur)))

psycopg2.extensions.register_type(
    psycopg2.extensions.new_type(
        psycopg2.BINARY.values, 'BINARY_PICKLE', cast_pickle))

class Database(object):
    def __init__(self, name='test'):
        self.name=name
        self.computer = my_socket.determine_computer()
        self.conn=None
        self.cur=None
        self.spf=None #po4t forwarding object
    
    def __del__(self):
        self.close()
    
    def connect(self):
        if self.computer=='supermicro':
#             self.conn, _=connect_from_inside_network()
            self.conn, self.spf=connect_from_outside_network(self.name)
            self.cur=self.conn.cursor()
            
        if self.computer=='mikaellaptop':
            self.conn, self.spf=connect_from_outside_network(self.name)
            self.cur=self.conn.cursor()

        if self.computer=='milner':
            filename='/cfs/milner/scratch/l/lindahlm/.postgresql_los'
            self.conn, self.spf=connect_from_outside_network(self.name,
                                                             filename)
            self.cur=self.conn.cursor()

    def check_if_table_exist(self, table_name):
        #An alternative using EXISTS is better in that it doesn't 
        #require that all rows be retrieved, but merely that at 
        #least one such row exists:

        s="SELECT EXISTS (SELECT * FROM information_schema.tables WHERE table_name={})"
        s=s.format("'"+table_name+"'")
        self.cur.execute(s)
        return self.cur.fetchone()[0]

    
    def close(self):
        if self.cur:self.cur.close()
        if self.conn:self.conn.close()
        self.kill_spf()
        
    def commit(self):
        '''Commite changes to table'''
        self.conn.commit()

    def create_table(self, table_name, keys, datatypes, **kwargs):
        
        if self.check_if_table_exist(table_name):
            warnings.warn('Table already exist')
            return 
            
        s='CREATE TABLE {} (id serial PRIMARY KEY'.format(table_name)
        s+=', added timestamp DEFAULT CURRENT_TIMESTAMP'

    
        for key, dt in zip(keys, datatypes):
            s+=', {} {}'.format(key, dt)
        
        s+=');'
        
        self.execute(s)
        self.commit()

    def insert(self, table_name, keys, values):
        s0= "INSERT INTO {} (".format(table_name)
        s1=  ', '.join([k.lower() for k in keys])
        s2=') VALUES ('
        
        # Strings need to have '' around them
        values=["'"+str(e)+"'"  if type(e)==str  else e for e in values  ]
      
        #arrays can be  inserted as ARRAY[1,1,2] or {1,1,2}
        s3= ', '.join([str(e) if type(e)!=list  else 'ARRAY'+str(e) for e in values])
        s3=s3[:]+')'
        
        s=s0+s1+s2+s3
#         print s
        self.execute(s)
        self.commit()

    def create_table_from_dic(self, d, table_name):
        s=create_table_string(table_name, dummy_data_small())
        self.execute(s)
        self.commit()

    def execute(self, *args):
        '''Execute SQL command'''
        
        self.cur.execute(*args)
        
    def fetchall(self):
        '''To fetch results from query'''
        return self.cur.fetchall()

    def fetchone(self):
        '''To fetch results from query'''
        return self.cur.fetchone()
    
    def get_column_names(self, table):
        s=query_table_names(table)
        self.execute(s) 
        l=self.fetchall()
        l,_= zip(*l)
        
        exclude=['tableoid', 'cmax', 'xmax', 'cmin', 'xmin', 'ctid']
        l=[e for e in l if not e in exclude]
        return l

    def kill_spf(self):
        if self.spf:
            self.spf.kill()

class SSH_port_forwarding():
    '''
    setup a tunnel ssh port forwarding from the client computer
    to the computer running postgresql. Then connect ot postgresql 
    true the port.   
    '''
    
    def __init__(self, **kwargs):
        default='ssh -p 23 -L 1234:localhost:5432 -N mikael@mikaellindahl.se'
        cmd=kwargs.get('cmd',default)
  
        self.cmd=cmd.split(' ')
        self.pid=None
        
    def do(self):
        
        print 'berore cmd'
        print self.cmd
        
        p=subprocess.Popen(self.cmd,
                            stdout=subprocess.PIPE)
        print 'after cmd'
   
        time.sleep(1)
        print 'after sleep'
#         out, err=p.communicate()
        print 'SSH_port_forwarding'
#         print out, err
        self.pid=p.pid
        
    def kill(self):
        if self.pid:
            subprocess.Popen(['kill', str(self.pid)])
        
    def __del__(self):
        #Need to import other complaine subprocess is empty
        import subprocess
        if self.pid:
            subprocess.Popen(['kill', str(self.pid)])


def run_data_base_dump(run, net, script_name, net_name, category, file_name, **kwargs):
    import nest
    ks=nest.GetKernelStatus()
        
    db_name=kwargs.get('database_name','inhibition')
    db_table=kwargs.get('database_name','simulations')
    pp(ks['local_num_threads'])
    #need to supply not right picked up by nest
    lnt=kwargs.get('local_num_threads',ks['local_num_threads'])
    

    t=time.time()
    dd=run(net)

    par_data = cPickle.dumps(net.par.dic, -1)
    
    keys_db=['computer',#varchar
             'category',
             'default_params', #bytea
             'duration',          #float
             'local_num_threads', #int
             'net_name',          #varchar
             'num_processes',     #int
             'script',        #varchar
             'simulation',
             'size',              #int 
             'simulation_time',   #float 
              'total_num_virtual_procs' #int
          ]
    
    l=script_name.split('/')
    if len(l)==2:
        simulation, script=l
    if len(l)==1:
        simulation=l[0]
        script=''
    values_db=[my_socket.determine_computer(),
               category,
#                psycopg2.Binary(par_data), 
               par_data,
                float(round(time.time()-t,1)),
                int(lnt),
                net_name,
                int(ks['num_processes']),
                script,
                simulation, 
                int(net.par.dic['netw']['size']),
                float(net.par.dic['simu']['sim_time']),
                int(ks['num_processes']*lnt),
            ]
    to_binary=[False,
            False,
            True
            ]+[False]*9
    
    data=[db_name, db_table, keys_db, values_db, to_binary]
    data_to_disk.pickle_save(data, file_name +'/'+net_name,
                             file_extension= '.db_dump')
    return dd

def insert(db_name, db_table, keys_db, values_db, db):
    DB=Database
    if not db or (db and db.name!=db_name):
        db=DB(db_name)
        db.connect()
        
    db.insert(db_table, keys_db, values_db)
#     db.close()
    return db
    

def alter_table_string(table, column, dataype):
    
    s='ALTER TABLE {} ADD {} {}'
    s=s.format(table, column, dataype)
    return s

def connect(**kwargs):
    return psycopg2.connect(**kwargs )
    

def connect_from_inside_network(db_name='test'):
    '''
    inside network connecti directly
    '''
    filename='/home/mikael/.postgresql_los'
    s=data_to_disk.text_load(filename)
    s=s[:-1]
    
    kwargs={'database':db_name,
            'user':'mikael',
            'password':s,
            'host':'192.168.1.14',
            'sslmode':'require'}
    return connect(**kwargs), None

def connect_from_outside_network(db_name='test', filename='/home/mikael/.postgresql_los'):
    
    s=data_to_disk.text_load(filename)
    s=s[:-1]

    print 'SSH_port_forwarding'
    spf=SSH_port_forwarding()
    spf.do()

    kwargs={'database':db_name,
            'user':'mikael',
            'password':s,
            'host':'localhost',
            'port':'1234',
            'sslmode':'require'}
    
    print 'connect'
    return connect(**kwargs), spf

def keys_and_dataypes_from_dic(d):
    d_red={}
    misc.dict_reduce(d, d_red, deliminator='_')
    
    keys, values=zip(*d_red.items())
    
    datatypes=[]
    extract_datatypes(values, datatypes) 
    
    return keys, datatypes

def create_table_string(name,d):
    
    keys, datatypes=keys_and_dataypes_from_dic(d)
    
    s='CREATE TABLE {} (id serial PRIMARY KEY'.format(name)
    s+=', {} {}'.format('timestamp', 'timestamp')
    s+=', {} {}'.format('computer', 'varchar')
    
    for key, dt in zip(keys, datatypes):
        s+=', {} {}'.format(key, dt)
    
    s+=');'
            
    return s
#     cur.execute(s)


def drop_table_string(name):
    return 'DROP TABLE {};'.format(name)
    
    
def _extract_datatype(val):

    if type(val) == int:
        out= 'integer'
    elif type(val) == float:
        out= 'float'
    elif type(val) in str:
        out= 'varchar'
    elif type(val) == bool:
        out= 'boolean'
    else:
        out= 'varchar'
    return out

    
def extract_datatypes(values):
    datatypes=[]
    for val in values:       
        dt=_extract_datatype(val, 0)
        datatypes.append(dt)
    return datatypes


def is_port_used():
    '''
    "netstat -ln | grep ':1234 ' | grep 'LISTEN'"
    
    | means pip and can be accomplished by feeding stdout 
    in to another subprocess stdin 
    '''
    s0=['netstat', '-ln']
    s1=['grep', ":1234 "]
    s2=['grep', "LISTEN"]
    
#         p = subprocess.Popen(args, **kwargs)

    kwargs={
            'stdout':subprocess.PIPE}
    p0 = subprocess.Popen(s0, **kwargs)

    kwargs={'stdin':p0.stdout,
            'stdout':subprocess.PIPE}
    p1 = subprocess.Popen(s1, **kwargs)
    p0.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.

    kwargs={'stdin':p1.stdout,
            'stdout':subprocess.PIPE}
    p2 = subprocess.Popen(s2, **kwargs)
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.

    out, err = p2.communicate()
    if out!='':
        return True
    else: 
        return False


def extract_data_from_dic():
 
    d_red={}
    misc.dict_reduce(d, d_red, deliminator='_')
    
    # Sort on keys
    items=sorted(d_red.items(), key=lambda x:x[0])   
    keys, values=zip(*items)    

    # Convert tpyes that are not
    values=[v if type(v) in [int, str, list, bool, float, numpy.ndarray] else str(v) for v in values]
 
    return keys, values

    
def query_table_names(table):
#     s=("SELECT *FROM information_schema.columns"+
# #      "WHERE table_schema = '{schema}'",
#      " WHERE table_name   = '{}';")
    s=("SELECT attname , typname FROM pg_attribute "
    +", pg_type WHERE typrelid=attrelid AND typname = {};")
    s=s.format("'"+table+"'")
    return s  
    

def dummy_data_small(beta=0.0):
    d={'conn': {'C1_M1_ampa': {'beta_fan_in': beta,
                                 'class_surface': 'Conn',
                                 'delay': {'params': 12.0, 'type': 'constant'},
                                 'lesion': False,
                                 'mask': [-0.5, 0.5],
                                 'netw_size': 10000.0,},},

        'netw': {'FF_curve': {'input': 'C1', 'output': 'M1'},
                  'GA_prop': 0.2,
                  'GI_prop': 0.8,
                  },
        
        'node':{ 'EA': {'class_population': 'MyPoissonInput'},  
                 'GA': {'I_vitro': 5.0,},},
        'nest':{ 'GA_M2_gaba': {'delay': 1.7},
                'GI': {'AMPA_1_E_rev': 0.0,
                       'type_id': 'my_aeif_cond_exp'},},
        'simu': {'do_reset': False,
                  'local_num_threads': 1,
                  'stop_rec': 2000.0}}
    return d



class Test_database(unittest.TestCase):
    def setUp(self):
        self.db=Database()
        self.db.connect()
           
    def tearDown(self):
        self.db.close()       
        
    def test_check_if_table_exist(self):
                 
        b=self.db.check_if_table_exist('table_to_check_for_existans')
        self.assertTrue(b)
        b=self.db.check_if_table_exist('none')
        self.assertFalse(b)
        
    def test_create_table(self):
        table_name='unittest_create_table'
        keys=['a','b','c']
        datatypes=['integer', 'varchar', 'float[]']
        self.db.create_table(table_name, keys, datatypes)
        
        s="SELECT EXISTS (SELECT * FROM information_schema.tables WHERE table_name={})"
        s=s.format("'"+table_name+"'")
        self.db.cur.execute(s)
        self.assertTrue(self.db.cur.fetchone()[0])
        
        s='DROP TABLE '+table_name
        self.db.cur.execute(s)

        s="SELECT EXISTS (SELECT * FROM information_schema.tables WHERE table_name={})"
        s=s.format("'"+table_name+"'")
        self.db.cur.execute(s)
        self.assertFalse(self.db.cur.fetchone()[0])
        
    def test_insert(self):
        # CREATE TABLE
        
        # "CREATE TABLE unittets_insert_table (id serial PRIMARY KEY, a integer, b varchar, c float, d boolean, e integer[], f varchar[], g float[])" 

        
        table_name='unittest_insert_table'
#         keys=['a', 'b','c','d','e', 'f', 'g']
#         values=['integer', 'varchar', 'float','boolean',
#                 'integer[]','varchar[]', 'float[]']
#         values=['integer', 'varchar', 'float','boolean',
#                 'integer[]','varchar[]', 'float[]']                
#         self.db.insert(table_name, keys, values)

        table_name='unittest_insert_table'
        keys=['a', 'b']
        values=[1, 'hej']
                
        self.db.insert(table_name, keys, values)
    
    def test_cPickle_binary(self):
        table_name='unittest_cPickle_binary'
        
        keys=['binary_pickle']
        datatypes=['bytea']
        self.db.create_table(table_name, keys, datatypes)
        
        obj = {'a': 10}
        data = cPickle.dumps(obj, -1)
        self.db.insert(table_name, ['binary_pickle'], [psycopg2.Binary(data)])
        # import psycopg2
        
        
        
#         cnn = psycopg2.connect('')
#         cur = cnn.cursor()
        self.db.execute("select %s::bytea", [psycopg2.Binary(data)])
        print self.db.fetchone()
        # ({'a': 10},)

        

        
class ModuleFunctions(unittest.TestCase):     
    def setUp(self):
        self.db=Database()
        
    def tearDown(self):
        self.db.kill_spf()

    def test_query_table_names(self):
        with self.db:                    
            s=query_table_names('unittest_small')
            
            self.db.execute(s) 
            l=self.db.fetchall()
            l0,_= zip(*l)
        l1=('tableoid', 'cmax', 'xmax', 'cmin', 'xmin', 'ctid', 'id', 
            'timestamp', 'computer', 'nest_gi_ampa_1_e_rev', 
            'simu_stop_rec', 'netw_ga_prop', 'simu_local_num_threads', 
            'conn_c1_m1_ampa_delay_type', 'nest_gi_type_id', 
            'netw_gi_prop', 'conn_c1_m1_ampa_netw_size', 
            'conn_c1_m1_ampa_mask', 'conn_c1_m1_ampa_delay_params', 
            'conn_c1_m1_ampa_beta_fan_in', 'simu_do_reset', 
            'netw_ff_curve_input', 'nest_ga_m2_gaba_delay', 
            'netw_ff_curve_output', 'node_ga_i_vitro', 
            'conn_c1_m1_ampa_class_surface', 'node_ea_class_population',
            'conn_c1_m1_ampa_lesion')
        self.assertListEqual(sorted(l0), sorted(l1))
            
        print 'hej'



    def test_is_port_used(self):
        
        self.assertFalse(is_port_used())
        spf=SSH_port_forwarding()
        spf.do()
#         s='ssh -p 23 -L 1234:localhost:5432 -N mikael@mikaellindahl.se'
#         p=subprocess.Popen(s.split(' '))
        print spf.pid
        self.assertTrue(is_port_used())
        
        
    def test_extract_datatypes(self):
        values=[1,1.2,'a',True,[1, 2], [1., 2.], ['a','b'], [True, False], numpy.array([1,2]),
                [[1,2],[1,2]], [[1,2],[1]], numpy.array([[1,2],[1,2]])]
        print extract_datatypes(values)
if __name__ == '__main__':
    d={
       Test_database:[
#                     'test_create',
#                     'test__enter__',
#                       'test_check_if_table_exist',  
#                       'test_create_table',
#                         'test_insert',
                    'test_cPickle_binary',
                      ],
       
        ModuleFunctions:[
#                         'test_create_table_string',
#                         'test_drop_table',
#                         'test_create_table',
#                         'test_query_table_names',
#                         'test_insert_table_string',
#                         'test_insert',
#                         'test_is_port_used',
                          'test_extract_datatypes',
                         
                         
                    ],

       }
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)

# conn=psycopg2.connect("dname=test")