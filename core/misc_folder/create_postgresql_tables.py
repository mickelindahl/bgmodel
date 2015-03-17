'''
Created on Feb 24, 2015

@author: mikael
'''
from toolbox import postgresql as psql

def create_table_main():
    db=psql.Database('inhibition')
    db.connect()
    table_name='simulations'
    keys=['computer', 
          'category',
          'default_params', 
          'duration',                       
          'local_num_threads', #int
          'net_name',
          'num_processes',     #int
          'script',
          'simulation',
          'size',              #int 
          'simulation_time',   #float 
          'total_num_virtual_procs',
          ]
    datatypes=['varchar', 
               'varchar',
               'bytea', 
               'float',
               'int',
               'varchar',
               'int',
               'varchar',
               'varchar',
               'int',
               'float',
               'int'
               ]
    db.create_table(table_name, keys, datatypes)    
    db.close()

create_table_main()