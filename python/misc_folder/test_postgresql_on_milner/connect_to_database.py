'''
Created on Feb 25, 2015

@author: mikael
'''

# import nest
# 
# if nest.Rank()==0:
#     

from toolbox import monkey_patch as mp
mp.patch_for_milner()
from toolbox.postgresql import Database as DB
from toolbox import my_socket


keys_db=['computer',
      ]
values_db=[my_socket.determine_computer(),
           
        ]
print 'Open database'
db=DB('inhibition')
print 'Connecting'
db.connect()
print 'Inserting'
db.insert('main', keys_db, values_db)
print 'Closing'
db.close()