import os
import sys
import shutil 
files=os.listdir('.')

n_folders=int(sys.argv[1])

for i in range(n_folders):
	if not os.path.isdir('./'+str(i)):
		os.mkdir('./'+str(i))
	else:
		fs=os.listdir('./'+str(i))
		for f in fs: 
			print f
			os.remove('./'+str(i)+'/'+f)

for f in files:
	print f 
	s=-1
	if len(f)>5:
		s=f[-5]
	if s in [str(i) for i in range(n_folders)]:	
		shutil.copy2(f, './'+s)

