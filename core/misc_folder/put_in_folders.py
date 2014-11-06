import os
import sys
import shutil 
files=os.listdir('.')

home=/'home
folder='/
n_folders=int(sys.argv[1])

for i in range(n_folders):
	if not os.path.isdir(folder+str(i)):
		os.mkdir(folder+str(i))
	else:
		fs=os.listdir(folder+str(i))
		for f in fs: 
			print f
			os.remove(folder+str(i)+'/'+f)

for f in files:
	print f 
	s=-1
	if len(f)>5:
		s=f[-5]
	if s in [str(i) for i in range(n_folders)]:	
		shutil.copy2(f,folder+s)

