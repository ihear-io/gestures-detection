import os, shutil
from numpy import load
import numpy as np, random, os
colors=os.listdir()

#Split data into train (80%) and test(20%)
#train
i=0
for c in colors:
  i=0
  files=os.listdir(c)
  print (len([name for name in files]))
 # if not os.path.exists('train/'+c):
  #  os.makedirs('train/'+c)
  #if not os.path.exists('test/'+c):
   # os.makedirs('test/'+c)
  for f in files:
    if (i<16):
      print(c+'/'+f)
      shutil.move(c+'/'+f, 'train/'+c)
      i+=1
    if (i>16):
      print('test', c+'/'+f)
      shutil.move(c+'/'+f, 'test/'+c)
      i+=1
     
#convert npz to np
from numpy import load
import numpy as np, random, os


colors=os.listdir()
for c in colors:
            files=os.listdir(c)

            for f in files:
              if (f!='train' and f!='test'):
                      path=c+'/'+f

                      data = load(path)
                      lst = data.files
                      print(f)


                      #store right hand key points only (for now)
                      newkr=[]
                      newkr.append(data['arr_1'])
                      newkr=np.array(newkr)
                      name=c+'/'+f.split('.')[0]
                      np.save(name, newkr)
  
  
  #delete npz

#HERE
colors=os.listdir()
for c in colors:
        
           
            files=os.listdir(c)

            for f in files:
              if (f!='train' and f!='test'):
                      path=c+'/'+f
                      if (f.find('.npz')>-1):
                        print(f)
                        os.remove(path)
