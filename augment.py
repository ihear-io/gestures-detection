from numpy import load
import numpy as np, random, os


#New keypoints for left hend equal zero (our dataset has right hand only)
newkl=np.zeros((21,3))
colors=os.listdir('Colors')
#generate a random array with numbers from 50 to 300, 100 elements
randnums=np.random.randint(50,300,100)

for c in colors:
           n=0
           path='Colors/'+c
           data = load(path)
           lst = data.files
           #100 new images for each color
           j=0
           #generate new 100 keypoints;25 with +x shift, 25 with +x & +y shift, 25 with -x shift, 25 -x & -y shift
           for i in range(100):
                      #New keypoints for right hend
                      newkr=[]
                      for k in data['arr_1']:
                          if(i<50):
                                     #change x
                                     k[0]+=randnums[j]
                                     if(i>=25):
                                     #change y
                                                k[1]+=randnums[j]
                          
                          if (i>=50):
                                     k[0]-=randnums[j]
                          if(i>=75):
                          #change y
                                     k[1]-=randnums[j]
                          
                          newkr.append(k)
                         
                      newkr=np.array(newkr)
                      name=c.split('.')[0]+str(n)
                        #save left and right hand keypoints as npz file
                      np.savez(name, newkl,newkr)
                      j+=1
                      n+=1  
                      
                          
                          
