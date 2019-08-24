import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

DATADIR = "train"

CATEGORIES = ["ابيض" , "احمر","اخضر","ازرق","اسود","اصفر","برتقاني","بمبي","بني","بيج"]

#print( CATEGORIES.index("Eswed"))

training_data = []

def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the class

        for img in os.listdir(path):  # iterate over each image 
            try:
                keyp_arr=np.load(os.path.join(path,img))
                training_data.append([keyp_arr, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))

#shuffle data

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])
    
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X)
y=np.array(y)


print(X.shape)
#(160, 1, 21, 3)
print(y.shape)
#(160,)

#Let's save this data, so that we don't need to keep calculating it every time we want to play with the neural network model:

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = np.reshape(X, (-1, 21,3, 1))


#Same for test
DATADIR = "test"

testing_data = []

def create_testing_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  # create path to classes
       # print(path)
        class_num = CATEGORIES.index(category)  # get the class

        for img in os.listdir(path):  # iterate over each image 
            try:
                keyp_arr=np.load(os.path.join(path,img))
                testing_data.append([keyp_arr, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_testing_data()
print(len(testing_data))
#40

random.shuffle(testing_data)
for sample in testing_data[:10]:
    print(sample[1])
    
    
X_test = []
y_test = []

for features,label in testing_data:
    X_test.append(features)
    y_test.append(label)
X_test = np.array(X_test).reshape(-1, 21,3, 1)
y_test=np.array(y_test)

print(X_test.shape)
#(40, 21, 3, 1)
print(y_test.shape)
#(40,)

pickle_out = open("X_test.pickle","wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

