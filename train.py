from keras.utils import np_utils
#one hot encode labels
y = np_utils.to_categorical(y, num_classes=10).astype('float32')
y_test = np_utils.to_categorical(y_test, num_classes=10).astype('float32')

#proper shape for LSTM
X=np.reshape(X,(160,63,1))


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.layers import TimeDistributed, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, precision_score
from sklearn.metrics import recall_score, accuracy_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K

print(X.shape, y.shape)




model = Sequential()
model.add(LSTM(128, input_shape=(63,1)))
model.add(Dropout(0.5))                             
#model.add(BatchNormalization())

# model.add(LSTM(128, input_shape=(63,1)))
# model.add(Dropout(0.5))                             
# model.add(BatchNormalization())

model.add((Dense(10))) 
model.add(Activation('softmax'))
myOptimizer = Adam(lr = 0.001) 
model.compile(loss='categorical_crossentropy', optimizer=myOptimizer, metrics=['categorical_accuracy'])

# summarize model
print(model.summary())


# train model
model.fit(X, y, batch_size=4, epochs = 20) 

# evaluate
loss, acc = model.evaluate(X_test, y_test)

print("Test set accuracy = ", acc)
print("Test set loss = ", loss)

# predict
predictions = model.predict(X_test)
model.save_weights('openpose_aweosome_model.h5')

