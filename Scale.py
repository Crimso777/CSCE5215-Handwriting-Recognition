import numpy as np
import pandas as pd
import segment5 as s5

import cv2
train= pd.read_csv ("Datasets/Handwritten Characters2/emnist-byclass-train.csv", header = None)
test= pd.read_csv ("Datasets/Handwritten Characters2/emnist-byclass-test.csv", header = None)

Y_train = train.drop(train.columns[1:], axis=1)
X_train = train.drop(train.columns[0], axis=1).to_numpy()

length = len(X_train)
X_train = X_train.reshape((length, 28,28))

Y_test = test.drop(test.columns[1:], axis=1)
X_test = test.drop(test.columns[0], axis=1).to_numpy()

length = len(X_test)
X_test = X_test.reshape((length, 28,28))

X_train = X_train.astype('uint8')
X_test = X_test.astype('uint8')

for i in range(len(X_train)):
   X_train[i] = s5.process(X_train[i])
for i in range(len(X_test)):
   X_test[i] = s5.process(X_test[i])


###Code below this point was taken from https://www.geeksforgeeks.org/convolutional-neural-network-cnn-in-tensorflow/
from tensorflow.keras.utils import to_categorical
    
# convert image datatype from integers to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
    
# normalising piel values
X_train = X_train/255.0
X_test = X_test/255.0
    
# reshape images to add channel dimension
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
# One-hot encoding label 
Y_train = to_categorical(Y_train, num_classes = 64)
Y_test = to_categorical(Y_test, num_classes = 64)
print(Y_train[:1])

tags = np.argmax(Y_train, axis=1)
#print(np.unique(tags))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
  
model = Sequential()
  
# Layer 1
# Conv 1
model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=1, activation = 'relu', input_shape = (28,28,1)))
# Pooling 1
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
  
# Layer 2
# Conv 2
#model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=1, activation='relu'))
# Pooling 2
#model.add(MaxPooling2D(pool_size = 2, strides = 2))
  
# Flatten
model.add(Flatten())
  
# Layer 3
# Fully connected layer 1
model.add(Dense(units=120, activation='relu'))
  
#Layer 4
#Fully connected layer 2
model.add(Dense(units=84, activation='relu'))
  
#Layer 5
#Output Layer
model.add(Dense(units=64, activation='softmax'))
  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 10
batch_size = 2048
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                    steps_per_epoch=X_train.shape[0]//batch_size, 
                    validation_data=(X_test, Y_test), 
                    validation_steps=X_test.shape[0]//batch_size, verbose = 1)
  
_, acc = model.evaluate(X_test, Y_test, verbose = 1)
print('%.3f' % (acc * 100.0))

import matplotlib.pyplot as plt  
plt.figure(figsize=(10,6))
#plt.plot(history.history['accuracy'], color = 'blue', label = 'train')
#plt.plot(history.history['val_accuracy'], color = 'red', label = 'val')
#plt.legend()
#plt.title('Accuracy')
#plt.show()

#print(model.predict_classes(X_train[:1]))
X_seg = s5.segments[0][0]
Y_seg = np.array([2, 21, 20]) 
Y_seg = to_categorical(Y_seg, num_classes = 27)

_, acc = model.evaluate(X_train, Y_train, verbose = 1)
print('%.3f' % (acc * 100.0))

input()
for word in s5.segments:
   images = word[0]
#   print(images.shape)
   images = images.astype('float32')
   images = images/255.0
#   images = images.reshape(images.shape[0], 28, 28, 1)
#   print(images.shape)
   chars = []
   pred = model.predict(images)
   chars = np.argmax(pred, axis = 1)

   print(word[1], " ", chars)
   