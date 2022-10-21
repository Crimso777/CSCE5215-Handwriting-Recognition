import numpy as np
import pandas as pd
dataset = pd.read_csv ("Datasets/Handwritten Characters2/emnist-letters-test.csv", header = None)

targets = dataset.drop(dataset.columns[1:], axis=1)
images = dataset.drop(dataset.columns[0], axis=1)

print(targets.shape)

print(images.shape)
array = images.to_numpy()
length = len(array)
array = array.reshape((length, 28,28))

print(array.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(array, targets, test_size = 0.2, random_state = 0)

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
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
  
model = Sequential()
  
# Layer 1
# Conv 1
model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=1, activation = 'relu', input_shape = (28,28,1)))
# Pooling 1
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
  
# Layer 2
# Conv 2
model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=1, activation='relu'))
# Pooling 2
model.add(MaxPooling2D(pool_size = 2, strides = 2))
  
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
model.add(Dense(units=20, activation='softmax'))
  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 50
batch_size = 512
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                    steps_per_epoch=X_train.shape[0]//batch_size, 
                    validation_data=(X_test, Y_test), 
                    validation_steps=X_test.shape[0]//batch_size, verbose = 1)
  
_, acc = model.evaluate(X_test, Y_test, verbose = 1)
print('%.3f' % (acc * 100.0))
import matplotlib.pyplot as plt  
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], color = 'blue', label = 'train')
plt.plot(history.history['val_accuracy'], color = 'red', label = 'val')
plt.legend()
plt.title('Accuracy')
plt.show()
