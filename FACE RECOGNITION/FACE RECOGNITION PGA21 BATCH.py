# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:00:21 2021

@author: soumendu
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.preprocessing import image

path=r'C:\Post Graduate Course in Data Analytics\MACHINE LEARNING\FACE RECOGNITION PGA21 BATCH\TRAIN'

datagen=ImageDataGenerator(rescale=1./255)

train_set=datagen.flow_from_directory(path,target_size=(64,64),batch_size=2,class_mode='categorical')

model=Sequential()
model.add(Conv2D(8,kernel_size=(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(8,kernel_size=(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(4,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics='acc')

history=model.fit_generator(train_set,epochs=10)

dic={}
for k,v in train_set.class_indices.items():
    dic[v]=k
    
import matplotlib.pyplot as plt

image_path=r'C:\Post Graduate Course in Data Analytics\MACHINE LEARNING\FACE RECOGNITION PGA21 BATCH\TEST\WhatsApp Image 2021-09-28 at 17.41.58 (1).jpeg'
img=image.load_img(image_path,target_size=(64,64,3))
plt.imshow(img)
plt.show()

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])
pred=model.predict(images)
label=np.argmax(pred,axis=1)
plt.title('{}'.format(dic[np.argmax(pred)]))
plt.imshow(img)

image_path=r'C:\Post Graduate Course in Data Analytics\MACHINE LEARNING\FACE RECOGNITION PGA21 BATCH\TEST\IMG_20210928_155821.jpg'
img=image.load_img(image_path,target_size=(64,64,3))
plt.imshow(img)
plt.show()

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])
pred=model.predict(images)
label=np.argmax(pred,axis=1)
plt.title('{}'.format(dic[np.argmax(pred)]))
plt.imshow(img)


image_path=r'C:\Post Graduate Course in Data Analytics\MACHINE LEARNING\FACE RECOGNITION PGA21 BATCH\TEST\9.jpeg'
img=image.load_img(image_path,target_size=(64,64,3))
plt.imshow(img)
plt.show()

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])
pred=model.predict(images)
label=np.argmax(pred,axis=1)
plt.title('{}'.format(dic[np.argmax(pred)]))
plt.imshow(img)

image_path=r'C:\Post Graduate Course in Data Analytics\MACHINE LEARNING\FACE RECOGNITION PGA21 BATCH\TEST\1632843690392.jpg'
img=image.load_img(image_path,target_size=(64,64,3))
plt.imshow(img)
plt.show()

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])
pred=model.predict(images)
label=np.argmax(pred,axis=1)
plt.title('{}'.format(dic[np.argmax(pred)]))
plt.imshow(img)
